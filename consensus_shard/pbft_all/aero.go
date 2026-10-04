package pbft_all

import (
	"blockEmulator/aero"
	"blockEmulator/consensus_shard/pbft_all/dataSupport"
	"blockEmulator/message"
	"blockEmulator/networks"
	"blockEmulator/params"
	"encoding/json"
	"log"
	"reflect"
	"sync"
	"time"
)

type aeroLogicalModule struct {
	mu               sync.Mutex
	node             *PbftConsensusNode
	data             *dataSupport.Data_supportCLPA
	identity         *aero.Identity
	worker           *aero.Worker
	round            *aero.Round
	snapshot         aero.Snapshot
	plan             aero.Plan
	transfer         map[string]uint64
	outgoing         map[string]aero.Envelope
	appliedEpoch     uint64
	migrationStarted uint64
	roundStarted     time.Time
}

func newAEROLogicalModule(p *PbftConsensusNode, d *dataSupport.Data_supportCLPA) *aeroLogicalModule {
	identity, e := aero.LoadIdentity(params.AERO.KeyDir, p.ShardID*p.node_nums+p.NodeID)
	if e != nil {
		log.Fatal("AERO keys: ", e)
	}
	w, e := aero.StartWorker(params.AERO)
	if e != nil {
		log.Fatal(e)
	}
	m := &aeroLogicalModule{node: p, data: d, identity: identity, worker: w, outgoing: make(map[string]aero.Envelope)}
	go func() {
		tick := time.NewTicker(2 * time.Second)
		defer tick.Stop()
		defer w.Close()
		for !p.stopSignal.Load() {
			<-tick.C
			m.mu.Lock()
			if m.round != nil && !m.round.Proposed && time.Since(m.roundStarted) > 5*time.Second {
				v := m.identity.SignView("viewchange", m.round.Epoch, m.round.View+1, m.round.Digest, nil)
				m.outgoing["viewchange"] = v
			}
			for _, e := range m.outgoing {
				m.broadcast(e)
			}
			if m.round != nil && m.round.Committed {
				m.sendSupervisor(m.identity.Sign("decided", m.round.Epoch, m.round.Digest, m.plan))
				if m.appliedEpoch == m.round.Epoch {
					m.sendSupervisor(m.identity.Sign("applied", m.round.Epoch, m.round.Digest, nil))
				}
			}
			m.mu.Unlock()
		}
	}()
	return m
}
func (m *aeroLogicalModule) broadcast(e aero.Envelope) {
	b, _ := json.Marshal(e)
	msg := message.MergeMessage(message.CAERO, b)
	for s := uint64(0); s < m.node.pbftChainConfig.ShardNums; s++ {
		for n := uint64(0); n < m.node.node_nums; n++ {
			networks.TcpDial(msg, m.node.ip_nodeTable[s][n])
		}
	}
}
func (m *aeroLogicalModule) sendSupervisor(e aero.Envelope) {
	b, _ := json.Marshal(e)
	networks.TcpDial(message.MergeMessage(message.CAERO, b), params.SupervisorAddr)
}
func (m *aeroLogicalModule) handle(content []byte) {
	var e aero.Envelope
	if json.Unmarshal(content, &e) != nil || !m.identity.Verify(e) {
		return
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	members := m.node.node_nums * m.node.pbftChainConfig.ShardNums
	if e.Stage == "input" {
		if e.Sender != members || (m.round != nil && e.Epoch <= m.round.Epoch) {
			return
		}
		if (m.round == nil && e.Epoch != 1) || (m.round != nil && (!m.round.Committed || e.Epoch != m.round.Epoch+1)) {
			return
		}
		var s aero.Snapshot
		if json.Unmarshal(e.Payload, &s) != nil || aero.Digest(s) != e.Digest || s.Epoch != e.Epoch {
			return
		}
		if s.Shards != int(m.node.pbftChainConfig.ShardNums) || s.PrefixDigits != params.AERO.PrefixDigits || s.MaxMoves != params.AERO.MaxMoves {
			return
		}
		if s.Validate(params.AERO, int(m.node.pbftChainConfig.ShardNums)) != nil {
			return
		}
		// Input is the emulator's agreed epoch record, authenticated by its supervisor.
		plan, err := m.worker.Plan(s)
		if err != nil {
			log.Panic("AERO inference failed: ", err)
		}
		transfer, err := aero.Expand(s, plan)
		if err != nil {
			log.Panic(err)
		}
		m.snapshot, m.plan, m.transfer = s, plan, transfer
		m.round = aero.NewRound(s.Epoch, aero.PlanDigest(s, plan), members)
		m.roundStarted = time.Now()
		m.outgoing = make(map[string]aero.Envelope)
		if m.identity.ID == (s.Epoch-1)%members {
			proposal := m.identity.Sign("proposal", s.Epoch, m.round.Digest, nil)
			m.outgoing["proposal"] = proposal
			m.broadcast(proposal)
		}
		return
	}
	if m.round == nil {
		return
	}
	oldView := m.round.View
	prepare, commit, done := m.round.Accept(e)
	if m.round.View > oldView {
		m.roundStarted = time.Now()
		delete(m.outgoing, "viewchange")
		if m.identity.ID == (m.round.Epoch-1+m.round.View)%members {
			v := m.identity.SignView("proposal", m.round.Epoch, m.round.View, m.round.Digest, nil)
			m.outgoing["proposal"] = v
			m.broadcast(v)
		}
	}
	if prepare {
		v := m.identity.Sign("prepare", e.Epoch, m.round.Digest, nil)
		m.outgoing["prepare"] = v
		m.broadcast(v)
	}
	if commit {
		v := m.identity.Sign("commit", e.Epoch, m.round.Digest, nil)
		m.outgoing["commit"] = v
		m.broadcast(v)
	}
	if done {
		m.sendSupervisor(m.identity.Sign("decided", e.Epoch, m.round.Digest, m.plan))
		m.node.pl.Plog.Printf("AERO logical epoch %d committed %d prefix moves\n", e.Epoch, len(m.plan.Moves))
	}
}

// The physical proposal goroutine owns the migration data. Logical consensus
// exposes its decision under a mutex instead of mutating CSTM concurrently.
func (m *aeroLogicalModule) beginPartition() {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.round != nil && m.round.Committed && m.migrationStarted != m.round.Epoch && m.appliedEpoch < m.round.Epoch {
		m.data.ModifiedMap = append(m.data.ModifiedMap, m.transfer)
		m.data.PartitionOn = true
		m.migrationStarted = m.round.Epoch
	}
}
func (m *aeroLogicalModule) validTransfer(atm *message.AccountTransferMsg) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.round != nil && m.round.Committed && atm.ATid == m.round.Epoch && reflect.DeepEqual(atm.ModifiedMap, m.transfer)
}
func (m *aeroLogicalModule) applied(epoch uint64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.round != nil && epoch == m.round.Epoch {
		m.appliedEpoch = epoch
		m.sendSupervisor(m.identity.Sign("applied", epoch, m.round.Digest, nil))
	}
}
