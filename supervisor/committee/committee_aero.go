package committee

import (
	"blockEmulator/aero"
	"blockEmulator/core"
	"blockEmulator/message"
	"blockEmulator/networks"
	"blockEmulator/params"
	"blockEmulator/supervisor/signal"
	"blockEmulator/supervisor/supervisor_log"
	"encoding/csv"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"sync"
	"time"
)

type AEROCommittee struct {
	mu            sync.Mutex
	ip            map[uint64]map[uint64]string
	signal        *signal.StopSignal
	logger        *supervisor_log.SupervisorLog
	identity      *aero.Identity
	epoch         uint64
	assignment    []uint64
	history       []aero.Move
	known         map[string]bool
	pairs         [][2]string
	seen          map[string]bool
	blocks        []int
	previous      aero.Stats
	started       time.Time
	pending       *aero.Snapshot
	plan          *aero.Plan
	digest        string
	decided       map[uint64]bool
	applied       map[uint64]bool
	rewardWeights *aero.RewardSummary
}

func NewAEROCommittee(ip map[uint64]map[uint64]string, ss *signal.StopSignal, sl *supervisor_log.SupervisorLog) *AEROCommittee {
	members := uint64(params.ShardNum * params.NodesInShard)
	identity, e := aero.LoadIdentity(params.AERO.KeyDir, members)
	if e != nil {
		log.Fatal("AERO supervisor identity: ", e)
	}
	a := &AEROCommittee{ip: ip, signal: ss, logger: sl, identity: identity,
		assignment: make([]uint64, 1<<(4*params.AERO.PrefixDigits)), known: make(map[string]bool),
		blocks: make([]int, params.ShardNum), seen: make(map[string]bool), started: time.Now()}
	for p := range a.assignment {
		a.assignment[p] = uint64(p % params.ShardNum)
	}
	a.previous = aero.Statistics(nil, a.assignment, params.ShardNum, params.AERO.PrefixDigits, 1)
	return a
}
func (a *AEROCommittee) HandleBlockInfo(b *message.BlockInfoMsg) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if b.SenderShardID >= uint64(params.ShardNum) || b.Epoch != int(a.epoch) {
		return
	}
	a.blocks[b.SenderShardID]++
	txs := append(append([]*core.Transaction{}, b.InnerShardTxs...), b.Relay2Txs...)
	for _, tx := range txs {
		hash := hex.EncodeToString(tx.TxHash)
		if a.seen[hash] {
			continue
		}
		a.seen[hash] = true
		a.pairs = append(a.pairs, [2]string{tx.Sender, tx.Recipient})
	}
}
func (a *AEROCommittee) HandleOtherMessage(msg []byte) {
	kind, content := message.SplitMessage(msg)
	if kind != message.CAERO {
		return
	}
	var e aero.Envelope
	if json.Unmarshal(content, &e) != nil || !a.identity.Verify(e) {
		return
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.pending == nil || e.Epoch != a.pending.Epoch || e.Sender >= uint64(params.ShardNum*params.NodesInShard) {
		return
	}
	if e.Stage == "decided" {
		var plan aero.Plan
		if json.Unmarshal(e.Payload, &plan) != nil {
			return
		}
		if _, err := aero.Expand(*a.pending, plan); err != nil {
			return
		}
		if aero.PlanDigest(*a.pending, plan) != e.Digest {
			return
		}
		if a.plan == nil {
			a.plan = &plan
			a.digest = e.Digest
		}
		if e.Digest == a.digest {
			a.decided[e.Sender] = true
		}
	} else if e.Stage == "applied" {
		// A node signs only after its physical PBFT has committed the state transfer.
		// Applied can arrive before decided; its digest is checked when counting below.
		if a.plan != nil && e.Digest == a.digest {
			a.applied[e.Sender] = true
		}
	}
}
func (a *AEROCommittee) due() bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	if params.ShardNum <= 1 || len(a.pairs) == 0 {
		return false
	}
	for _, n := range a.blocks {
		if n < params.AERO.EpochBlocks {
			return false
		}
	}
	return true
}
func (a *AEROCommittee) broadcast(e aero.Envelope) {
	b, _ := json.Marshal(e)
	msg := message.MergeMessage(message.CAERO, b)
	for s := 0; s < params.ShardNum; s++ {
		for n := 0; n < params.NodesInShard; n++ {
			networks.TcpDial(msg, a.ip[uint64(s)][uint64(n)])
		}
	}
}
func (a *AEROCommittee) reconfigure() {
	a.mu.Lock()
	addresses := make([]string, 0, len(a.known))
	for address := range a.known {
		addresses = append(addresses, address)
	}
	sort.Strings(addresses)
	stats := aero.Statistics(a.pairs, a.assignment, params.ShardNum, params.AERO.PrefixDigits, time.Since(a.started).Seconds())
	s := aero.Snapshot{Epoch: a.epoch + 1, Seed: params.AERO.Seed + int64(a.epoch+1), Shards: params.ShardNum,
		PrefixDigits: params.AERO.PrefixDigits, MaxMoves: params.AERO.MaxMoves, Assignment: append([]uint64{}, a.assignment...),
		History: append([]aero.Move{}, a.history...), Current: stats, Previous: a.previous, Addresses: addresses}
	s.HistoryWindow = params.AERO.HistoryWindow
	a.pending = &s
	a.plan = nil
	a.digest = ""
	a.decided = make(map[uint64]bool)
	a.applied = make(map[uint64]bool)
	input := a.identity.Sign("input", s.Epoch, aero.Digest(s), s)
	a.mu.Unlock()
	a.logger.Slog.Printf("AERO epoch %d: pausing injection and proposing migration\n", s.Epoch)
	a.broadcast(input)
	deadline := time.Now().Add(time.Duration(params.AERO.TimeoutSeconds) * time.Second)
	lastSend := time.Now()
	for {
		a.mu.Lock()
		members := params.ShardNum * params.NodesInShard
		quorum := members - (members-1)/3
		ready := a.plan != nil && len(a.decided) >= quorum
		shardQuorum := params.NodesInShard - (params.NodesInShard-1)/3
		for shard := 0; shard < params.ShardNum; shard++ {
			count := 0
			for node := 0; node < params.NodesInShard; node++ {
				if a.applied[uint64(shard*params.NodesInShard+node)] {
					count++
				}
			}
			if count < shardQuorum || !a.applied[uint64(shard*params.NodesInShard)] {
				ready = false
			}
		}
		if ready {
			if a.plan.Reward == nil {
				log.Panic("AERO worker returned no reward components")
			}
			a.rewardWeights = a.plan.Reward
			for _, move := range a.plan.Moves {
				a.assignment[move.Prefix] = move.Receiver
				a.history = append(a.history, move)
			}
			if len(a.history) > params.AERO.HistoryWindow {
				a.history = a.history[len(a.history)-params.AERO.HistoryWindow:]
			}
			a.writeEpoch(s, *a.plan, a.digest)
			a.writeReward(s.Epoch-1, *a.plan.Reward, completedTransfers(s.Current), false)
			a.epoch = s.Epoch
			a.previous = stats
			a.pairs = nil
			a.seen = make(map[string]bool)
			a.blocks = make([]int, params.ShardNum)
			a.started = time.Now()
			a.pending = nil
			a.signal.StopGap_Reset()
			a.mu.Unlock()
			break
		}
		a.mu.Unlock()
		if time.Now().After(deadline) {
			log.Panicf("AERO epoch %d timed out waiting for logical consensus / every shard's migration", s.Epoch)
		}
		if time.Since(lastSend) > 2*time.Second {
			a.broadcast(input)
			lastSend = time.Now()
		}
		time.Sleep(100 * time.Millisecond)
	}
	a.logger.Slog.Printf("AERO epoch %d: state transfer finished in every shard; resuming injection\n", s.Epoch)
}
func (a *AEROCommittee) writeEpoch(s aero.Snapshot, p aero.Plan, digest string) {
	if e := os.MkdirAll(params.DataWrite_path, 0755); e != nil {
		log.Panic(e)
	}
	f, e := os.OpenFile(filepath.Join(params.DataWrite_path, "aero_epochs.jsonl"), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644)
	if e != nil {
		log.Panic(e)
	}
	defer f.Close()
	e = json.NewEncoder(f).Encode(struct {
		Snapshot      aero.Snapshot       `json:"snapshot"`
		Plan          aero.Plan           `json:"plan"`
		Digest        string              `json:"digest"`
		ObservedEpoch uint64              `json:"observed_epoch"`
		Reward        *aero.RewardSummary `json:"reward"`
	}{s, p, digest, s.Epoch - 1, p.Reward})
	if e != nil {
		log.Panic(e)
	}
}

func completedTransfers(s aero.Stats) int {
	total := 0.
	for _, row := range s.Cross {
		for _, v := range row {
			total += v / 2
		}
	}
	for _, row := range s.Intra {
		for _, v := range row {
			total += v
		}
	}
	return int(total)
}

func (a *AEROCommittee) writeReward(epoch uint64, r aero.RewardSummary, completed int, final bool) {
	if e := os.MkdirAll(params.DataWrite_path, 0755); e != nil {
		log.Panic(e)
	}
	path := filepath.Join(params.DataWrite_path, "aero_rewards.csv")
	f, e := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0644)
	if e != nil {
		log.Panic(e)
	}
	defer f.Close()
	info, e := f.Stat()
	if e != nil {
		log.Panic(e)
	}
	writer := csv.NewWriter(f)
	if info.Size() == 0 {
		if e = writer.Write([]string{"epoch", "action_epoch", "reward", "cross_ratio", "cross_variance", "normalized_cross_variance", "cross_weight", "variance_weight", "completed_transfers", "used_for_online_feedback", "final_epoch"}); e != nil {
			log.Panic(e)
		}
	}
	actionEpoch := ""
	if epoch > 0 {
		actionEpoch = strconv.FormatUint(epoch, 10)
	}
	format := func(v float64) string { return strconv.FormatFloat(v, 'g', 17, 64) }
	e = writer.Write([]string{strconv.FormatUint(epoch, 10), actionEpoch, format(r.Reward), format(r.CrossRatio), format(r.CrossVariance), format(r.NormalizedCrossVariance), format(r.CrossWeight), format(r.VarianceWeight), strconv.Itoa(completed), strconv.FormatBool(params.AERO.OnlineTraining && epoch > 0 && !final), strconv.FormatBool(final)})
	if e != nil {
		log.Panic(e)
	}
	writer.Flush()
	if e = writer.Error(); e != nil {
		log.Panic(e)
	}
	a.logger.Slog.Printf("AERO reward epoch %d: reward=%.8f cross_ratio=%.8f normalized_variance=%.8f final=%t\n", epoch, r.Reward, r.CrossRatio, r.NormalizedCrossVariance, final)
}

func (a *AEROCommittee) writeFinalReward() {
	a.mu.Lock()
	defer a.mu.Unlock()
	stats := aero.Statistics(a.pairs, a.assignment, params.ShardNum, params.AERO.PrefixDigits, time.Since(a.started).Seconds())
	var r aero.RewardSummary
	if a.rewardWeights != nil {
		r = aero.EvaluateReward(stats, a.rewardWeights.CrossWeight, a.rewardWeights.VarianceWeight)
	} else {
		// Short traces can finish without an epoch migration. Still read the actual
		// checkpoint weights instead of assuming both reward coefficients are one.
		worker, e := aero.StartWorker(params.AERO)
		if e != nil {
			log.Panic(e)
		}
		var err error
		r, err = worker.MeasureReward(stats)
		worker.Close()
		if err != nil {
			log.Panic(err)
		}
	}
	a.writeReward(a.epoch, r, len(a.pairs), true)
}
func (a *AEROCommittee) sendBatch(txs []*core.Transaction) {
	for start := 0; start < len(txs); start += params.InjectSpeed {
		end := start + params.InjectSpeed
		if end > len(txs) {
			end = len(txs)
		}
		batches := make([][]*core.Transaction, params.ShardNum)
		a.mu.Lock()
		for _, tx := range txs[start:end] {
			a.known[tx.Sender] = true
			a.known[tx.Recipient] = true
			p, e := aero.Prefix(tx.Sender, params.AERO.PrefixDigits)
			if e != nil {
				log.Panic(e)
			}
			s := a.assignment[p]
			batches[s] = append(batches[s], tx)
		}
		a.mu.Unlock()
		for s, txs := range batches {
			if len(txs) == 0 {
				continue
			}
			b, _ := json.Marshal(message.InjectTxs{Txs: txs, ToShardID: uint64(s)})
			networks.TcpDial(message.MergeMessage(message.CInject, b), a.ip[uint64(s)][0])
		}
		a.signal.StopGap_Reset()
		time.Sleep(time.Second)
		if a.due() {
			a.reconfigure()
		}
	}
}
func (a *AEROCommittee) MsgSendingControl() {
	f, e := os.Open(params.DatasetFile)
	if e != nil {
		log.Panic(e)
	}
	defer f.Close()
	reader := csv.NewReader(f)
	count := 0
	batch := make([]*core.Transaction, 0, params.TxBatchSize)
	for count < params.TotalDataSize {
		row, e := reader.Read()
		if e == io.EOF {
			break
		}
		if e != nil {
			log.Panic(e)
		}
		if len(row) < 9 {
			continue
		}
		tx, ok := data2tx(row, uint64(count))
		if !ok {
			continue
		}
		if _, e := aero.Prefix(tx.Sender, params.AERO.PrefixDigits); e != nil {
			continue
		}
		if _, e := aero.Prefix(tx.Recipient, params.AERO.PrefixDigits); e != nil {
			continue
		}
		batch = append(batch, tx)
		count++
		if len(batch) >= params.TxBatchSize {
			a.sendBatch(batch)
			batch = nil
		}
	}
	if len(batch) > 0 {
		a.sendBatch(batch)
	}
	a.logger.Slog.Println(fmt.Sprintf("AERO injected %d transactions", count))
	for !a.signal.GapEnough() {
		if a.due() {
			a.reconfigure()
		}
		time.Sleep(time.Second)
	}
	a.writeFinalReward()
}
