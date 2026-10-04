package aero

import (
	"crypto/ed25519"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
)

// Keys are provisioned once before launching the emulator, not negotiated by peers.
type Identity struct {
	ID      uint64
	Private ed25519.PrivateKey
	Public  map[uint64]ed25519.PublicKey
}

func LoadIdentity(dir string, id uint64) (*Identity, error) {
	b, e := os.ReadFile(filepath.Join(dir, "public.json"))
	if e != nil {
		return nil, e
	}
	var pub map[uint64][]byte
	if e = json.Unmarshal(b, &pub); e != nil {
		return nil, e
	}
	b, e = os.ReadFile(filepath.Join(dir, strconv.FormatUint(id, 10)+".key"))
	if e != nil {
		return nil, e
	}
	if len(b) != ed25519.PrivateKeySize {
		return nil, fmt.Errorf("invalid private key")
	}
	i := &Identity{ID: id, Private: b, Public: make(map[uint64]ed25519.PublicKey)}
	for id, k := range pub {
		if len(k) != ed25519.PublicKeySize {
			return nil, fmt.Errorf("invalid public key")
		}
		i.Public[id] = k
	}
	if !ed25519.PublicKey(i.Private.Public().(ed25519.PublicKey)).Equal(i.Public[id]) {
		return nil, fmt.Errorf("private/public mismatch")
	}
	return i, nil
}

type Envelope struct {
	Stage     string          `json:"stage"`
	Epoch     uint64          `json:"epoch"`
	Sender    uint64          `json:"sender"`
	Digest    string          `json:"digest"`
	Payload   json.RawMessage `json:"payload,omitempty"`
	Signature []byte          `json:"signature"`
	View      uint64          `json:"view,omitempty"`
}

func (e Envelope) signingBytes() []byte { e.Signature = nil; b, _ := json.Marshal(e); return b }
func (i *Identity) Sign(stage string, epoch uint64, digest string, payload interface{}) Envelope {
	return i.SignView(stage, epoch, 0, digest, payload)
}
func (i *Identity) SignView(stage string, epoch, view uint64, digest string, payload interface{}) Envelope {
	e := Envelope{Stage: stage, Epoch: epoch, Sender: i.ID, Digest: digest}
	e.View = view
	if payload != nil {
		e.Payload, _ = json.Marshal(payload)
	}
	e.Signature = ed25519.Sign(i.Private, e.signingBytes())
	return e
}
func (i *Identity) Verify(e Envelope) bool {
	key, ok := i.Public[e.Sender]
	return ok && len(key) == ed25519.PublicKeySize && ed25519.Verify(key, e.signingBytes(), e.Signature)
}

// Round buffers valid out-of-order votes and counts each identity once. A proposal
// is accepted only after the local replica has independently reproduced its digest.
// N-f is used for non-3f+1 memberships (e.g. the paper's 128 nodes).
type Round struct {
	Epoch       uint64
	Digest      string
	Members     uint64
	Quorum      int
	Proposed    bool
	PrepareSent bool
	CommitSent  bool
	Committed   bool
	prepares    map[uint64]bool
	commits     map[uint64]bool
	View        uint64
	viewVotes   map[uint64]map[uint64]bool
}

func NewRound(epoch uint64, digest string, members uint64) *Round {
	f := (members - 1) / 3
	return &Round{Epoch: epoch, Digest: digest, Members: members, Quorum: int(members - f), prepares: make(map[uint64]bool), commits: make(map[uint64]bool), viewVotes: make(map[uint64]map[uint64]bool)}
}
func (r *Round) Accept(e Envelope) (prepare, commit, done bool) {
	if e.Epoch != r.Epoch || e.Digest != r.Digest || e.Sender >= r.Members {
		return
	}
	switch e.Stage {
	case "proposal":
		// Proposal origin rotates each epoch. Timeout retransmission is handled by peers.
		if e.View != r.View || e.Sender != (r.Epoch-1+r.View)%r.Members {
			return
		}
		r.Proposed = true
	case "prepare":
		r.prepares[e.Sender] = true
	case "commit":
		r.commits[e.Sender] = true
	case "viewchange":
		if e.View <= r.View || e.View != r.View+1 {
			return
		}
		if r.viewVotes[e.View] == nil {
			r.viewVotes[e.View] = make(map[uint64]bool)
		}
		r.viewVotes[e.View][e.Sender] = true
		if len(r.viewVotes[e.View]) >= r.Quorum {
			r.View = e.View
		}
	default:
		return
	}
	if r.Proposed && !r.PrepareSent {
		r.PrepareSent = true
		prepare = true
	}
	if r.Proposed && len(r.prepares) >= r.Quorum && !r.CommitSent {
		r.CommitSent = true
		commit = true
	}
	if r.CommitSent && len(r.commits) >= r.Quorum && !r.Committed {
		r.Committed = true
		done = true
	}
	return
}
