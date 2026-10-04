package aero

import (
	"crypto/ed25519"
	"crypto/rand"
	"testing"
)

func TestExpandPrefixRouting(t *testing.T) {
	s := Snapshot{Shards: 2, PrefixDigits: 1, MaxMoves: 3, Assignment: make([]uint64, 16), Addresses: []string{"0xab00", "af01", "bf02"}}
	s.Assignment[11] = 1
	plan := Plan{Moves: []Move{{Sender: 0, Receiver: 1, Prefix: 10}}}
	m, e := Expand(s, plan)
	if e != nil {
		t.Fatal(e)
	}
	if len(m) != 3 || m["0xab00"] != 1 || m["af01"] != 1 {
		t.Fatal(m)
	}
	if Lookup("a1234", m, 2, 1) != 1 {
		t.Fatal("new account missed migrated prefix")
	}
	plan.Moves = append(plan.Moves, plan.Moves[0])
	if _, e = Expand(s, plan); e == nil {
		t.Fatal("accepted repeated prefix")
	}
}
func TestStatisticsCountRelayOncePerLeg(t *testing.T) {
	assignment := make([]uint64, 16)
	assignment[1] = 1
	s := Statistics([][2]string{{"0000", "1000"}, {"0000", "0001"}}, assignment, 2, 1, 2)
	if s.Cross[0][0] != 1 || s.Cross[1][1] != 1 || s.Intra[0][0] != 1 || s.Throughput[0] != 1 {
		t.Fatalf("%+v", s)
	}
}
func TestQuorumOrderingDuplicatesAndEquivocation(t *testing.T) {
	r := NewRound(1, "expected", 4)
	for i := 0; i < 4; i++ {
		p, c, d := r.Accept(Envelope{Stage: "commit", Epoch: 1, Sender: uint64(i), Digest: "expected"})
		if p || c || d {
			t.Fatal("committed before proposal/prepares")
		}
	}
	p, c, d := r.Accept(Envelope{Stage: "proposal", Epoch: 1, Sender: 1, Digest: "expected"})
	if p || c || d {
		t.Fatal("accepted wrong leader")
	}
	p, _, _ = r.Accept(Envelope{Stage: "proposal", Epoch: 1, Sender: 0, Digest: "expected"})
	if !p {
		t.Fatal("no prepare")
	}
	for i := 0; i < 10; i++ {
		_, c, _ = r.Accept(Envelope{Stage: "prepare", Epoch: 1, Sender: 0, Digest: "expected"})
		if c {
			t.Fatal("duplicate inflated quorum")
		}
	}
	r.Accept(Envelope{Stage: "prepare", Epoch: 1, Sender: 1, Digest: "wrong"})
	if r.CommitSent {
		t.Fatal("equivocation inflated quorum")
	}
	r.Accept(Envelope{Stage: "prepare", Epoch: 1, Sender: 1, Digest: "expected"})
	_, c, d = r.Accept(Envelope{Stage: "prepare", Epoch: 1, Sender: 2, Digest: "expected"})
	if !c || !d {
		t.Fatal("buffered commits did not complete after prepares")
	}
	_, _, d = r.Accept(Envelope{Stage: "commit", Epoch: 1, Sender: 3, Digest: "expected"})
	if d {
		t.Fatal("duplicate commit emitted twice")
	}
	if NewRound(1, "x", 8).Quorum != 6 || NewRound(1, "x", 128).Quorum != 86 {
		t.Fatal("non 3f+1 quorum lacks intersection")
	}
}
func TestSignatureRejectsTamperingAndForgery(t *testing.T) {
	pub, key, e := ed25519.GenerateKey(rand.Reader)
	if e != nil {
		t.Fatal(e)
	}
	i := &Identity{ID: 0, Private: key, Public: map[uint64]ed25519.PublicKey{0: pub}}
	envelope := i.Sign("prepare", 1, "digest", nil)
	if !i.Verify(envelope) {
		t.Fatal("signature failed")
	}
	envelope.Digest = "tampered"
	if i.Verify(envelope) {
		t.Fatal("tampering accepted")
	}
	envelope = i.Sign("prepare", 1, "digest", nil)
	envelope.Sender = 1
	if i.Verify(envelope) {
		t.Fatal("forged identity accepted")
	}
}

func TestLeaderFailureRotatesAfterViewChangeQuorum(t *testing.T) {
	r := NewRound(1, "canonical", 4)
	for i := uint64(1); i <= 2; i++ {
		r.Accept(Envelope{Stage: "viewchange", Epoch: 1, View: 1, Sender: i, Digest: "canonical"})
	}
	if r.View != 0 {
		t.Fatal("view changed before quorum")
	}
	r.Accept(Envelope{Stage: "viewchange", Epoch: 1, View: 1, Sender: 3, Digest: "canonical"})
	if r.View != 1 {
		t.Fatal("failed to rotate leader")
	}
	p, _, _ := r.Accept(Envelope{Stage: "proposal", Epoch: 1, View: 0, Sender: 0, Digest: "canonical"})
	if p {
		t.Fatal("stale leader accepted")
	}
	p, _, _ = r.Accept(Envelope{Stage: "proposal", Epoch: 1, View: 1, Sender: 1, Digest: "canonical"})
	if !p {
		t.Fatal("new leader proposal rejected")
	}
}
