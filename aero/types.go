// Package aero implements AERO's prefix state, policy bridge and logical consensus.
package aero

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"strings"
)

type Config struct {
	Python         string `json:"Python"`
	Checkpoint     string `json:"Checkpoint"`
	KeyDir         string `json:"KeyDir"`
	PrefixDigits   int    `json:"PrefixDigits"`
	EpochBlocks    int    `json:"EpochBlocks"`
	MaxMoves       int    `json:"MaxMoves"`
	HistoryWindow  int    `json:"HistoryWindow"`
	Seed           int64  `json:"Seed"`
	TimeoutSeconds int    `json:"TimeoutSeconds"`
	OnlineTraining bool   `json:"OnlineTraining"`
}

func DefaultConfig() Config {
	return Config{Python: "python", Checkpoint: "aero/checkpoints/policy.pt", KeyDir: "aero/keys",
		PrefixDigits: 2, EpochBlocks: 100, MaxMoves: 32, HistoryWindow: 32, Seed: 1, TimeoutSeconds: 300, OnlineTraining: true}
}

func (c Config) Validate() error {
	if c.Python == "" || c.Checkpoint == "" || c.KeyDir == "" || c.PrefixDigits < 1 || c.PrefixDigits > 3 ||
		c.EpochBlocks < 1 || c.MaxMoves < 1 || c.HistoryWindow < 1 || c.TimeoutSeconds < 1 || c.Seed < 0 {
		return fmt.Errorf("invalid AERO config: paths, positive sizes/timeouts, digits 1..3 and nonnegative seed required")
	}
	return nil
}

func Prefix(address string, digits int) (int, error) {
	address = strings.TrimPrefix(strings.ToLower(address), "0x")
	if digits < 1 || digits > 3 || len(address) < digits {
		return 0, fmt.Errorf("invalid prefix length")
	}
	for _, c := range address {
		if !strings.ContainsRune("0123456789abcdef", c) {
			return 0, fmt.Errorf("invalid account address")
		}
	}
	p, err := strconv.ParseUint(address[:digits], 16, 16)
	return int(p), err
}

// Prefix entries coexist with concrete account entries in the transfer map.
const prefixMarker = "@aero:"

func PrefixKey(p, digits int) string { return fmt.Sprintf("%s%0*x", prefixMarker, digits, p) }
func IsPrefixKey(key string) bool    { return strings.HasPrefix(key, prefixMarker) }
func Lookup(address string, table map[string]uint64, shards, digits int) uint64 {
	p, err := Prefix(address, digits)
	if err != nil {
		panic(err)
	}
	if s, ok := table[PrefixKey(p, digits)]; ok {
		return s
	}
	return uint64(p % shards)
}

type Move struct {
	Sender   uint64 `json:"sender"`
	Receiver uint64 `json:"receiver"`
	Prefix   int    `json:"prefix"`
}
type Stats struct {
	Throughput []float64   `json:"throughput"`
	Ratio      []float64   `json:"ratio"`
	Variance   []float64   `json:"variance"`
	Cross      [][]float64 `json:"cross"`
	Intra      [][]float64 `json:"intra"`
}
type Snapshot struct {
	Epoch         uint64   `json:"epoch"`
	Seed          int64    `json:"seed"`
	Shards        int      `json:"shards"`
	PrefixDigits  int      `json:"prefix_digits"`
	MaxMoves      int      `json:"max_moves"`
	Assignment    []uint64 `json:"assignment"`
	History       []Move   `json:"history"`
	Current       Stats    `json:"current"`
	Previous      Stats    `json:"previous"`
	Addresses     []string `json:"addresses"`
	HistoryWindow int      `json:"history_window"`
}
type Plan struct {
	Moves       []Move         `json:"moves"`
	ModelDigest string         `json:"model_digest"`
	Reward      *RewardSummary `json:"reward,omitempty"`
}

func (s Snapshot) Validate(c Config, shards int) error {
	count := 1 << (4 * c.PrefixDigits)
	if s.Epoch == 0 || s.Seed != c.Seed+int64(s.Epoch) || s.Shards != shards || s.PrefixDigits != c.PrefixDigits ||
		s.MaxMoves != c.MaxMoves || s.HistoryWindow != c.HistoryWindow || len(s.Assignment) != count || len(s.History) > c.HistoryWindow {
		return fmt.Errorf("epoch/configuration mismatch")
	}
	for _, owner := range s.Assignment {
		if owner >= uint64(shards) {
			return fmt.Errorf("invalid prefix owner")
		}
	}
	for _, stats := range []Stats{s.Current, s.Previous} {
		if len(stats.Throughput) != shards || len(stats.Ratio) != shards || len(stats.Variance) != shards || len(stats.Cross) != count || len(stats.Intra) != count {
			return fmt.Errorf("invalid statistics dimensions")
		}
		for _, row := range append(append([][]float64{stats.Throughput, stats.Ratio, stats.Variance}, stats.Cross...), stats.Intra...) {
			if len(row) != shards {
				return fmt.Errorf("invalid statistics row")
			}
			for _, v := range row {
				if v < 0 || math.IsNaN(v) || math.IsInf(v, 0) {
					return fmt.Errorf("invalid statistic")
				}
			}
		}
	}
	return nil
}

func Digest(v interface{}) string {
	b, err := json.Marshal(v)
	if err != nil {
		panic(err)
	}
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])
}
func PlanDigest(s Snapshot, p Plan) string {
	return Digest(struct {
		Snapshot Snapshot
		Plan     Plan
	}{s, p})
}

// Expand produces both concrete state transfers and persistent prefix ownership.
func Expand(s Snapshot, p Plan) (map[string]uint64, error) {
	result := make(map[string]uint64)
	seen := make(map[int]bool)
	for _, m := range p.Moves {
		if m.Prefix < 0 || m.Prefix >= len(s.Assignment) || m.Sender >= uint64(s.Shards) || m.Receiver >= uint64(s.Shards) || m.Sender == m.Receiver ||
			seen[m.Prefix] || s.Assignment[m.Prefix] != m.Sender {
			return nil, fmt.Errorf("invalid migration: %+v", m)
		}
		seen[m.Prefix] = true
		result[PrefixKey(m.Prefix, s.PrefixDigits)] = m.Receiver
	}
	if len(p.Moves) > s.MaxMoves {
		return nil, fmt.Errorf("migration list exceeds limit")
	}
	for _, address := range s.Addresses {
		prefix, err := Prefix(address, s.PrefixDigits)
		if err != nil {
			return nil, err
		}
		if dest, ok := result[PrefixKey(prefix, s.PrefixDigits)]; ok {
			result[address] = dest
		}
	}
	return result, nil
}
