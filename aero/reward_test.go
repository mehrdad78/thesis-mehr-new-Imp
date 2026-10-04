package aero

import (
	"math"
	"testing"
)

func TestRewardComponentsAndCustomWeights(t *testing.T) {
	stats := Stats{Throughput: make([]float64, 4), Cross: [][]float64{{4, 0, 0, 0}, {0, 2, 1, 1}}, Intra: [][]float64{{0, 0, 0, 2}, {0, 0, 0, 0}}}
	r := EvaluateReward(stats, 2, 3)
	if math.Abs(r.Reward-(-2.32)) > 1e-12 || math.Abs(r.CrossRatio-.8) > 1e-12 || r.CrossVariance != 1.5 || math.Abs(r.NormalizedCrossVariance-.24) > 1e-12 {
		t.Fatalf("incorrect reward components: %+v", r)
	}
	empty := Stats{Throughput: make([]float64, 4), Cross: [][]float64{{0, 0, 0, 0}}, Intra: [][]float64{{0, 0, 0, 0}}}
	if zero := EvaluateReward(empty, 2, 3); zero.Reward != 0 || zero.CrossRatio != 0 || zero.NormalizedCrossVariance != 0 {
		t.Fatal(zero)
	}
}
