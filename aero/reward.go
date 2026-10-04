package aero

import "math"

// RewardSummary uses the PPO reward weights saved in the Python checkpoint.
type RewardSummary struct {
	Reward                  float64 `json:"reward"`
	CrossRatio              float64 `json:"cross_ratio"`
	CrossVariance           float64 `json:"cross_variance"`
	NormalizedCrossVariance float64 `json:"normalized_cross_variance"`
	CrossWeight             float64 `json:"cross_weight"`
	VarianceWeight          float64 `json:"variance_weight"`
}

// EvaluateReward mirrors Python's reward_components for the final epoch, which
// has no following migration decision and therefore no worker feedback response.
func EvaluateReward(s Stats, crossWeight, varianceWeight float64) RewardSummary {
	n := len(s.Throughput)
	result := RewardSummary{CrossWeight: crossWeight, VarianceWeight: varianceWeight}
	if n == 0 {
		return result
	}
	cross, intra := make([]float64, n), make([]float64, n)
	for _, row := range s.Cross {
		for i, v := range row {
			cross[i] += v
		}
	}
	for _, row := range s.Intra {
		for i, v := range row {
			intra[i] += v
		}
	}
	var crossTotal, intraTotal float64
	for i := 0; i < n; i++ {
		crossTotal += cross[i]
		intraTotal += intra[i]
	}
	mean := crossTotal / float64(n)
	for _, v := range cross {
		result.CrossVariance += (v - mean) * (v - mean)
	}
	result.CrossVariance /= float64(n)
	result.CrossRatio = crossTotal / math.Max(1, crossTotal+intraTotal)
	meanLoad := (crossTotal + intraTotal) / float64(n)
	result.NormalizedCrossVariance = result.CrossVariance / math.Max(1, meanLoad*meanLoad)
	result.Reward = -crossWeight*result.CrossRatio - varianceWeight*result.NormalizedCrossVariance
	return result
}
