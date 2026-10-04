package aero

// Statistics count one execution for intra-shard transfers and two for relay.
// Input pairs are original completed transactions, never duplicated relay legs.
func Statistics(pairs [][2]string, assignment []uint64, shards, digits int, seconds float64) Stats {
	prefixes := 1 << (4 * digits)
	s := Stats{Throughput: make([]float64, shards), Ratio: make([]float64, shards), Variance: make([]float64, shards),
		Cross: make([][]float64, prefixes), Intra: make([][]float64, prefixes)}
	for p := 0; p < prefixes; p++ {
		s.Cross[p] = make([]float64, shards)
		s.Intra[p] = make([]float64, shards)
	}
	c, b := make([]float64, shards), make([]float64, shards)
	for _, pair := range pairs {
		p, e := Prefix(pair[0], digits)
		if e != nil {
			panic(e)
		}
		q, e := Prefix(pair[1], digits)
		if e != nil {
			panic(e)
		}
		a, z := assignment[p], assignment[q]
		if a == z {
			s.Intra[p][a]++
			b[a]++
		} else {
			s.Cross[p][a]++
			s.Cross[q][z]++
			c[a]++
			c[z]++
		}
	}
	mean := 0.
	for _, v := range c {
		mean += v / float64(shards)
	}
	if seconds <= 0 {
		seconds = 1
	}
	for i := 0; i < shards; i++ {
		s.Throughput[i] = (c[i] + b[i]) / seconds
		if c[i]+b[i] > 0 {
			s.Ratio[i] = c[i] / (c[i] + b[i])
		}
		s.Variance[i] = (c[i] - mean) * (c[i] - mean)
	}
	return s
}
