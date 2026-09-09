package model

import "testing"

func TestMinimumPair(t *testing.T) {
	for n := 2; n <= 32; n++ {
		for winner := 0; winner < n; winner++ {
			var keys [32]uint32
			for i := range keys {
				keys[i] = noPair
			}
			if got := minimumPair(&keys, n); got != noPair {
				t.Fatal("inactive slots selected")
			}
			for i := winner; i < n; i++ {
				keys[i] = 12345<<5 | uint32(i)
			}
			if got := minimumPair(&keys, n); got != keys[winner] {
				t.Fatalf("n=%d slot=%d got=%d", n, winner, got)
			}
		}
	}
}
