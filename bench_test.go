package hyperloglog

import (
	"hash/fnv"
	"math"
	"math/rand"
	"testing"
)

// Pre-generate batches of hashes from random data, with the bechmark timer
// disabled, so we don't count rand and hash time in the benchmark
type hashMaker struct {
	b      *testing.B
	fake32 []fakeHash32
	fake64 []fakeHash64
}

func (h *hashMaker) get32() *fakeHash32 {
	if len(h.fake32) == 0 {
		h.b.StopTimer()
		junk := make([]byte, 80000)
		rand.Read(junk)

		hash32 := fnv.New32a()
		for i := 0; i < len(junk); i += 8 {
			hash32.Reset()
			hash32.Write(junk[i : i+8])
			h.fake32 = append(h.fake32, fakeHash32(hash32.Sum32()))
		}
		h.b.StartTimer()
	}

	f := &h.fake32[len(h.fake32)-1]
	h.fake32 = h.fake32[:len(h.fake32)-1]
	return f
}

func (h *hashMaker) get64() *fakeHash64 {
	if len(h.fake64) == 0 {
		h.b.StopTimer()
		junk := make([]byte, 80000)
		rand.Read(junk)

		hash64 := fnv.New64a()
		for i := 0; i < len(junk); i += 8 {
			hash64.Reset()
			hash64.Write(junk[i : i+8])
			h.fake64 = append(h.fake64, fakeHash64(hash64.Sum64()))
		}
		h.b.StartTimer()
	}

	f := &h.fake64[len(h.fake64)-1]
	h.fake64 = h.fake64[:len(h.fake64)-1]
	return f
}

func benchmark(precision uint8, b *testing.B) {
	hashes := hashMaker{b: b}
	h, _ := New(precision)

	for i := 0; i < b.N; i++ {
		h.Add(hashes.get32())
	}

	if testing.Verbose() {
		var percentErr = func(est uint64) float64 {
			return 100.0 * math.Abs(float64(b.N)-float64(est)) / float64(b.N)
		}

		e := h.Count()
		b.Logf("HyperLogLog   P=%d Real: %8d, Est: %8d, Error: %6.4f%%\n", precision, b.N, e, percentErr(e))
	}
}

func BenchmarkHll4(b *testing.B) {
	benchmark(4, b)
}

func BenchmarkHll6(b *testing.B) {
	benchmark(6, b)
}

func BenchmarkHll8(b *testing.B) {
	benchmark(8, b)
}

func BenchmarkHll10(b *testing.B) {
	benchmark(10, b)
}

func BenchmarkHll14(b *testing.B) {
	benchmark(14, b)
}

func BenchmarkHll16(b *testing.B) {
	benchmark(16, b)
}

func benchmarkPlus(precision uint8, b *testing.B) {
	hashes := hashMaker{b: b}
	h, _ := NewPlus(precision)

	for i := 0; i < b.N; i++ {
		h.Add(hashes.get64())
	}

	if testing.Verbose() {
		var percentErr = func(est uint64) float64 {
			return 100.0 * math.Abs(float64(b.N)-float64(est)) / float64(b.N)
		}

		e := h.Count()
		b.Logf("HyperLogLog++ P=%d Real: %8d, Est: %8d, Error: %6.4f%%\n", precision, b.N, e, percentErr(e))
	}
}

func BenchmarkHllPlus4(b *testing.B) {
	benchmarkPlus(4, b)
}

func BenchmarkHllPlus6(b *testing.B) {
	benchmarkPlus(6, b)
}

func BenchmarkHllPlus8(b *testing.B) {
	benchmarkPlus(8, b)
}

func BenchmarkHllPlus10(b *testing.B) {
	benchmarkPlus(10, b)
}

func BenchmarkHllPlus14(b *testing.B) {
	benchmarkPlus(14, b)
}

func BenchmarkHllPlus16(b *testing.B) {
	benchmarkPlus(16, b)
}

func BenchmarkHLLSparseAdd(b *testing.B) {
	hashes := hashMaker{b: b}
	for n := 0; n < b.N; n++ {
		hpp, _ := NewPlus(10)
		for i := 0; i < 100; i++ {
			hpp.Add(hashes.get64())
		}
	}
}

func BenchmarkHLLSparseMerge(b *testing.B) {
	hashes := hashMaker{b: b}

	hpp1, _ := NewPlus(10)
	for i := 0; i < 100; i++ {
		hpp1.Add(hashes.get64())
	}

	hpp2, _ := NewPlus(10)
	for i := 0; i < 100; i++ {
		hpp2.Add(hashes.get64())
	}

	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		hpp3, _ := NewPlus(10)
		hpp3.Merge(hpp1)
		hpp3.Merge(hpp2)
	}
}
