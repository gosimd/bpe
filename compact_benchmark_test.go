package bpe_test

import (
	"os"
	"strings"
	"testing"
	"time"

	"github.com/gosimd/bpe"
	"github.com/gosimd/bpe/internal/benchsuite"
)

var compactChecksumSink uint64

// BenchmarkCompact is opt-in, repository-only measurement tooling. One op is
// one pass over all five independent language corpora. The script requires 1x,
// repeats the process and preserves individual language timings for diagnosis.
// Protocol v1: three warmup passes, one thread, IDs checksum in every timed path,
// no BPE cache, special-token insertion off, preallocated ID destination.
func BenchmarkCompact(b *testing.B) {
	b.StopTimer()
	path := os.Getenv("BPE_COMPACT_SUITE")
	if path == "" {
		b.Skip("run scripts/benchmark_compact.py")
	}
	root := os.Getenv("BPE_HF_REPOSITORY")
	if root == "" {
		b.Fatal("BPE_HF_REPOSITORY required")
	}
	suite, err := benchsuite.Load(path)
	if err != nil {
		b.Fatal(err)
	}
	if len(suite.Models) != 2 || len(suite.Corpora) != 5 {
		b.Fatal("compact-v1 requires two models and five corpora")
	}
	for _, spec := range suite.Models {
		b.Run(spec.Name, func(b *testing.B) {
			groups := make([]*corpusBenchmarkData, len(suite.Corpora))
			var records, sourceBytes, modelBytes int64
			var modelSum, encodeSum uint64
			maxIDs := 0
			for i, corpus := range suite.Corpora {
				data := loadCorpusBenchmarkShared(b, root, spec, corpus.CorpusPath(root), benchsuite.ReferencePath(root, spec, corpus), groups[0])
				if data.tokenizer == nil {
					b.Fatal("compact requires a checked full JSON pipeline")
				}
				data.verify(b)
				groups[i] = data
				records += int64(len(data.records))
				sourceBytes += data.sourceBytes
				modelBytes += data.modelBytes
				for _, record := range data.records {
					maxIDs = max(maxIDs, len(record.ids))
					for _, id := range record.ids {
						encodeSum += uint64(id)
					}
				}
				for _, piece := range data.modelPieces {
					for _, id := range piece.ids {
						modelSum += uint64(id)
					}
				}
			}
			for _, operation := range []string{"ModelTokenize", "TokenizerAppendIDs", "TokenizerEncode"} {
				b.Run(operation, func(b *testing.B) {
					b.StopTimer()
					if b.N != 1 {
						b.Fatal("compact-v1 requires -benchtime=1x")
					}
					inputBytes, want := sourceBytes, encodeSum
					if operation == "ModelTokenize" {
						inputBytes, want = modelBytes, modelSum
					}
					dst := make([]uint32, 0, maxIDs)
					pass := func(timings []time.Duration) uint64 {
						var sum uint64
						for index, data := range groups {
							start := time.Now()
							switch operation {
							case "ModelTokenize":
								for _, piece := range data.modelPieces {
									tokens, err := data.model.Tokenize(piece.value)
									if err != nil {
										b.Fatal(err)
									}
									for _, token := range tokens {
										sum += uint64(token.ID)
									}
									corpusBenchmarkTokensSink = tokens
								}
							case "TokenizerAppendIDs":
								for _, record := range data.records {
									var err error
									dst, err = data.tokenizer.AppendIDsWithOptions(dst[:0], record.text, bpe.EncodeOptions{AddSpecialTokens: false})
									if err != nil {
										b.Fatal(err)
									}
									for _, id := range dst {
										sum += uint64(id)
									}
								}
								corpusBenchmarkIDsSink = dst
							case "TokenizerEncode":
								for _, record := range data.records {
									encoding, err := data.tokenizer.EncodeWithOptions(record.text, bpe.EncodeOptions{AddSpecialTokens: false})
									if err != nil {
										b.Fatal(err)
									}
									for _, id := range encoding.IDs {
										sum += uint64(id)
									}
									corpusBenchmarkEncodingSink = encoding
								}
							}
							if timings != nil {
								timings[index] = time.Since(start)
							}
						}
						compactChecksumSink = sum
						return sum
					}
					for range 3 {
						if sum := pass(nil); sum != want {
							b.Fatal("warmup checksum differs from HF", sum, want)
						}
					}
					timings := make([]time.Duration, len(groups))
					b.ReportAllocs()
					b.SetBytes(inputBytes)
					b.ResetTimer()
					b.StartTimer()
					sum := pass(timings)
					b.StopTimer()
					if sum != want {
						b.Fatal("measurement checksum differs from HF", sum, want)
					}
					b.ReportMetric(float64(inputBytes), "input-bytes/op")
					b.ReportMetric(float64(records), "records/op")
					b.ReportMetric(float64(sum), "id-sum/op")
					for i, elapsed := range timings {
						b.ReportMetric(float64(elapsed.Nanoseconds()), strings.TrimSuffix(suite.Corpora[i].Name, "-compact")+"-ns/op")
					}
				})
			}
		})
	}
}
