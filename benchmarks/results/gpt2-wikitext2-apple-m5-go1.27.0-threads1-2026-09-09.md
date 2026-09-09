# Go vs HF GPT-2 / WikiText-2 baseline — Apple M5

This report compares the scalar Go implementation with the retained
Hugging Face Tokenizers 0.23.2 baseline on the same machine and corpus. Raw Go
samples are in
[`gpt2-wikitext2-apple-m5-go1.27.0-threads1-2026-09-09.txt`](gpt2-wikitext2-apple-m5-go1.27.0-threads1-2026-09-09.txt).
The HF timings and methodology are in the
[`hf-reference` report](../../tools/hf-reference/results/gpt2-wikitext2-apple-m5-macos-26.6.2-threads1-2026-09-09.md).

## Environment and workload

- Date: 2026-09-09.
- Machine: MacBook Pro `Mac17,2`, Apple M5, 10 cores (4 performance and 6
  efficiency), 32 GB RAM.
- OS: macOS 26.6.2 build 25G83, arm64.
- Power/load: AC Power reported, battery discharging; no affinity; ordinary
  foreground desktop load.
- Go: `go1.27.0 darwin/arm64`, `GOMAXPROCS=1`, benchmark `-cpu=1`.
- Corpus: 3,760 independent records, 1,144,248 source bytes, 219,615 prepared
  BPE pieces, 1,366,387 transformed-piece bytes, and 247,289 tokens.
- Sampling: ten independent samples, exactly one full corpus pass per sample.

Corpus/oracle validation and model construction occur before timing. An
untimed pass checks all produced IDs plus pre-tokenized values and offsets
against the HF oracle. During measurement, results are retained in
package-level sinks and token counts are consumed so the compiler cannot remove
the work. The input order is fixed.

## Results

Medians and ranges below are calculated from the ten raw `ns/op` samples.
Throughput uses MiB (1,048,576 bytes) so it can be compared with the retained
HF report; Go's raw `MB/s` column uses the `testing` package's decimal unit.

| Go operation | Median | Range | ns/input-byte | MiB/s | Tokens/s | B/op | allocs/op |
|---|---:|---:|---:|---:|---:|---:|---:|
| `PreTokenizer` | 37.182 ms | 35.832–38.027 ms | 32.495 | 29.35 | — | 74,561,048 | 482,101 |
| `ModelTokenize` | 81.011 ms | 77.480–89.045 ms | 59.288 | 16.09 | 3,052,540 | 39,827,528 | 439,234 |
| `ModelAppendIDs` | 61.959 ms | 60.604–64.607 ms | 45.345 | 21.03 | 3,991,202 | 264 | 3 |
| `TokenizerEncode` | 133.260 ms | 127.933–143.115 ms | 116.461 | 8.19 | 1,855,684 | 145,450,640 | 715,614 |
| `TokenizerAppendIDs` | 71.966 ms | 71.235–73.893 ms | 62.894 | 15.16 | 3,436,202 | 2,184 | 3 |

The first `TokenizerAppendIDs` sample was cold with respect to its internal
scratch pool (`27,544 B/op`, 35 allocations); the other nine and the reported
median were `2,184 B/op`, 3 allocations for the entire corpus.

## Equivalent Go/HF comparisons

| Equivalent output contract | Go median | HF median | Result |
|---|---:|---:|---:|
| `Model.Tokenize` ↔ uncached HF BPE | 81.011 ms | 81.421 ms | effectively equal; Go 1.005× |
| `Tokenizer.Encode` ↔ uncached HF full encode | 133.260 ms | 197.386 ms | Go 1.481× faster |

The 0.5% Model difference is below what should be treated as a meaningful win
without tighter same-process controls. The end-to-end result is substantial:
Go has 32.5% lower median latency. The diagnostic pre-tokenizer measurement is
37.182 ms for Go versus 96.273 ms for HF, but the two implementations expose
different internal output types, so this explains the pipeline result without
being a strict public-API comparison.

The Go and Rust samples were collected sequentially rather than interleaved,
so the ratios are a reproducible baseline, not a formal statistical proof.

Within Go, the ID-only model path is 1.308× faster than `Model.Tokenize`, and
the ID-only tokenizer path is 1.852× faster than `Tokenizer.Encode`. These are
different output contracts and are not claimed as speedups over HF.

No SIMD backend participated in these measurements; this is the scalar
baseline that future optimization must beat without changing output.

## Reproduction

The measured command was:

```sh
BPE_HF_REFERENCE=/Users/ax2/gosimd/bpe/testdata/generated/hf_gpt2_wikitext2_v0.23.2.jsonl \
BPE_HF_REPOSITORY=/Users/ax2/gosimd/bpe \
GOMAXPROCS=1 \
go test -run '^$' -bench '^BenchmarkGPT2WikiText2$' \
  -benchmem -benchtime=1x -count=10 -cpu=1 .
```

The source revision was `097b84c8774a93c5d618130360f74898524b2b13` with
the benchmark changes in the working tree. Relevant SHA-256 hashes:

| Source | SHA-256 |
|---|---|
| `gpt2_wikitext2_benchmark_test.go` | `f2e623b5b4548d1c5c5e486adf18ae7bf2ceb315c5de6ed9e2370faef2105eb3` |
| `model.go` | `d20737b1def5bf3b3e4f2d4be590a4889d02011493e736b565d301f675b816fa` |
| `tokenizer.go` | `19ab0a2dd763c367f090a72471978e6e539794fc895c5d6787a6044226aeecef` |
| `internal/model/encode.go` | `e19836f0f0cb2ca57fd6dead05bddbaf7ad68852bf54f22a23216763190a1d4e` |
| `internal/tokenizer/tokenizer.go` | `dbb5db3ba91d963367f090a72471978a43a415d3c8a9a9334f27dd2d476541` |
| `internal/bytelevel/bytelevel.go` | `db4f9fa28dc1ce0421c3a7b59f06b65bf40d7850f501537349faf5b8fd3ecf59` |

Input hashes are inherited from the validated oracle and match the HF report:
tokenizer `8414cab9…`, corpus `d8b897d4…`, and oracle `8c7b5720…`.
