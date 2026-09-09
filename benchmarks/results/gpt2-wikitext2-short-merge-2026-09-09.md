# Go vs HF after short-piece merge optimization — Apple M5

Fresh measurements of both libraries on 2026-09-09. This report does not reuse
the HF timings from the earlier baseline. The production Go build uses the
direct two-symbol path, scalar current-pair array for short pieces, and heap
fallback. The ID-only paths still use the heap. No SIMD backend or experiment
build tags are enabled.

## Equivalent API results

Each time is the median duration of a complete corpus pass, computed from
all ten samples of the corresponding operation, not from block medians.

| Equivalent output contract | Go | HF 0.23.2 | HF / Go | Go latency reduction |
| --- | ---: | ---: | ---: | ---: |
| `Model.Tokenize` ↔ `bpe_only_uncached` | 61.001 ms | 86.020 ms | 1.410× | 29.1% |
| `Tokenizer.Encode` ↔ `encode_full_uncached` | 108.037 ms | 199.002 ms | 1.842× | 45.7% |

These are workload-specific materializing API comparisons, not claims about
all tokenizers, SIMD, or the asymptotic BPE algorithm. HF result-cache capacity
is explicitly zero in both comparison cases. The public Model measurement
includes symbol initialization and token/ID/offset construction, not just
the merge loop. Full encode includes the whole supported pipeline.

## Environment and workload

- CPU: Apple M5, confirmed by the Go benchmark header. HF reports 10 available
  logical CPUs, but the runs use one execution thread/Go CPU. Hardware model
  and RAM queries through `sysctl` were denied by the sandbox and are not
  inferred from an older report.
- OS: macOS 26.6.2 build 25G83, arm64.
- Power: AC Power reported; battery at 67% and discharging before measurement.
  Ordinary desktop load; no affinity or thermal control.
- Go: `go1.27.0 darwin/arm64`, `GOARM64=v8.0`, empty `GOEXPERIMENT`,
  `GOMAXPROCS=1`, `-test.cpu=1`; normal GC, no race instrumentation.
- Rust: `rustc 1.98.1 (48a229cea 2026-09-01)`, pinned `tokenizers = "=0.23.2"`;
  locked release build, thin LTO, one codegen unit, Oniguruma feature.
- GPT-2 model and WikiText-2 raw validation: 3,760 independent records,
  1,144,248 source UTF-8 bytes, 219,615 prepared BPE pieces,
  1,366,387 transformed-piece bytes, 247,289 output tokens.
- Encoding mode: no added special tokens, padding, or truncation;
  byte offsets; records and pieces retain their original fixed order.

Input hashes:

| Input | SHA-256 |
| --- | --- |
| `tokenizer.json` | `8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6` |
| `vocab.json` | `196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783` |
| `merges.txt` | `1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5` |
| `validation.jsonl` | `d8b897d49c70e35db334af33364549e4db97c89fcfaf5917d306208abcd7587d` |
| HF three-level oracle | `8c7b57203e7828b5409ca6aef4bddb2b376e2dbeb1bf9c3bfad5879b3e6a8857` |

Source and executable hashes, Git revision and staged working-tree state,
and exact executed commands are preserved in
[metadata](gpt2-wikitext2-short-merge-2026-09-09.metadata.json).
The benchmark used the staged short-piece implementation, not merely HEAD.

## Sampling and validation

Process order was **HF1 → Go1 → Go2 → HF2**. No benchmark or build ran
concurrently with another measurement. Each process measures all of its
operations in the existing harness order; this is block counterbalancing,
not per-operation randomization or independent paired observations.

- HF: two processes, each with three untimed warm-up corpus passes and five
  measured single-pass samples per operation: ten samples total.
- Go: two processes, each with `-test.count=5 -test.benchtime=5x`: ten
  samples per operation, each an average over five corpus passes. Go's
  one-iteration calibration and untimed full-output validation additionally
  warm the paths. The per-sample timing distributions are therefore not
  identically collected across languages.
- Corpus/oracle loading, model construction, precomputed BPE-piece preparation,
  and validation are outside the timer. Timed BPE-only cases receive prepared
  pieces in both languages. Output allocation remains inside the measurement.
- Rust constructs and destroys results within each operation. Go retains
  results in package-level sinks and uses normal GC; token counts are consumed.
  These are the runtimes' normal API costs, not matched allocator kernels.
- Both tools validate outputs against the same oracle before timing. The
  separate full differential test also passed before benchmarking, checking
  Model strings/IDs/offsets and the surrounding pipeline boundaries. No Rust
  generation was invoked by Go tests and no fixtures were changed.

All samples are retained, including the slow HF observations. Block drift is
visible: HF BPE block medians are 92.797 and 84.553 ms; Go Model block medians
are 60.474 and 61.289 ms. HF uncached encode block medians are 204.966 and
195.123 ms; Go encode block medians are 106.626 and 108.534 ms. The direction
of both main comparisons holds in both blocks, but the ratios should not be
treated as precise universal speedup constants or a formal significance test.

## All Go operations

One operation is one complete corpus pass. Rates use the median time:
transformed-piece bytes for Model, source bytes otherwise. MiB/s uses
1,048,576 bytes; the raw Go MB/s column uses decimal megabytes.

| Go operation | Median | Min–max | ns/input-byte | MiB/s | Tokens/s | Median B/op | Median allocs/op |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `PreTokenizer` | 39.516 ms | 38.329–42.129 ms | 34.534 | 27.62 | — | 74,561,036.5 | 482,101 |
| `ModelTokenize` | 61.001 ms | 60.096–62.265 ms | 44.644 | 21.36 | 4,053,850 | 39,827,392 | 439,232 |
| `ModelAppendIDs` | 58.998 ms | 57.689–59.977 ms | 43.178 | 22.09 | 4,191,480 | 52 | 0 |
| `TokenizerEncode` | 108.037 ms | 106.306–109.118 ms | 94.417 | 10.10 | 2,288,939 | 145,450,414 | 715,610 |
| `TokenizerAppendIDs` | 69.731 ms | 68.691–71.140 ms | 60.941 | 15.65 | 3,546,315 | 436 | 0 |

Go reports integer per-operation allocation counts. `0 allocs/op` alongside
nonzero B/op on the ID-only paths is an amortized/truncated five-pass result,
not proof of zero allocations across the entire measured invocation. Output
destinations and internal scratch are reused. Median B/op is calculated from
the ten reported values; the fractional pre-tokenizer median is intentional.

Do not compare AppendIDs directly with HF's materializing APIs. The narrow
gap between ModelTokenize and ModelAppendIDs here is also influenced by their
different merge implementations: only Tokenize uses the new short-piece path.

## All HF operations

| HF operation | Median | Min–max | ns/input-byte | MiB/s | Tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| `pretokenize_only` | 98.003 ms | 96.536–100.538 ms | 85.648 | 11.13 | — |
| `bpe_only_uncached` | 86.020 ms | 83.606–93.743 ms | 62.954 | 15.15 | 2,874,788 |
| `encode_full_uncached` | 199.002 ms | 192.347–213.630 ms | 173.915 | 5.48 | 1,242,646 |
| `encode_full_warm_cache` | 139.111 ms | 135.803–153.913 ms | 121.574 | 7.84 | 1,777,642 |

Pre-tokenizer outputs have different internal representations and allocation
contracts, so the pre-tokenizer rows are diagnostic, not an equivalent public
API speedup. BPE time is measured directly, never inferred by subtraction.

HF warm-cache encode has a default thread-local capacity of 10,000 and
19,736 distinct cache-eligible corpus pieces. It receives a complete untimed
prefill plus warm-ups, but this is not an all-cache-hits workload. Retain it
as a separate production-oriented metric; it is not the uncached algorithm
comparison. This run did not measure Rust allocations, batch parallelism, or
collect a new CPU profile.

## Raw results and reproduction

- Go: [block 1](gpt2-wikitext2-short-merge-2026-09-09-go-1.txt),
  [block 2](gpt2-wikitext2-short-merge-2026-09-09-go-2.txt).
- HF: [block 1 JSON](../../tools/hf-reference/results/gpt2-wikitext2-short-merge-2026-09-09-hf-1.json),
  [block 2 JSON](../../tools/hf-reference/results/gpt2-wikitext2-short-merge-2026-09-09-hf-2.json),
  with stdout logs beside them.
- [Commands and source hashes](gpt2-wikitext2-short-merge-2026-09-09.metadata.json).

From the repository root, build outside the measurement interval:

```sh
GOCACHE=/private/tmp/bpe-gocache GOPATH=/private/tmp/bpe-gopath \
  go test -c -o /private/tmp/bpe-hf-comparison.test .

CARGO_HOME=/private/tmp/bpe-cargo \
RUSTC=/Users/ax2/.rustup/toolchains/1.98.1-aarch64-apple-darwin/bin/rustc \
  /Users/ax2/.rustup/toolchains/1.98.1-aarch64-apple-darwin/bin/cargo \
  build --locked --release --offline --manifest-path tools/hf-reference/Cargo.toml
```

These are the actual machine-local build commands. Cargo was not on PATH;
the installed pinned compiler and existing offline dependency cache were
selected explicitly. Neither toolchain nor dependencies were updated.

Each Go block executes:

```sh
BPE_HF_REPOSITORY=/Users/ax2/gosimd/bpe \
BPE_HF_REFERENCE=/Users/ax2/gosimd/bpe/testdata/generated/hf_gpt2_wikitext2_v0.23.2.jsonl \
GOMAXPROCS=1 \
  /private/tmp/bpe-hf-comparison.test -test.run='^$' \
  -test.bench='^BenchmarkGPT2WikiText2$' -test.benchtime=5x \
  -test.count=5 -test.cpu=1 -test.benchmem
```

Each HF block executes the following command, using a distinct output path
and recording actual power/load conditions (the complete arguments used here
are in metadata):

```sh
GOMAXPROCS=1 tools/hf-reference/target/release/hf-reference benchmark \
  --tokenizer testdata/gpt2/tokenizer.json \
  --vocab testdata/gpt2/vocab.json --merges testdata/gpt2/merges.txt \
  --corpus testdata/wikitext2/validation.jsonl \
  --fixture testdata/generated/hf_gpt2_wikitext2_v0.23.2.jsonl \
  --output tools/hf-reference/results/NEW-RESULT.json \
  --warmup 3 --iterations 5 --threads 1 \
  --environment 'Record CPU, OS, power, load and process order'
```

Repeat in HF1/Go1/Go2/HF2 order. The corpus is already local; use
`make generate-fixtures` separately only if the ignored oracle is absent.
The pre-measurement correctness command was:

```sh
GOCACHE=/private/tmp/bpe-gocache GOPATH=/private/tmp/bpe-gopath \
BPE_HF_REPOSITORY=/Users/ax2/gosimd/bpe \
BPE_HF_REFERENCE=/Users/ax2/gosimd/bpe/testdata/generated/hf_gpt2_wikitext2_v0.23.2.jsonl \
  go test -count=1 -run='^TestHFGPT2WikiText2' ./internal/bytelevel .
```

The earlier Go/HF report remains an unchanged historical baseline. Its
one-pass Go samples and older HF measurements differ from this run; subtracting
those numbers is not a controlled estimate of the short-piece change alone.
