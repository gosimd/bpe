# HF GPT-2 / WikiText-2 baseline — Apple M5

This is the first retained upstream baseline for the representative corpus.
The machine-specific raw data is in
[`gpt2-wikitext2-apple-m5-macos-26.6.2-threads1-2026-09-09.json`](gpt2-wikitext2-apple-m5-macos-26.6.2-threads1-2026-09-09.json),
and the raw CPU sample is in
[`gpt2-wikitext2-apple-m5-macos-26.6.2-threads1-2026-09-09-encode-full-uncached.sample.txt`](gpt2-wikitext2-apple-m5-macos-26.6.2-threads1-2026-09-09-encode-full-uncached.sample.txt).

## Environment

- Date: 2026-09-09.
- Machine: MacBook Pro `Mac17,2`, Apple M5, 10 cores (4 performance and 6
  efficiency), 32 GB RAM.
- OS: macOS 26.6.2 build 25G83, arm64.
- Power/load: AC Power reported, battery discharging at start; no affinity;
  ordinary foreground desktop load.
- Software: Rust 1.98.1, `tokenizers` 0.23.2, release build with thin LTO and
  one codegen unit.

## Workload and method

WikiText-2 raw validation contains 3,760 independently encoded records and
1,144,248 source UTF-8 bytes. The generated oracle contains 219,615 BPE model
pieces, 1,366,387 transformed-piece UTF-8 bytes, and 247,289 final tokens per
corpus pass. Special tokens, padding, and truncation are disabled; offsets are
bytes.

The run used one thread, three untimed warm-up passes, and ten measured passes.
Input loading, oracle validation, model construction, and cache prefill are
excluded. Output construction and destruction are included. The uncached model
was built with `cache_capacity(0)` and checked against the oracle before timing.
The warm-cache tokenizer retained upstream's default thread-local capacity of
10,000 entries and was prefilled once on the measurement thread. The corpus has
19,736 distinct cache-eligible pieces, so this is a realistic partially warm
cache, not a 100% hit-rate microbenchmark.

| Operation | Median | Range | Throughput | Byte basis |
|---|---:|---:|---:|---|
| `pretokenize_only` | 96.273 ms | 95.013–97.110 ms | 11.33 MiB/s | source text |
| `bpe_only_uncached` | 81.421 ms | 80.388–86.619 ms | 16.00 MiB/s | transformed pieces |
| `encode_full_uncached` | 197.386 ms | 188.845–206.395 ms | 5.53 MiB/s | source text |
| `encode_full_warm_cache` | 137.606 ms | 132.826–141.529 ms | 7.93 MiB/s | source text |

The warm-cache full encode was 1.434× as fast as the uncached full encode in
this run, or 30.3% lower median latency. It is a production-oriented metric,
not the BPE-algorithm baseline.

These operations are deliberately measured independently. In particular,
`encode_full_uncached - pretokenize_only` is not BPE time: full encode also
contains pipeline bookkeeping, Encoding and offset construction,
post-processing, and allocations. The BPE-only throughput is also based on
transformed-piece bytes, whereas the other rows use source bytes.

The benchmark command was:

```sh
cargo run --locked --release \
  --manifest-path tools/hf-reference/Cargo.toml -- benchmark \
  --tokenizer testdata/gpt2/tokenizer.json \
  --vocab testdata/gpt2/vocab.json \
  --merges testdata/gpt2/merges.txt \
  --corpus testdata/wikitext2/validation.jsonl \
  --fixture testdata/generated/hf_gpt2_wikitext2_v0.23.2.jsonl \
  --output tools/hf-reference/results/gpt2-wikitext2-apple-m5-macos-26.6.2-threads1-2026-09-09.json \
  --warmup 3 \
  --iterations 10 \
  --threads 1 \
  --environment "MacBook Pro Mac17,2; Apple M5, 10 cores (4 performance + 6 efficiency), 32 GB; macOS 26.6.2 build 25G83, arm64; AC Power reported, battery discharging at start; one benchmark thread, no affinity, ordinary foreground desktop load"
```

## CPU profile

The release executable ran `encode-full-uncached` for 35 seconds and completed
179 corpus passes. `/usr/bin/sample` attached only after `PROFILE_READY` and
captured the main thread every 1 ms for 10 seconds. It observed 8,470 main
thread samples, including 8,105 under `encode_corpus`.

Within the inclusive `main` call tree, 3,714 samples passed through model
tokenization and 3,677 through ByteLevel pre-tokenization. The largest leaf was
`BPE::merge_word` with 1,540 samples; Oniguruma's `match_at` had 510. Allocation,
free, reallocation, and memory movement routines were also prominent leaves.
These are sampled inclusive stacks and leaf counts, not exact elapsed-time
percentages, but they show that both regex-based pre-tokenization and BPE merge
work are major parts of full encode on this corpus.

The profiled workload was started with:

```sh
tools/hf-reference/target/release/hf-reference profile \
  --tokenizer testdata/gpt2/tokenizer.json \
  --vocab testdata/gpt2/vocab.json \
  --merges testdata/gpt2/merges.txt \
  --corpus testdata/wikitext2/validation.jsonl \
  --fixture testdata/generated/hf_gpt2_wikitext2_v0.23.2.jsonl \
  --operation encode-full-uncached \
  --seconds 35 \
  --threads 1
```

After `PROFILE_READY`, the sampler command was:

```sh
/usr/bin/sample <pid> 10 1 -mayDie -fullPaths \
  -file tools/hf-reference/results/gpt2-wikitext2-apple-m5-macos-26.6.2-threads1-2026-09-09-encode-full-uncached.sample.txt
```
