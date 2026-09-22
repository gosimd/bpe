# Compact BPE baseline — apple-m5

UTC: 2026-09-22T10:02:28.545636+00:00 · commit: `58905369b51afe76fa51c571bab32220dda0c4b1` · protocol: `compact-v1`

CPU: Apple M5 · Darwin 25.6.0 / arm64
go version go1.27.0 darwin/arm64 · BPE backend: **scalar** (no SIMD dispatch implemented).
pretok/unicode use their default dependency dispatch; this is not a forced-scalar full pipeline.
Single thread; 3 warmup passes before each sample; 20 samples per row.
Five NFC-prepared language pools (en/ru/ar/ja/zh), independently encoded ~4 KiB requests.
Normalization, loading, compilation, allocation of dst, and HF correctness checks are outside timing.
No BPE cache. Special-token insertion off; AddedToken matching on. AppendIDs reuses sufficient dst capacity.

| Model | API | GB/s ↑ (95% median CI) | B/request ↓ | allocs/request ↓ | × previous | × scalar |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GPT-2 | Model.Tokenize | 0.031 [0.0307, 0.0312] | 191843 | 906.92 | — | — |
| GPT-2 | Tokenizer.AppendIDs | 0.0164 [0.0163, 0.0165] | 0 | 0.00 | — | — |
| GPT-2 | Tokenizer.Encode | 0.0105 [0.0104, 0.0106] | 716453 | 477.48 | — | — |
| Qwen3.8 / prepared NFC | Model.Tokenize | 0.0312 [0.0309, 0.0315] | 138513 | 865.16 | — | — |
| Qwen3.8 / prepared NFC | Tokenizer.AppendIDs | 0.014 [0.0139, 0.0141] | 0 | 0.00 | — | — |
| Qwen3.8 / prepared NFC | Tokenizer.Encode | 0.0117 [0.0116, 0.0117] | 324768 | 454.60 | — | — |

1 GB = 10⁹ bytes; GB/s = input bytes / nanoseconds. Model counts transformed-piece UTF-8 bytes; Tokenizer counts prepared source bytes.
One timed op is a complete five-language pass; allocations are divided by source-record count, including for Model.
Timed paths consume every output ID in a checksum. Language timing boundaries add five clock intervals per pass.
CI: distribution-free order-statistic interval for the median. Ratios, when present, use an independent-sample bootstrap (4,000 resamples, seed 1729).
Scalar comparison is unavailable: both current production merge paths are scalar; no heap-oracle comparison is substituted.

## Execution conditions

MacBook Pro M5; macOS 26.6.2; power observations recorded; no affinity pinning; background load and thermal conditions uncontrolled; no concurrent agent test or build jobs during sampling

Affinity: not pinned

Power at start:
```text
Now drawing from 'Battery Power'
 -InternalBattery-0 (id=34930787)	21%; discharging; 2:08 remaining present: true
```

Source snapshot: `source.patch` applies to the recorded commit; hashes in `run.json` include dirty/new benchmark files.
Raw samples: `samples.txt`; full metadata, source/prepared hashes, exact record lengths and summary: `run.json`.
Full-corpus oracle output is ignored and regenerated locally. HF 0.23.2 is a correctness oracle here, not a timed competitor.

First baseline: no previous version was supplied; no speedup is claimed.
