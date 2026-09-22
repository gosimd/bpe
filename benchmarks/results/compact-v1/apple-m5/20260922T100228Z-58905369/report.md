# Compact BPE baseline — apple-m5

UTC: 2026-09-22T10:02:28.545636+00:00 · commit: `58905369b51afe76fa51c571bab32220dda0c4b1` · protocol: `compact-v1`

CPU: Apple M5 · Darwin 25.6.0 / arm64
go version go1.27.0 darwin/arm64 · BPE backend: **scalar** (no SIMD dispatch implemented).
pretok/unicode use their default dependency dispatch; this is not a forced-scalar full pipeline.
Single thread; 3 warmup passes before each sample; 20 samples per row.
Five NFC-prepared language pools (en/ru/ar/ja/zh), independently encoded ~4 KiB requests.
Normalization, loading, compilation, allocation of dst, and HF correctness checks are outside timing.
No BPE cache. Special-token insertion off; AddedToken matching on. AppendIDs reuses sufficient dst capacity.

| Model | API | MB/s ↑ (95% median CI) | B/request ↓ | allocs/request ↓ | × previous | × scalar |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GPT-2 | Model.Tokenize | 31 [30.7, 31.2] | 191843 | 906.92 | — | — |
| GPT-2 | Tokenizer.AppendIDs | 16.4 [16.3, 16.5] | 0 | 0.00 | — | — |
| GPT-2 | Tokenizer.Encode | 10.5 [10.4, 10.6] | 716453 | 477.48 | — | — |
| Qwen3.8 / prepared NFC | Model.Tokenize | 31.2 [30.9, 31.5] | 138513 | 865.16 | — | — |
| Qwen3.8 / prepared NFC | Tokenizer.AppendIDs | 14 [13.9, 14.1] | 0 | 0.00 | — | — |
| Qwen3.8 / prepared NFC | Tokenizer.Encode | 11.7 [11.6, 11.7] | 324768 | 454.60 | — | — |

1 MB = 10⁶ bytes; MB/s = 1000 × input bytes / nanoseconds. Model counts transformed-piece UTF-8 bytes; Tokenizer counts prepared source bytes.
One timed op is a complete five-language pass; allocations are divided by source-record count, including for Model.
Timed paths consume every output ID in a checksum. Language timing boundaries add five clock intervals per pass.
CI: distribution-free order-statistic interval for the median. Ratios, when present, use an independent-sample bootstrap (4,000 resamples, seed 1729).
Scalar comparison is unavailable: both current production merge paths are scalar; no heap-oracle comparison is substituted.

## Execution conditions

MacBook Pro M5; macOS 26.6.2; battery power; no affinity pinning; background load and thermal conditions uncontrolled.

Affinity: not pinned. Power source: Now drawing from 'Battery Power'.

Raw samples: [samples.txt](samples.txt). Measurement metadata: [run.json](run.json).
HF 0.23.2 is a correctness oracle here, not a timed competitor.

Measured source includes uncommitted changes; its file hashes are recorded in run.json.

First baseline: no previous version was supplied; no speedup is claimed.
