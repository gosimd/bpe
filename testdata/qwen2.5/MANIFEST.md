# Qwen2.5 tokenizer assets

- Repository: [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B).
- Exact revision: `060db6499f32faf8b98477b0a26969ef7d8b9987`.
- License: Apache-2.0; the upstream `LICENSE` is included unchanged. These
  third-party assets are not relicensed under this Go library's MIT license.
- Downloaded: 2026-09-10. No model weights or chat templates are included.
- Suite name: `qwen2-5`; Go pipeline: `model-only`.

All four files are unchanged upstream downloads from the pinned base URL:

```text
https://huggingface.co/Qwen/Qwen2.5-0.5B/resolve/060db6499f32faf8b98477b0a26969ef7d8b9987/
```

| File (append to base URL) | Bytes |
| --- | ---: |
| tokenizer.json | 7,031,645 |
| vocab.json | 2,776,833 |
| merges.txt | 1,671,839 |
| LICENSE | 11,343 |

SHA-256 hashes are pinned in `SHA256SUMS`; verify from this directory with
`shasum -a 256 -c SHA256SUMS`. Re-download these exact file names from the base
URL, not `main`. The Go tests check every hash and compare the exported vocabulary
and ordered merges with `tokenizer.json`.

The BPE model contains **151,643 vocabulary entries and 151,387 merge rules**.
The JSON also defines 22 added tokens; they are not part of the BPE-only workload.
HF applies NFC normalization, its configured Split regex, then ByteLevel with
`use_regex=false`. Do not substitute another split configuration or raw byte
spans. The benchmark consumes transformed pieces from the pinned HF 0.23.2 oracle and
validates Model IDs and piece-local byte offsets before timing.

Go does not yet implement this complete raw-text pipeline. ModelTokenize and
ModelAppendIDs are enabled; Go pre-tokenization and full encode are not.
