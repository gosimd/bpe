# Qwen3.8-Flash-Next tokenizer assets

- Repository: [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).
- Exact revision: `de4b8e4d43b917e7706784d8bb445c9af86a3540`.
- Upstream license: **Qwen Community License 1.0** (`qwen-community-1.0`),
  not Apache-2.0 or the Go library's MIT license. The complete upstream
  `LICENSE`, including its attribution and commercial-use conditions, is
  included unchanged. Retain and review it when using or redistributing assets.
- Downloaded: 2026-09-10. Only text-tokenizer assets and their license are
  included: no model weights, chat templates, image/video processors or code.
- Suite name: `qwen3-8`; Go pipeline: `model-only`.

All four files are unchanged upstream downloads from this exact base URL:

```text
https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/de4b8e4d43b917e7706784d8bb445c9af86a3540/
```

| File (append to base URL) | Bytes |
| --- | ---: |
| tokenizer.json | 12,809,320 |
| vocab.json | 6,722,759 |
| merges.txt | 3,353,259 |
| LICENSE | 3,235 |

SHA-256 hashes are pinned in `SHA256SUMS`; verify from this directory with
`shasum -a 256 -c SHA256SUMS`. To reproduce the input snapshot, download these
exact names from the base URL, not `main`. Go tests verify all hashes and
compare every vocabulary entry and ordered merge against `tokenizer.json`.

The BPE model contains **248,044 vocabulary entries and 247,587 merge rules**.
The JSON defines 33 added tokens; pieces extracted as added tokens are excluded
from the BPE-only benchmark. Counts describe the tokenizer, not model embeddings.

HF applies NFC, the serialized Split regex, and ByteLevel with
`use_regex=false`, followed by ByteLevel post-processing. The Split regex
includes Unicode marks (`\p{M}`), unlike the pinned Qwen2.5 regex; the models
must use their own HF-generated pieces. The regex also matches the pinned
upstream `tokenizer_config.json`'s `pretokenize_regex` field. The benchmark's
behavioral reference is HF Tokenizers 0.23.2 loading `tokenizer.json`, not
Transformers chat or multimodal processing.

Go ModelTokenize and ModelAppendIDs consume the transformed HF pieces and
validate IDs and piece-local byte offsets before timing. Go pre-tokenization
and full encoding are not enabled for this model; this entry does not claim
support for its complete raw-text or multimodal pipeline.

From the repository root:

```sh
make test-reference MODELS=qwen3-8
make benchmark-suite MODELS=qwen3-8
```

Generated full-corpus oracles stay under ignored `testdata/generated/`.
