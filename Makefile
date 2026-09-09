CARGO ?= cargo

HF_REFERENCE_MANIFEST := tools/hf-reference/Cargo.toml
HF_REFERENCE := $(CARGO) run --locked --release --manifest-path $(HF_REFERENCE_MANIFEST) --
HF_GPT2_TOKENIZER := testdata/gpt2/tokenizer.json
HF_WIKITEXT2_CORPUS := testdata/wikitext2/validation.jsonl
HF_GPT2_FIXTURE := testdata/generated/hf_gpt2_wikitext2_v0.23.2.jsonl

.PHONY: generate-fixtures
generate-fixtures:
	$(HF_REFERENCE) generate \
		--tokenizer $(HF_GPT2_TOKENIZER) \
		--corpus $(HF_WIKITEXT2_CORPUS) \
		--output $(HF_GPT2_FIXTURE)
