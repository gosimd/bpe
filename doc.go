// Package bpe provides an inference-oriented, Hugging Face-compatible BPE model.
//
// Model operates on one already pre-tokenized piece. Tokenizer adds AddedToken
// matching plus standard GPT-2 ByteLevel splitting and byte mapping for raw
// text; other tokenizer-pipeline stages will be provided separately. See
// docs/Model.md and docs/Tokenizer.md for the public contracts and
// compatibility scope.
package bpe
