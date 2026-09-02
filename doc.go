// Package bpe provides an inference-oriented, Hugging Face-compatible BPE model.
//
// Model operates on one already pre-tokenized piece. It does not normalize or
// split raw text; those tokenizer-pipeline stages will be provided separately.
// See docs/Model.md for the public contract and compatibility scope.
package bpe
