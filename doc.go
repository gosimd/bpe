// Package bpe provides an inference-oriented, Hugging Face-compatible BPE model.
//
// Model operates on one already pre-tokenized piece. Tokenizer adds AddedToken
// matching and a compiled pre-tokenizer pipeline for raw text, then optionally
// applies TemplateProcessing or ByteLevel offset post-processing.
// See docs/Model.md, docs/Tokenizer.md, and
// docs/PostProcessors.md for the public contracts and compatibility scope.
package bpe
