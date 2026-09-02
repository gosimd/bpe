package bpe

import (
	"fmt"
	"unicode/utf8"

	"github.com/gosimd/bpe/internal/postprocess"
)

// PostProcessor is a supported immutable tokenizer post-processor
// configuration. Implementations are provided by this package; callers cannot
// supply arbitrary processing code.
type PostProcessor interface {
	postProcessor()
}

// ByteLevelPostProcessor applies Hugging Face ByteLevel offset trimming after
// tokenization. It does not add, remove, or reorder token IDs.
type ByteLevelPostProcessor struct {
	AddPrefixSpace bool
	TrimOffsets    bool
}

func (ByteLevelPostProcessor) postProcessor() {}

// NewByteLevelPostProcessor returns Hugging Face ByteLevel's default
// post-processor configuration.
func NewByteLevelPostProcessor() ByteLevelPostProcessor {
	return ByteLevelPostProcessor{AddPrefixSpace: true, TrimOffsets: true}
}

// TemplateProcessing inserts special tokens and assigns type IDs according to
// Hugging Face TemplateProcessing syntax.
//
// Empty Single and Pair values use "$A" and "$A:0 $B:1" respectively.
type TemplateProcessing struct {
	Single        string
	Pair          string
	SpecialTokens []TemplateSpecialToken
}

func (TemplateProcessing) postProcessor() {}

// TemplateSpecialToken defines one special-token identifier referenced by a
// TemplateProcessing template. A definition may emit multiple IDs and token
// strings; IDs and Tokens must have equal length.
type TemplateSpecialToken struct {
	Identifier string
	IDs        []uint32
	Tokens     []string
}

// NewTemplateSpecialToken constructs the common one-identifier, one-token
// special-token definition.
func NewTemplateSpecialToken(identifier string, id uint32) TemplateSpecialToken {
	return TemplateSpecialToken{
		Identifier: identifier,
		IDs:        []uint32{id},
		Tokens:     []string{identifier},
	}
}

func compilePostProcessor(configuration PostProcessor) (postprocess.Processor, error) {
	switch processor := configuration.(type) {
	case nil:
		return nil, nil
	case ByteLevelPostProcessor:
		return postprocess.NewByteLevel(processor.AddPrefixSpace, processor.TrimOffsets), nil
	case *ByteLevelPostProcessor:
		if processor == nil {
			return nil, fmt.Errorf("%w: nil *ByteLevelPostProcessor", ErrInvalidPostProcessor)
		}
		return postprocess.NewByteLevel(processor.AddPrefixSpace, processor.TrimOffsets), nil
	case TemplateProcessing:
		return compileTemplateProcessing(processor)
	case *TemplateProcessing:
		if processor == nil {
			return nil, fmt.Errorf("%w: nil *TemplateProcessing", ErrInvalidPostProcessor)
		}
		return compileTemplateProcessing(*processor)
	default:
		return nil, fmt.Errorf("%w: unsupported configuration %T", ErrInvalidPostProcessor, configuration)
	}
}

func compileTemplateProcessing(configuration TemplateProcessing) (postprocess.Processor, error) {
	if !utf8.ValidString(configuration.Single) {
		return nil, fmt.Errorf("%w: single template is not valid UTF-8", ErrInvalidTemplate)
	}
	if !utf8.ValidString(configuration.Pair) {
		return nil, fmt.Errorf("%w: pair template is not valid UTF-8", ErrInvalidTemplate)
	}
	specials := make([]postprocess.SpecialToken, len(configuration.SpecialTokens))
	seen := make(map[string]int, len(configuration.SpecialTokens))
	for index, special := range configuration.SpecialTokens {
		if special.Identifier == "" {
			return nil, fmt.Errorf("%w: special token %d has an empty identifier", ErrInvalidTemplate, index)
		}
		if !utf8.ValidString(special.Identifier) {
			return nil, fmt.Errorf("%w: special token %d identifier is not valid UTF-8", ErrInvalidTemplate, index)
		}
		if previous, ok := seen[special.Identifier]; ok {
			return nil, fmt.Errorf("%w: special tokens %d and %d have duplicate identifier %q", ErrInvalidTemplate, previous, index, special.Identifier)
		}
		if len(special.IDs) != len(special.Tokens) {
			return nil, fmt.Errorf("%w: special token %q has %d IDs and %d token strings", ErrInvalidTemplate, special.Identifier, len(special.IDs), len(special.Tokens))
		}
		for tokenIndex, token := range special.Tokens {
			if !utf8.ValidString(token) {
				return nil, fmt.Errorf("%w: special token %q value %d is not valid UTF-8", ErrInvalidTemplate, special.Identifier, tokenIndex)
			}
		}
		seen[special.Identifier] = index
		specials[index] = postprocess.SpecialToken{
			Identifier: special.Identifier,
			IDs:        append([]uint32(nil), special.IDs...),
			Tokens:     append([]string(nil), special.Tokens...),
		}
	}

	processor, err := postprocess.NewTemplate(configuration.Single, configuration.Pair, specials)
	if err != nil {
		return nil, fmt.Errorf("%w: %v", ErrInvalidTemplate, err)
	}
	return processor, nil
}
