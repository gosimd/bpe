package bpe_test

import (
	"fmt"

	"github.com/gosimd/bpe"
)

// ExampleParseTokenizer is the self-contained README quick start.
func ExampleParseTokenizer() {
	tokenizer, err := bpe.ParseTokenizer([]byte(`{
        "model": {
            "type": "BPE",
            "vocab": {"h": 0, "i": 1, "hi": 2},
            "merges": [["h", "i"]]
        },
        "pre_tokenizer": {"type": "WhitespaceSplit"}
    }`))
	if err != nil {
		panic(err)
	}

	encoding, err := tokenizer.Encode("hi hi")
	if err != nil {
		panic(err)
	}
	fmt.Println(encoding.IDs)

	// Reuse the destination for independent inputs when only IDs are needed.
	ids := make([]uint32, 0, 32)
	ids, err = tokenizer.AppendIDs(ids[:0], "hi")
	if err != nil {
		panic(err)
	}
	fmt.Println(ids)
	// Output:
	// [2 2]
	// [2]
}
