package jsoncheck

import (
	"encoding/json"
	"strings"
	"testing"
	"unicode/utf8"
)

func TestValidate(t *testing.T) {
	for _, data := range []string{
		`null`, `true`, `123`, `[]`, `{}`, `"猫�😀"`,
		`{"a":0,"nested":{"a":1}}`, `[{"a":0},{"a":1}]`,
		`{"\uD83D\uDE00":0,"\ufffd":1,"\\ud800":2}`,
		`{"a\"b":"\\\"","a\\b":"\u0061"}`,
		strings.Repeat("[", 80) + "0" + strings.Repeat("]", 80),
	} {
		t.Run("valid/"+data, func(t *testing.T) {
			if err := Validate([]byte(data)); err != nil {
				t.Fatal(err)
			}
		})
	}
	for _, data := range []string{
		``, `{"a":0,"a":1}`, `{"a":0,"\u0061":1}`,
		`{"x":{"猫":0,"\u732b":1}}`, `{"x":"\ud800"}`,
		`"\udc00"`, `"\ud800x"`, `"\ud800\ud800"`, `"\ud800\u1234"`,
		`"\u12"`, `"\uGGGG"`, "\"\xff\"", "\"\xe2\x82\"",
		`{} []`, `{} garbage`, `{"a":1,}`, `[1}`, `{`, `"unfinished`,
		strings.Repeat("[", 81) + "0" + strings.Repeat("]", 81),
	} {
		t.Run("invalid/"+data, func(t *testing.T) {
			if err := Validate([]byte(data)); err == nil {
				t.Fatal("accepted invalid JSON")
			}
		})
	}
}

func FuzzValidate(f *testing.F) {
	for _, data := range []string{`{"a":0}`, `{"a":0,"\u0061":1}`, `"\ud800"`, `"\ud83d\ude00"`, `"\\ud800"`, "\xff"} {
		f.Add([]byte(data))
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		if err := Validate(data); err == nil && (!json.Valid(data) || !utf8.Valid(data)) {
			t.Fatal("accepted malformed JSON or UTF-8")
		}
		// A valid UTF-8 string marshaled by encoding/json must remain accepted,
		// including literal backslashes, quotation marks and replacement runes.
		if utf8.Valid(data) {
			encoded, err := json.Marshal(string(data))
			if err != nil {
				t.Fatal(err)
			}
			if err := Validate(encoded); err != nil {
				t.Fatalf("rejected valid JSON string %q: %v", encoded, err)
			}
		}
	})
}
