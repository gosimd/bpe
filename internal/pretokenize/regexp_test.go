package pretokenize

import (
	"errors"
	"regexp/syntax"
	"slices"
	"testing"
)

func TestRegexpGapBridge(t *testing.T) {
	r, err := NewRegexp("[0-9]")
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		input string
		want  [][2]int
	}{
		{"", nil}, {"abc", [][2]int{{0, 3}}}, {"1", [][2]int{{0, 1}}},
		{"α12猫", [][2]int{{0, 2}, {2, 3}, {3, 4}, {4, 7}}},
	} {
		var got [][2]int
		err := r.Split(c.input, func(start, end int) { got = append(got, [2]int{start, end}) })
		if err != nil || !slices.Equal(got, c.want) {
			t.Fatalf("%q: %v, %v", c.input, got, err)
		}
	}
}

func TestRegexpBackendSelection(t *testing.T) {
	fast, err := NewRegexp("[0-9]")
	if err != nil || fast.compiled == nil || fast.fallback != nil {
		t.Fatalf("registered expression must use pretok: %v, %v", fast, err)
	}
	for _, pattern := range []string{"[a-z]+", "[0123456789]", "", "^ab|12$"} {
		r, err := NewRegexp(pattern)
		if err != nil || r.fallback == nil || r.compiled != nil {
			t.Fatalf("%q must use regexp: %v, %v", pattern, r, err)
		}
		if err := r.Split("ok\xff", func(int, int) { t.Fatal("emitted invalid input") }); !errors.Is(err, ErrInvalidUTF8) {
			t.Fatal(err)
		}
	}
	for _, pattern := range []string{"[", "(?=a)", `(a)\1`} {
		r, err := NewRegexp(pattern)
		var syntaxErr *syntax.Error
		if r != nil || !errors.Is(err, ErrUnsupportedPattern) || !errors.As(err, &syntaxErr) {
			t.Fatalf("%q: expected wrapped Go syntax error, got %v", pattern, err)
		}
	}
}

func TestRegexpFallbackBoundaries(t *testing.T) {
	for _, c := range []struct {
		pattern, input string
		want           [][2]int
	}{
		{"[a-z]+", "猫ab12cd🙂", [][2]int{{0, 3}, {3, 5}, {5, 7}, {7, 9}, {9, 13}}},
		{"^ab|12$", "ab猫ab12", [][2]int{{0, 2}, {2, 7}, {7, 9}}},
		{`\b`, "ab cd", [][2]int{{0, 2}, {2, 3}, {3, 5}}},
		{"", "a猫", [][2]int{{0, 1}, {1, 4}}},
		{"^|$", "a猫", [][2]int{{0, 4}}},
		{"a*", "aa猫bb", [][2]int{{0, 2}, {2, 5}, {5, 6}, {6, 7}}},
		{"a|ab", "ab", [][2]int{{0, 1}, {1, 2}}},
		{"", "", nil},
	} {
		r, err := NewRegexp(c.pattern)
		if err != nil {
			t.Fatal(err)
		}
		var got [][2]int
		err = r.Split(c.input, func(start, end int) { got = append(got, [2]int{start, end}) })
		if err != nil || !slices.Equal(got, c.want) {
			t.Fatalf("%q / %q: %v, %v; want %v", c.pattern, c.input, got, err, c.want)
		}
	}
}

func TestRegexpValidationBeforeEmission(t *testing.T) {
	r, err := NewRegexp("[0-9]")
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		r     *Regexp
		input string
		want  error
	}{
		{nil, "1", ErrUninitialized}, {new(Regexp), "\xff", ErrUninitialized}, {r, "12\xff", ErrInvalidUTF8},
	} {
		err := c.r.Split(c.input, func(int, int) { t.Error("emitted before validation") })
		if !errors.Is(err, c.want) {
			t.Fatalf("got %v, want %v", err, c.want)
		}
	}
}
