package bpe_test

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/gosimd/bpe"
	"github.com/gosimd/bpe/internal/addedtoken"
	"github.com/gosimd/bpe/internal/bytelevel"
	"github.com/gosimd/bpe/internal/hforacle"
)

type gpt2BenchmarkPiece struct {
	value   string
	offsets [2]int
	ids     []uint32
	added   bool
}

type gpt2BenchmarkRecord struct {
	text   string
	pieces []gpt2BenchmarkPiece
	ids    []uint32
}

type gpt2BenchmarkData struct {
	model       *bpe.Model
	tokenizer   *bpe.Tokenizer
	addedTokens *addedtoken.Matcher
	records     []gpt2BenchmarkRecord
	modelPieces []gpt2BenchmarkPiece
	sourceBytes int64
	modelBytes  int64
	pieceCount  int64
	tokenCount  int64
	modelTokens int64
}

type gpt2PretokenizedPiece struct {
	value   string
	offsets [2]int
	added   bool
}

var (
	gpt2BenchmarkPretokenizedSink []gpt2PretokenizedPiece
	gpt2BenchmarkTokensSink       []bpe.Token
	gpt2BenchmarkEncodingSink     bpe.Encoding
	gpt2BenchmarkIDsSink          []uint32
	gpt2BenchmarkCountSink        int
)

// BenchmarkGPT2WikiText2 measures one complete, fixed-order pass over the
// pinned WikiText-2 raw validation split. It is opt-in because its prepared
// model pieces come from the ignored Hugging Face oracle.
func BenchmarkGPT2WikiText2(b *testing.B) {
	b.StopTimer()
	data := loadGPT2BenchmarkData(b)
	data.verify(b)

	b.Run("PreTokenizer", func(b *testing.B) {
		benchmarkShape(b, data.sourceBytes, data.pieceCount, 0)
		b.ResetTimer()
		for range b.N {
			pieces := 0
			for _, record := range data.records {
				got := pretokenizeGPT2(data.addedTokens, record.text)
				pieces += len(got)
				gpt2BenchmarkPretokenizedSink = got
			}
			gpt2BenchmarkCountSink = pieces
		}
		b.StopTimer()
		reportCorpusRates(b, data.sourceBytes, data.pieceCount, 0)
	})

	b.Run("ModelTokenize", func(b *testing.B) {
		benchmarkShape(b, data.modelBytes, int64(len(data.modelPieces)), data.modelTokens)
		b.ResetTimer()
		for range b.N {
			tokens := 0
			for _, piece := range data.modelPieces {
				got, err := data.model.Tokenize(piece.value)
				if err != nil {
					b.Fatal(err)
				}
				tokens += len(got)
				gpt2BenchmarkTokensSink = got
			}
			gpt2BenchmarkCountSink = tokens
		}
		b.StopTimer()
		reportCorpusRates(b, data.modelBytes, int64(len(data.modelPieces)), data.modelTokens)
	})

	b.Run("ModelAppendIDs", func(b *testing.B) {
		benchmarkShape(b, data.modelBytes, int64(len(data.modelPieces)), data.modelTokens)
		dst := make([]uint32, 0, 16)
		b.ResetTimer()
		for range b.N {
			tokens := 0
			for _, piece := range data.modelPieces {
				var err error
				dst, err = data.model.AppendIDs(dst[:0], piece.value)
				if err != nil {
					b.Fatal(err)
				}
				tokens += len(dst)
			}
			gpt2BenchmarkIDsSink = dst
			gpt2BenchmarkCountSink = tokens
		}
		b.StopTimer()
		reportCorpusRates(b, data.modelBytes, int64(len(data.modelPieces)), data.modelTokens)
	})

	b.Run("TokenizerEncode", func(b *testing.B) {
		benchmarkShape(b, data.sourceBytes, data.pieceCount, data.tokenCount)
		b.ResetTimer()
		for range b.N {
			tokens := 0
			for _, record := range data.records {
				got, err := data.tokenizer.EncodeWithOptions(record.text, bpe.EncodeOptions{AddSpecialTokens: false})
				if err != nil {
					b.Fatal(err)
				}
				tokens += len(got.IDs)
				gpt2BenchmarkEncodingSink = got
			}
			gpt2BenchmarkCountSink = tokens
		}
		b.StopTimer()
		reportCorpusRates(b, data.sourceBytes, data.pieceCount, data.tokenCount)
	})

	b.Run("TokenizerAppendIDs", func(b *testing.B) {
		benchmarkShape(b, data.sourceBytes, data.pieceCount, data.tokenCount)
		dst := make([]uint32, 0, 256)
		b.ResetTimer()
		for range b.N {
			tokens := 0
			for _, record := range data.records {
				var err error
				dst, err = data.tokenizer.AppendIDsWithOptions(dst[:0], record.text, bpe.EncodeOptions{AddSpecialTokens: false})
				if err != nil {
					b.Fatal(err)
				}
				tokens += len(dst)
			}
			gpt2BenchmarkIDsSink = dst
			gpt2BenchmarkCountSink = tokens
		}
		b.StopTimer()
		reportCorpusRates(b, data.sourceBytes, data.pieceCount, data.tokenCount)
	})
}

func loadGPT2BenchmarkData(b *testing.B) *gpt2BenchmarkData {
	b.Helper()
	referencePath := os.Getenv(hforacle.ReferenceEnvironment)
	if referencePath == "" {
		b.Skipf("full Hugging Face corpus oracle is opt-in; run make benchmark-go-reference (missing %s)", hforacle.ReferenceEnvironment)
	}
	repository := os.Getenv(hforacle.RepositoryEnvironment)
	if repository == "" {
		b.Fatalf("%s must name the repository root", hforacle.RepositoryEnvironment)
	}

	model, err := bpe.OpenModel(
		filepath.Join(repository, "testdata", "gpt2", "vocab.json"),
		filepath.Join(repository, "testdata", "gpt2", "merges.txt"),
		bpe.ModelOptions{},
	)
	if err != nil {
		b.Fatal(err)
	}
	endOfText := bpe.NewAddedToken("<|endoftext|>", 50256, true)
	endOfText.Normalized = true
	tokenizer, err := bpe.NewTokenizerWithOptions(model, bpe.TokenizerOptions{
		AddPrefixSpace: false,
		AddedTokens:    []bpe.AddedToken{endOfText},
		PostProcessor: bpe.ByteLevelPostProcessor{
			AddPrefixSpace: true,
			TrimOffsets:    false,
		},
	})
	if err != nil {
		b.Fatal(err)
	}

	reader, err := hforacle.Open(
		referencePath,
		filepath.Join(repository, "testdata", "wikitext2", "validation.jsonl"),
	)
	if err != nil {
		b.Fatal(err)
	}
	defer func() {
		if closeErr := reader.Close(); closeErr != nil {
			b.Error(closeErr)
		}
	}()
	if err := reader.Metadata.ValidateTokenizer(filepath.Join(repository, "testdata", "gpt2", "tokenizer.json")); err != nil {
		b.Fatal(err)
	}

	data := &gpt2BenchmarkData{
		model:     model,
		tokenizer: tokenizer,
		addedTokens: addedtoken.New([]addedtoken.Definition{{
			ID:         endOfText.ID,
			Content:    endOfText.Content,
			Normalized: endOfText.Normalized,
		}}),
	}
	for {
		fixture, ok, err := reader.Next()
		if err != nil {
			b.Fatal(err)
		}
		if !ok {
			break
		}
		record := gpt2BenchmarkRecord{
			text: fixture.Text,
			ids:  append([]uint32(nil), fixture.Record.Encoding.IDs...),
		}
		data.sourceBytes += int64(len(fixture.Text))
		data.tokenCount += int64(len(record.ids))
		for _, piece := range fixture.Record.PreTokenization {
			prepared := gpt2BenchmarkPiece{
				value:   piece.Value,
				offsets: piece.Offsets,
				ids:     append([]uint32(nil), piece.Model.IDs...),
				added:   piece.Model.Source == "added_token",
			}
			record.pieces = append(record.pieces, prepared)
			data.pieceCount++
			if piece.Model.Source == "bpe" {
				data.modelBytes += int64(len(piece.Value))
				data.modelTokens += int64(len(piece.Model.IDs))
				data.modelPieces = append(data.modelPieces, prepared)
			} else if piece.Model.Source != "added_token" {
				b.Fatalf("record %d has unsupported piece source %q", fixture.Record.Index, piece.Model.Source)
			}
		}
		data.records = append(data.records, record)
	}
	return data
}

func (data *gpt2BenchmarkData) verify(b *testing.B) {
	b.Helper()
	for recordIndex, record := range data.records {
		gotPieces := pretokenizeGPT2(data.addedTokens, record.text)
		if len(gotPieces) != len(record.pieces) {
			b.Fatalf("pre-tokenizer record %d produced %d pieces, want %d", recordIndex, len(gotPieces), len(record.pieces))
		}
		for pieceIndex, got := range gotPieces {
			want := record.pieces[pieceIndex]
			if got.value != want.value || got.offsets != want.offsets || got.added != want.added {
				b.Fatalf("pre-tokenizer record %d piece %d = {%q %v added=%t}, want {%q %v added=%t}", recordIndex, pieceIndex, got.value, got.offsets, got.added, want.value, want.offsets, want.added)
			}
		}

		encoding, err := data.tokenizer.EncodeWithOptions(record.text, bpe.EncodeOptions{AddSpecialTokens: false})
		if err != nil {
			b.Fatal(err)
		}
		if err := equalIDs(encoding.IDs, record.ids); err != nil {
			b.Fatalf("Tokenizer.Encode record %d: %v", recordIndex, err)
		}
		dst, err := data.tokenizer.AppendIDsWithOptions(nil, record.text, bpe.EncodeOptions{AddSpecialTokens: false})
		if err != nil {
			b.Fatal(err)
		}
		if err := equalIDs(dst, record.ids); err != nil {
			b.Fatalf("Tokenizer.AppendIDs record %d: %v", recordIndex, err)
		}
	}

	for pieceIndex, piece := range data.modelPieces {
		tokens, err := data.model.Tokenize(piece.value)
		if err != nil {
			b.Fatal(err)
		}
		got := make([]uint32, len(tokens))
		for index := range tokens {
			got[index] = tokens[index].ID
		}
		if err := equalIDs(got, piece.ids); err != nil {
			b.Fatalf("Model.Tokenize piece %d: %v", pieceIndex, err)
		}
		got, err = data.model.AppendIDs(got[:0], piece.value)
		if err != nil {
			b.Fatal(err)
		}
		if err := equalIDs(got, piece.ids); err != nil {
			b.Fatalf("Model.AppendIDs piece %d: %v", pieceIndex, err)
		}
	}
}

func pretokenizeGPT2(matcher *addedtoken.Matcher, text string) []gpt2PretokenizedPiece {
	segments := matcher.Split(text)
	result := make([]gpt2PretokenizedPiece, 0, len(segments))
	for _, segment := range segments {
		if segment.Added {
			result = append(result, gpt2PretokenizedPiece{
				value:   text[segment.Start:segment.End],
				offsets: [2]int{segment.Start, segment.End},
				added:   true,
			})
			continue
		}
		ordinary := text[segment.Start:segment.End]
		for _, piece := range bytelevel.Split(ordinary) {
			mapped := bytelevel.Map(ordinary[piece.Start:piece.End], piece.Start, func(offset int) int { return offset })
			start, end := mapped.RawRange(0, len(mapped.Text))
			result = append(result, gpt2PretokenizedPiece{
				value:   mapped.Text,
				offsets: [2]int{segment.Start + start, segment.Start + end},
			})
		}
	}
	return result
}

func equalIDs(got, want []uint32) error {
	if len(got) != len(want) {
		return fmt.Errorf("got %d IDs, want %d", len(got), len(want))
	}
	for index := range got {
		if got[index] != want[index] {
			return fmt.Errorf("ID %d is %d, want %d", index, got[index], want[index])
		}
	}
	return nil
}

func benchmarkShape(b *testing.B, inputBytes, pieces, tokens int64) {
	b.Helper()
	b.ReportAllocs()
	b.SetBytes(inputBytes)
	b.ReportMetric(float64(inputBytes), "input-bytes/op")
	b.ReportMetric(float64(pieces), "pieces/op")
	if tokens != 0 {
		b.ReportMetric(float64(tokens), "tokens/op")
	}
}

func reportCorpusRates(b *testing.B, inputBytes, pieces, tokens int64) {
	b.Helper()
	seconds := b.Elapsed().Seconds()
	operations := float64(b.N)
	b.ReportMetric(seconds*1e9/(operations*float64(inputBytes)), "ns/input-byte")
	b.ReportMetric(operations*float64(pieces)/seconds, "pieces/s")
	if tokens != 0 {
		b.ReportMetric(operations*float64(tokens)/seconds, "tokens/s")
	}
}
