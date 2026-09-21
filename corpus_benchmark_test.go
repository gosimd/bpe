package bpe_test

import (
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/gosimd/bpe"
	"github.com/gosimd/bpe/internal/addedtoken"
	"github.com/gosimd/bpe/internal/benchsuite"
	"github.com/gosimd/bpe/internal/hforacle"
	"github.com/gosimd/bpe/internal/pretokenize"
)

type corpusBenchmarkPiece struct {
	value        string
	offsets      [2]int
	ids          []uint32
	added        bool
	localOffsets [][2]int
	recordIndex  int
	pieceIndex   int
}

type corpusBenchmarkRecord struct {
	text    string
	pieces  []corpusBenchmarkPiece
	ids     []uint32
	offsets [][2]int
}

type corpusBenchmarkData struct {
	model        *bpe.Model
	tokenizer    *bpe.Tokenizer
	addedTokens  *addedtoken.Matcher
	preTokenizer *pretokenize.Pipeline
	records      []corpusBenchmarkRecord
	modelPieces  []corpusBenchmarkPiece
	sourceBytes  int64
	modelBytes   int64
	pieceCount   int64
	tokenCount   int64
	modelTokens  int64
}

type corpusPretokenizedPiece struct {
	value   string
	offsets [2]int
	added   bool
}

var (
	corpusBenchmarkPretokenizedSink []corpusPretokenizedPiece
	corpusBenchmarkTokensSink       []bpe.Token
	corpusBenchmarkEncodingSink     bpe.Encoding
	corpusBenchmarkIDsSink          []uint32
	corpusBenchmarkCountSink        int
)

// BenchmarkBPE runs the selected catalogue's model/corpus cross-product.
// Selection and oracle generation are explicit; ordinary go test does neither.
func BenchmarkBPE(b *testing.B) {
	b.StopTimer()
	path := os.Getenv(benchsuite.Environment)
	if path == "" {
		b.Skip("run make benchmark-suite (missing BPE_BENCH_SUITE)")
	}
	root := os.Getenv(hforacle.RepositoryEnvironment)
	if root == "" {
		b.Fatal("BPE_HF_REPOSITORY must name the repository root")
	}
	suite, err := benchsuite.Load(path)
	if err != nil {
		b.Fatal(err)
	}
	suite, err = suite.Select(os.Getenv("BPE_BENCH_MODELS"), os.Getenv("BPE_BENCH_CORPORA"))
	if err != nil {
		b.Fatal(err)
	}
	for _, model := range suite.Models {
		b.Run(model.Name, func(b *testing.B) {
			for _, corpus := range suite.Corpora {
				b.Run(corpus.Name, func(b *testing.B) {
					b.StopTimer()
					if err := benchsuite.ValidateCorpus(root, corpus); err != nil {
						b.Fatal(err)
					}
					data := loadCorpusBenchmarkData(b, root, model, corpus.CorpusPath(root), benchsuite.ReferencePath(root, model, corpus))
					data.verify(b)
					runCorpusBenchmark(b, data)
				})
			}
		})
	}
}

func runCorpusBenchmark(b *testing.B, data *corpusBenchmarkData) {
	if data.tokenizer != nil {
		b.Run("PreTokenizer", func(b *testing.B) {
			benchmarkShape(b, data.sourceBytes, data.pieceCount, 0)
			b.ResetTimer()
			for range b.N {
				pieces := 0
				for _, record := range data.records {
					got := pretokenizeGPT2(b, data.preTokenizer, data.addedTokens, record.text)
					pieces += len(got)
					corpusBenchmarkPretokenizedSink = got
				}
				corpusBenchmarkCountSink = pieces
			}
			b.StopTimer()
			reportCorpusRates(b, data.sourceBytes, data.pieceCount, 0)
		})
	}

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
				corpusBenchmarkTokensSink = got
			}
			corpusBenchmarkCountSink = tokens
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
			corpusBenchmarkIDsSink = dst
			corpusBenchmarkCountSink = tokens
		}
		b.StopTimer()
		reportCorpusRates(b, data.modelBytes, int64(len(data.modelPieces)), data.modelTokens)
	})

	if data.tokenizer == nil {
		return
	}
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
				corpusBenchmarkEncodingSink = got
			}
			corpusBenchmarkCountSink = tokens
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
			corpusBenchmarkIDsSink = dst
			corpusBenchmarkCountSink = tokens
		}
		b.StopTimer()
		reportCorpusRates(b, data.sourceBytes, data.pieceCount, data.tokenCount)
	})
}

func loadCorpusBenchmarkData(b testing.TB, repository string, spec benchsuite.Model, corpusPath, referencePath string) *corpusBenchmarkData {
	b.Helper()
	if err := benchsuite.ValidateModelFiles(repository, spec); err != nil {
		b.Fatal(err)
	}
	options, err := benchsuite.ModelOptions(filepath.Join(repository, spec.Tokenizer), spec.Pipeline)
	if err != nil {
		b.Fatal(err)
	}
	model, err := bpe.OpenModel(filepath.Join(repository, spec.Vocab), filepath.Join(repository, spec.Merges), bpe.ModelOptions{
		UnknownToken: options.UnknownToken, ContinuingSubwordPrefix: options.ContinuingSubwordPrefix, EndOfWordSuffix: options.EndOfWordSuffix,
		FuseUnknown: options.FuseUnknown, ByteFallback: options.ByteFallback, IgnoreMerges: options.IgnoreMerges,
	})
	if err != nil {
		b.Fatal(err)
	}
	data := &corpusBenchmarkData{model: model}
	if spec.Pipeline == "gpt2" {
		endOfText := bpe.NewAddedToken("<|endoftext|>", 50256, true)
		endOfText.Normalized = true
		tokenizer, err := bpe.OpenTokenizer(filepath.Join(repository, spec.Tokenizer))
		if err != nil {
			b.Fatal(err)
		}

		data.tokenizer = tokenizer
		data.preTokenizer, err = pretokenize.NewByteLevel(false, true)
		if err != nil {
			b.Fatal(err)
		}
		data.addedTokens = addedtoken.New([]addedtoken.Definition{{
			ID: endOfText.ID, Content: endOfText.Content, Normalized: endOfText.Normalized,
		}})
	}

	reader, err := hforacle.Open(referencePath, corpusPath)
	if err != nil {
		b.Fatal(err)
	}
	defer func() {
		if closeErr := reader.Close(); closeErr != nil {
			b.Error(closeErr)
		}
	}()
	if err := reader.Metadata.ValidateTokenizer(filepath.Join(repository, spec.Tokenizer)); err != nil {
		b.Fatal(err)
	}

	for {
		fixture, ok, err := reader.Next()
		if err != nil {
			b.Fatal(err)
		}
		if !ok {
			break
		}
		record := corpusBenchmarkRecord{
			text:    fixture.Text,
			ids:     append([]uint32(nil), fixture.Record.Encoding.IDs...),
			offsets: append([][2]int(nil), fixture.Record.Encoding.Offsets...),
		}
		data.sourceBytes += int64(len(fixture.Text))
		data.tokenCount += int64(len(record.ids))
		for _, piece := range fixture.Record.PreTokenization {
			prepared := corpusBenchmarkPiece{
				value:        piece.Value,
				offsets:      piece.Offsets,
				ids:          append([]uint32(nil), piece.Model.IDs...),
				added:        piece.Model.Source == "added_token",
				localOffsets: append([][2]int(nil), piece.Model.LocalOffsets...),
				recordIndex:  fixture.Record.Index,
				pieceIndex:   piece.Index,
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
	if data.sourceBytes == 0 || data.modelBytes == 0 {
		b.Fatal("benchmark requires nonempty source text and BPE pieces")
	}
	return data
}

func (data *corpusBenchmarkData) verify(b testing.TB) {
	b.Helper()
	if data.tokenizer != nil {
		for recordIndex, record := range data.records {
			gotPieces := pretokenizeGPT2(b, data.preTokenizer, data.addedTokens, record.text)
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
			gotOffsets := make([][2]int, len(encoding.Offsets))
			for i, offset := range encoding.Offsets {
				gotOffsets[i] = [2]int{offset.Start, offset.End}
			}
			if !slices.Equal(gotOffsets, record.offsets) {
				b.Fatalf("Tokenizer.Encode record %d: offsets differ from HF", recordIndex)
			}
			dst, err := data.tokenizer.AppendIDsWithOptions(nil, record.text, bpe.EncodeOptions{AddSpecialTokens: false})
			if err != nil {
				b.Fatal(err)
			}
			if err := equalIDs(dst, record.ids); err != nil {
				b.Fatalf("Tokenizer.AppendIDs record %d: %v", recordIndex, err)
			}
		}
	}

	for _, piece := range data.modelPieces {
		tokens, err := data.model.Tokenize(piece.value)
		if err != nil {
			b.Fatal(err)
		}
		got := make([]uint32, len(tokens))
		for index := range tokens {
			got[index] = tokens[index].ID
		}
		gotOffsets := make([][2]int, len(tokens))
		for i, token := range tokens {
			gotOffsets[i] = [2]int{token.Offset.Start, token.Offset.End}
		}
		if !slices.Equal(gotOffsets, piece.localOffsets) {
			b.Fatalf("Model.Tokenize record %d piece %d %q: offsets differ from HF", piece.recordIndex, piece.pieceIndex, piece.value)
		}
		if err := equalIDs(got, piece.ids); err != nil {
			b.Fatalf("Model.Tokenize record %d piece %d %q: %v", piece.recordIndex, piece.pieceIndex, piece.value, err)
		}
		got, err = data.model.AppendIDs(got[:0], piece.value)
		if err != nil {
			b.Fatal(err)
		}
		if err := equalIDs(got, piece.ids); err != nil {
			b.Fatalf("Model.AppendIDs record %d piece %d %q: %v", piece.recordIndex, piece.pieceIndex, piece.value, err)
		}
	}
}

func pretokenizeGPT2(b testing.TB, pipeline *pretokenize.Pipeline, matcher *addedtoken.Matcher, text string) []corpusPretokenizedPiece {
	segments := matcher.Split(text)
	result := make([]corpusPretokenizedPiece, 0, len(segments))
	for _, segment := range segments {
		if segment.Added {
			result = append(result, corpusPretokenizedPiece{
				value:   text[segment.Start:segment.End],
				offsets: [2]int{segment.Start, segment.End},
				added:   true,
			})
			continue
		}
		ordinary := text[segment.Start:segment.End]
		err := pipeline.Walk(ordinary, func(piece pretokenize.Piece) error {
			start, end, err := piece.OriginalRange(0, len(piece.Text()))
			if err != nil {
				return err
			}
			result = append(result, corpusPretokenizedPiece{
				value:   strings.Clone(piece.Text()),
				offsets: [2]int{segment.Start + start, segment.Start + end},
			})
			return nil
		})
		if err != nil {
			b.Fatal(err)
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
	// ResetTimer clears custom metrics, so publish shape counts after timing.
	benchmarkShape(b, inputBytes, pieces, tokens)
	seconds := b.Elapsed().Seconds()
	operations := float64(b.N)
	b.ReportMetric(seconds*1e9/(operations*float64(inputBytes)), "ns/input-byte")
	b.ReportMetric(operations*float64(pieces)/seconds, "pieces/s")
	if tokens != 0 {
		b.ReportMetric(operations*float64(tokens)/seconds, "tokens/s")
	}
}
