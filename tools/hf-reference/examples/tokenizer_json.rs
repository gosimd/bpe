//! Compact loader/pipeline oracle. Explicit tooling; never executed by go test.
use anyhow::{anyhow, Result};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{env, fs};
use tokenizers::Tokenizer;

fn main() -> Result<()> {
    let output = env::args()
        .nth(1)
        .ok_or_else(|| anyhow!("supply output path"))?;
    let source = "testdata/gpt2/vocab.json";
    let bytes = fs::read(source)?;
    let mut vocab: Value = serde_json::from_slice(&bytes)?;
    vocab
        .as_object_mut()
        .unwrap()
        .retain(|_, id| id.as_u64().unwrap() < 256);
    for token in [
        " ", "猫", "中文", "中", "文", "日", "本", "Я", "ع", "ر", "ب", "ي", "🙂", "▁", "\u{301}",
        "\r", "\n", "\t", "ab",
    ] {
        if vocab.get(token).is_none() {
            let id = vocab.as_object().unwrap().len();
            vocab[token] = json!(id);
        }
    }
    let next = vocab.as_object().unwrap().len();
    let model = json!({"type":"BPE","dropout":null,"unk_token":null,
        "continuing_subword_prefix":null,"end_of_word_suffix":null,
        "fuse_unk":false,"byte_fallback":false,"ignore_merges":false,
        "vocab":vocab,"merges":[["a","b"]]});
    let mut stages = vec![
        ("identity".to_owned(), Value::Null),
        (
            "bytelevel".to_owned(),
            json!({"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true}),
        ),
        (
            "bytelevel-prefix".to_owned(),
            json!({"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true}),
        ),
        (
            "bytelevel-no-regex".to_owned(),
            json!({"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}),
        ),
        ("whitespace".to_owned(), json!({"type":"Whitespace"})),
        (
            "whitespace-split".to_owned(),
            json!({"type":"WhitespaceSplit"}),
        ),
        (
            "digits".to_owned(),
            json!({"type":"Digits","individual_digits":false}),
        ),
        (
            "individual-digits".to_owned(),
            json!({"type":"Digits","individual_digits":true}),
        ),
        ("bert".to_owned(), json!({"type":"BertPreTokenizer"})),
        ("scripts".to_owned(), json!({"type":"UnicodeScripts"})),
        (
            "delimiter".to_owned(),
            json!({"type":"CharDelimiterSplit","delimiter":"|"}),
        ),
        (
            "split-regex".to_owned(),
            json!({"type":"Split","pattern":{"Regex":"[0-9]"},"behavior":"Isolated","invert":false}),
        ),
        (
            "sequence".to_owned(),
            json!({"type":"Sequence","pretokenizers":[
            {"type":"Split","pattern":{"Regex":"[0-9]"},"behavior":"Isolated","invert":false},
            {"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":false}]}),
        ),
        (
            "nested-sequence".to_owned(),
            json!({"type":"Sequence","pretokenizers":[
            {"type":"WhitespaceSplit"},{"type":"Sequence","pretokenizers":[
            {"type":"Digits","individual_digits":true},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}]}]}),
        ),
    ];
    for mode in [
        "Isolated",
        "Removed",
        "Contiguous",
        "MergedWithPrevious",
        "MergedWithNext",
    ] {
        stages.push((
            format!("punctuation-{mode}"),
            json!({"type":"Punctuation","behavior":mode}),
        ));
        stages.push((
            format!("literal-{mode}"),
            json!({"type":"Split","pattern":{"String":"|"},"behavior":mode,"invert":false}),
        ));
    }
    stages.push((
        "literal-invert".to_owned(),
        json!({"type":"Split","pattern":{"String":"|"},"behavior":"Removed","invert":true}),
    ));
    for scheme in ["always", "never", "first"] {
        stages.push((
            format!("metaspace-{scheme}"),
            json!({"type":"Metaspace","replacement":"▁","prepend_scheme":scheme,"split":true}),
        ));
    }
    let mut pipelines = Vec::new();
    for (name, pre) in stages {
        let added = if name == "metaspace-first" {
            json!([])
        } else {
            json!([{"id":next,"content":"<tag>","single_word":false,"lstrip":true,"rstrip":true,"normalized":false,"special":true}])
        };
        let post = if name == "sequence" || name == "identity" {
            json!({"type":"TemplateProcessing",
                "single":[{"SpecialToken":{"id":"[S]:$ x","type_id":2}},{"Sequence":{"id":"A","type_id":3}}],
                "pair":[{"Sequence":{"id":"A","type_id":4}},{"SpecialToken":{"id":"[S]:$ x","type_id":2}},{"Sequence":{"id":"B","type_id":5}}],
                "special_tokens":{"[S]:$ x":{"id":"[S]:$ x","ids":[900,901],"tokens":["[X]","[Y]"]}}})
        } else if name.starts_with("bytelevel") {
            json!({"type":"ByteLevel","add_prefix_space":name == "bytelevel-prefix","trim_offsets":true,"use_regex":true})
        } else {
            Value::Null
        };
        let config = json!({"version":"1.0","normalizer":null,"padding":null,"truncation":null,
            "decoder":null,"model":model,"pre_tokenizer":pre,"post_processor":post,"added_tokens":added});
        let tokenizer = Tokenizer::from_bytes(serde_json::to_vec(&config)?)
            .map_err(|e| anyhow!("{name}: {e}"))?;
        let mut cases = Vec::new();
        for (text, second) in [
            ("", None),
            ("ab12猫|é  !?\r\n", None),
            (" a <tag> b🙂  ", None),
            ("ab|中文 e\u{301}", Some(" 12 Яع<tag> b")),
        ] {
            for special in [false, true] {
                let e = match second {
                    Some(b) => tokenizer.encode((text, b), special),
                    None => tokenizer.encode(text, special),
                }
                .map_err(|e| anyhow!("{name}: {e}"))?;
                cases.push(json!({"text":text,"second":second,"add_special_tokens":special,
                    "ids":e.get_ids(),"tokens":e.get_tokens(),"offsets":e.get_offsets(),
                    "type_ids":e.get_type_ids(),"sequence_ids":e.get_sequence_ids(),"special_tokens_mask":e.get_special_tokens_mask()}));
            }
        }
        pipelines.push(json!({"name":name,"pre_tokenizer":pre,"post_processor":post,"added_tokens":added,"cases":cases}));
    }
    let output_data = json!({"tokenizers_version":"0.23.2","rust_toolchain":"1.98.1",
        "source":{"path":source,"sha256":format!("{:x}",Sha256::digest(&bytes))},
        "offsets":"byte","normalization":false,"padding":false,"truncation":false,
        "model":model,"pipelines":pipelines});
    fs::write(output, serde_json::to_string_pretty(&output_data)? + "\n")?;
    Ok(())
}
