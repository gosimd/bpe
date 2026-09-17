//! Explicit compact integration oracle: Split(Isolated) -> ByteLevel(false).
//! No normalizer, padding, truncation, network access or full-Qwen claim.
use anyhow::{anyhow, Result};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{env, fs};
use tokenizers::{OffsetReferential, OffsetType, PreTokenizedString, PreTokenizer, Tokenizer};

fn main() -> Result<()> {
    let output = env::args()
        .nth(1)
        .ok_or_else(|| anyhow!("supply output path"))?;
    let source = "testdata/gpt2/tokenizer.json";
    let bytes = fs::read(source)?;
    let mut config: Value = serde_json::from_slice(&bytes)?;
    config["normalizer"] = Value::Null;
    config["padding"] = Value::Null;
    config["truncation"] = Value::Null;
    config["added_tokens"] = json!([{"id":50257,"content":"<tag>","single_word":false,
        "lstrip":true,"rstrip":true,"normalized":false,"special":true}]);
    let qwen: Value = serde_json::from_slice(&fs::read("testdata/qwen2.5/tokenizer.json")?)?;
    let qwen_pattern = qwen["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
        .as_str()
        .ok_or_else(|| anyhow!("missing Qwen pattern"))?;
    let patterns = ["[0-9]", "[a-z]+", "^ab|[0-9]+$", "", "a*", qwen_pattern];
    let texts = [
        "",
        "ab12猫3z",
        "hello, world! 123",
        "e\u{301} é\r\n\t🙂",
        "Я عربي 日本中文",
    ];
    let mut cases = Vec::new();
    for pattern in patterns {
        for prefix in [false, true] {
            config["pre_tokenizer"] = json!({"type":"Sequence","pretokenizers":[
                {"type":"Split","pattern":{"Regex":pattern},"behavior":"Isolated","invert":false},
                {"type":"ByteLevel","add_prefix_space":prefix,"trim_offsets":false,"use_regex":false}
            ]});
            for processor in ["none", "bytelevel", "template"] {
                config["post_processor"] = match processor {
                    "bytelevel" => {
                        json!({"type":"ByteLevel","add_prefix_space":prefix,"trim_offsets":true,"use_regex":false})
                    }
                    "template" => json!({"type":"TemplateProcessing",
                        "single":[{"SpecialToken":{"id":"[S]","type_id":0}},{"Sequence":{"id":"A","type_id":0}}],
                        "pair":[{"SpecialToken":{"id":"[S]","type_id":0}},{"Sequence":{"id":"A","type_id":0}},
                            {"Sequence":{"id":"B","type_id":1}},{"SpecialToken":{"id":"[S]","type_id":1}}],
                        "special_tokens":{"[S]":{"id":"[S]","ids":[50258],"tokens":["[S]"]}}}),
                    _ => Value::Null,
                };
                let tokenizer = Tokenizer::from_bytes(serde_json::to_vec(&config)?)
                    .map_err(|e| anyhow!("{e}"))?;
                if processor == "none" {
                    for text in texts {
                        cases.push(snapshot(
                            &tokenizer, pattern, prefix, processor, text, None, false,
                        )?);
                    }
                } else if pattern == "[0-9]" || pattern == "[a-z]+" {
                    for pair in [false, true] {
                        for special in [false, true] {
                            cases.push(snapshot(
                                &tokenizer,
                                pattern,
                                prefix,
                                processor,
                                "a <tag> b12猫!  ",
                                pair.then_some("  34é <tag> z"),
                                special,
                            )?);
                        }
                    }
                }
            }
        }
    }
    let result = json!({"tokenizers_version":"0.23.2","rust_toolchain":"1.98.1",
        "source":{"path":source,"sha256":format!("{:x}",Sha256::digest(&bytes))},
        "offsets":"byte","normalization":false,"padding":false,"truncation":false,
        "pipeline":"AddedTokens -> Split(Isolated, invert=false) -> ByteLevel(use_regex=false) -> BPE -> postprocessor",
        "cases":cases});
    fs::write(output, serde_json::to_string_pretty(&result)? + "\n")?;
    Ok(())
}

fn snapshot(
    tokenizer: &Tokenizer,
    pattern: &str,
    prefix: bool,
    processor: &str,
    text: &str,
    second: Option<&str>,
    special: bool,
) -> Result<Value> {
    let encoding = match second {
        Some(other) => tokenizer.encode((text, other), special),
        None => tokenizer.encode(text, special),
    }
    .map_err(|e| anyhow!("{e}"))?;
    // Independently expose the raw pre-tokenizer stage (without AddedTokens).
    let mut input = PreTokenizedString::from(text);
    tokenizer
        .get_pre_tokenizer()
        .unwrap()
        .pre_tokenize(&mut input)
        .map_err(|e| anyhow!("{e}"))?;
    let pieces: Vec<_> = input
        .get_splits(OffsetReferential::Original, OffsetType::Byte)
        .into_iter()
        .map(|(value, offsets, _)| json!({"value":value,"offsets":offsets}))
        .collect();
    Ok(
        json!({"pattern":pattern,"add_prefix_space":prefix,"processor":processor,
        "text":text,"second":second,"add_special_tokens":special,"pieces":pieces,
        "encoding":{"ids":encoding.get_ids(),"tokens":encoding.get_tokens(),
            "offsets":encoding.get_offsets(),"type_ids":encoding.get_type_ids(),
            "sequence_ids":encoding.get_sequence_ids(),"special_tokens_mask":encoding.get_special_tokens_mask()}}),
    )
}
