//! Compact regexp-boundary fixtures; run explicitly, never from go test.
use anyhow::{anyhow, Result};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{env, fs};
use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
use tokenizers::{
    OffsetReferential, OffsetType, PreTokenizedString, PreTokenizer, SplitDelimiterBehavior,
};

fn main() -> Result<()> {
    let output = env::args()
        .nth(1)
        .ok_or_else(|| anyhow!("supply output fixture path; run from repository root"))?;
    let mut patterns = vec![
        ("ascii-digit".to_owned(), "[0-9]".to_owned()),
        ("ascii-triplet".to_owned(), "[0-9][0-9][0-9]".to_owned()),
    ];
    let mut sources = Vec::new();
    for name in ["qwen2.5", "qwen3.8", "bloom"] {
        let path = format!("testdata/{name}/tokenizer.json");
        let bytes = fs::read(&path)?;
        let tokenizer: Value = serde_json::from_slice(&bytes)?;
        let pattern = tokenizer["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
            .as_str()
            .ok_or_else(|| anyhow!("missing regexp: {path}"))?;
        patterns.push((name.to_owned(), pattern.to_owned()));
        sources.push(json!({"path":path,"sha256":format!("{:x}",Sha256::digest(&bytes))}));
    }
    let texts = [
        "",
        "abc",
        "1234567",
        "α12猫3z",
        "  hello world  ",
        "I'm WE'RE he'll",
        "e\u{301} é",
        "Привет, мир!",
        "العَرَبِيَّة ١٢٣",
        "日本語。中文！",
        "x\r\n  y\n\t",
        "a\u{a0}\u{2003}b ",
        "👩‍💻1🙂",
        "\0a1\0",
        "a\u{301}b",
        "   \r\n",
    ];
    let mut cases = Vec::new();
    for (name, pattern) in patterns {
        let split = Split::new(
            SplitPattern::Regex(pattern.clone()),
            SplitDelimiterBehavior::Isolated,
            false,
        )
        .map_err(|e| anyhow!("{e}"))?;
        for text in texts {
            let mut input = PreTokenizedString::from(text);
            split.pre_tokenize(&mut input).map_err(|e| anyhow!("{e}"))?;
            let pieces: Vec<_> = input
                .get_splits(OffsetReferential::Original, OffsetType::Byte)
                .into_iter()
                .map(|(value, (start, end), _)| json!({"value":value,"offsets":[start,end]}))
                .collect();
            cases.push(json!({"name":name,"pattern":pattern,"text":text,"pieces":pieces}));
        }
    }
    let result = json!({"tokenizers_version":"0.23.2", "rust_toolchain":"1.98.1",
        "behavior":"Isolated", "invert":false, "offsets":"byte", "normalization":false,
        "sources":sources, "cases":cases});
    fs::write(output, serde_json::to_string_pretty(&result)? + "\n")?;
    Ok(())
}
