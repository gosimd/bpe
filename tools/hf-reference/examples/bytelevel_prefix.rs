//! Regression oracle for unmerged synthetic ByteLevel prefix-space offsets.
use anyhow::{anyhow, Result};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{env, fs};
use tokenizers::Tokenizer;

fn main() -> Result<()> {
    let output = env::args()
        .nth(1)
        .ok_or_else(|| anyhow!("supply output path"))?;
    let source = "testdata/gpt2/tokenizer.json";
    let bytes = fs::read(source)?;
    let mut config: Value = serde_json::from_slice(&bytes)?;
    config["model"]["vocab"]
        .as_object_mut()
        .unwrap()
        .retain(|_, id| id.as_u64().unwrap() < 256);
    config["model"]["merges"] = json!([]);
    config["normalizer"] = Value::Null;
    config["padding"] = Value::Null;
    config["truncation"] = Value::Null;
    config["added_tokens"] = json!([{"id":256,"content":"<tag>","single_word":false,
        "lstrip":false,"rstrip":false,"normalized":false,"special":true}]);
    let mut cases = Vec::new();
    for prefix in [false, true] {
        config["pre_tokenizer"] = json!({"type":"ByteLevel","add_prefix_space":prefix,
            "trim_offsets":false,"use_regex":true});
        for trim in [false, true] {
            config["post_processor"] = if trim {
                json!({"type":"ByteLevel","add_prefix_space":prefix,"trim_offsets":true,"use_regex":true})
            } else {
                Value::Null
            };
            let tokenizer =
                Tokenizer::from_bytes(serde_json::to_vec(&config)?).map_err(|e| anyhow!("{e}"))?;
            for text in ["é", "🙂", "<tag>é", "a<tag>🙂", "e\u{301}", "\0a", ""] {
                let encoding = tokenizer.encode(text, false).map_err(|e| anyhow!("{e}"))?;
                cases.push(json!({"text":text,"add_prefix_space":prefix,"trim_offsets":trim,
                    "ids":encoding.get_ids(),"tokens":encoding.get_tokens(),"offsets":encoding.get_offsets()}));
            }
        }
    }
    let result = json!({"tokenizers_version":"0.23.2","rust_toolchain":"1.98.1",
        "source":{"path":source,"sha256":format!("{:x}",Sha256::digest(&bytes))},
        "model":"GPT-2 byte alphabet (IDs < 256), no merges; AddedToken <tag> = 256",
        "offsets":"byte","normalization":false,"padding":false,"truncation":false,
        "cases":cases});
    fs::write(output, serde_json::to_string_pretty(&result)? + "\n")?;
    Ok(())
}
