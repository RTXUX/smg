// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Token-aware DSML grammar lowering, adapted from Dynamo frontend-crates
//! 623a76c2 (deepseek/src/grammar.rs). Resolve before a backend validates the
//! structural tag without a tokenizer. User schemas are opaque data.

use std::sync::OnceLock;

use anyhow::{Context, Result};
use serde_json::{json, Value};
use tokenizers::{AddedVocabulary, Tokenizer};

#[derive(Default)]
pub(crate) struct GrammarTokens(OnceLock<Result<Vocabulary, String>>);

impl GrammarTokens {
    /// Scan once at tokenizer setup, outside the request path. A model without
    /// DSML pays no vocabulary scan; unsupported vocabularies fail on use.
    pub(crate) fn prepare(tokenizer: &Tokenizer) -> Self {
        let tokens = Self::default();
        if control_token_id(tokenizer, "｜DSML｜").is_ok() {
            let _ = tokens.vocabulary(tokenizer);
        }
        tokens
    }

    fn vocabulary(&self, tokenizer: &Tokenizer) -> Result<&Vocabulary> {
        self.0
            .get_or_init(|| Vocabulary::new(tokenizer).map_err(|error| error.to_string()))
            .as_ref()
            .map_err(|error| anyhow::anyhow!("{error}"))
    }

    pub(crate) fn resolve(&self, tokenizer: &Tokenizer, format: &mut Value) -> Result<()> {
        walk(format, &mut |node| {
            match node["type"].as_str() {
                Some("token_span") => {
                    anyhow::ensure!(
                        node["prefix"] == STRING_PREFIX && node["suffix"] == "</",
                        "unsupported token span"
                    );
                    *node = self.vocabulary(tokenizer)?.string_value.clone();
                }
                Some("token_json") => *node = self.vocabulary(tokenizer)?.json_value.clone(),
                Some("token_suffix") => {
                    *node = self
                        .vocabulary(tokenizer)?
                        .format(node["suffix"].as_str().context("missing token suffix")?)?;
                }
                Some("token") => resolve(tokenizer, &mut node["token"])?,
                Some("any_tokens" | "exclude_token") => {
                    for token in node["exclude_tokens"]
                        .as_array_mut()
                        .context("missing token exclusions")?
                    {
                        resolve(tokenizer, token)?;
                    }
                }
                _ => {}
            }
            Ok(())
        })
    }
}

/// Other tokenizer implementations can pass already resolved grammars, but
/// must never silently send symbolic terminals to a backend.
pub(crate) fn require_resolved(format: &mut Value) -> Result<()> {
    walk(format, &mut |node| {
        let unresolved = match node["type"].as_str() {
            Some("token_span" | "token_suffix" | "token_json") => true,
            Some("token") => !node["token"].is_u64(),
            Some("any_tokens" | "exclude_token") => node["exclude_tokens"]
                .as_array()
                .is_none_or(|tokens| tokens.iter().any(|token| !token.is_u64())),
            _ => false,
        };
        anyhow::ensure!(
            !unresolved,
            "tokenizer does not support symbolic grammar tokens"
        );
        Ok(())
    })
}

fn walk(format: &mut Value, leaf: &mut impl FnMut(&mut Value) -> Result<()>) -> Result<()> {
    match format["type"].as_str() {
        Some(
            "token_span" | "token_suffix" | "token_json" | "token" | "any_tokens" | "exclude_token",
        ) => leaf(format),
        Some("sequence" | "or") => {
            for child in format["elements"]
                .as_array_mut()
                .context("missing grammar elements")?
            {
                walk(child, leaf)?;
            }
            Ok(())
        }
        Some("star" | "plus" | "optional") => walk(&mut format["content"], leaf),
        Some("tag") => {
            for key in ["begin", "content", "end"] {
                if format[key].is_object() {
                    walk(&mut format[key], leaf)?;
                }
            }
            Ok(())
        }
        Some("triggered_tags" | "token_triggered_tags" | "tags_with_separator") => {
            for tag in format["tags"]
                .as_array_mut()
                .context("missing grammar tags")?
            {
                walk(tag, leaf)?;
            }
            Ok(())
        }
        Some("structural_tag") => walk(&mut format["format"], leaf),
        // Never descend into user schemas, regexes, or literal text.
        _ => Ok(()),
    }
}

/// A plain prefix before DSML can share its token with preceding content, e.g.
/// `\n\n<` or `value</`. Match the suffix at that token boundary without requiring
/// those ordinary tokens to be split. Never classify added tokens as plain.
struct Vocabulary {
    open: Vec<u32>,
    close: Vec<u32>,
    slash: Vec<u32>,
    string_value: Value,
    json_value: Value,
}

const STRING_PREFIX: &str = "\" string=\"true\">";

impl Vocabulary {
    fn new(tokenizer: &Tokenizer) -> Result<Self> {
        let mut tokens = Self {
            open: Vec::new(),
            close: Vec::new(),
            slash: Vec::new(),
            string_value: Value::Null,
            json_value: Value::Null,
        };
        let mut starts = vec![Vec::new(); STRING_PREFIX.len()];
        let mut complete_close = starts.clone();
        let mut complete_open = starts.clone();
        let mut json_pipe_tokens = Vec::new();
        for id in tokenizer.get_vocab(true).into_values() {
            if tokenizer
                .get_added_vocabulary()
                .get_added_tokens_decoder()
                .contains_key(&id)
            {
                continue;
            }
            let text = tokenizer.decode(&[id], false).map_err(anyhow::Error::msg)?;
            if text.ends_with('<') {
                tokens.open.push(id);
            }
            if text.ends_with("</") {
                tokens.close.push(id);
            }
            if text == "/" {
                tokens.slash.push(id);
            }
            if text.contains('｜')
                && !text
                    .chars()
                    .any(|ch| ch == '"' || ch == '\\' || ch < '\u{20}')
            {
                json_pipe_tokens.push(id);
            }
            // A token may contain both the end of the fixed header and some
            // string data. Keep all possible header/token boundary positions.
            for offset in 0..STRING_PREFIX.len() {
                if let Some(rest) = text.strip_prefix(&STRING_PREFIX[offset..]) {
                    starts[offset].push(id);
                    if rest.ends_with("</") {
                        complete_close[offset].push(id);
                    }
                    if rest.ends_with('<') {
                        complete_open[offset].push(id);
                    }
                }
            }
        }
        for ids in [&mut tokens.open, &mut tokens.close, &mut tokens.slash] {
            ids.sort_unstable();
            ids.dedup();
        }
        anyhow::ensure!(
            !tokens.open.is_empty(),
            "tokenizer has no plain DSML opening prefix"
        );
        let dsml = control_token_id(tokenizer, "｜DSML｜")?;
        let quoted = serde_json::to_string(STRING_PREFIX)?;
        let mut begin = vec![quoted];
        let mut complete = Vec::new();
        for offset in 0..STRING_PREFIX.len() {
            let prefix = serde_json::to_string(&STRING_PREFIX[..offset])?;
            if let Some(ids) = token_set(&mut starts[offset]) {
                begin.push(format!("{prefix} {ids}"));
            }
            if let Some(ids) = token_set(&mut complete_close[offset]) {
                complete.push(format!("{prefix} {ids}"));
            }
            if let (Some(ids), Some(slash)) = (
                token_set(&mut complete_open[offset]),
                token_set(&mut tokens.slash),
            ) {
                complete.push(format!("{prefix} {ids} {slash}"));
            }
        }
        let mut end = Vec::new();
        if let Some(ids) = token_set(&mut tokens.close) {
            end.push(ids);
        }
        if let Some(slash) = token_set(&mut tokens.slash) {
            let open = token_set(&mut tokens.open).context("missing plain opening prefix")?;
            end.push(format!("{open} {slash}"));
        }
        anyhow::ensure!(
            !end.is_empty(),
            "tokenizer has no plain DSML closing prefix"
        );
        let mut alternatives = vec![format!(
            "({}) ExcludeToken({dsml})* ({})",
            begin.join(" | "),
            end.join(" | ")
        )];
        alternatives.extend(complete);
        tokens.string_value = json!({"type":"grammar", "grammar": format!("root ::= {}\n", alternatives.join(" | "))});
        // XGrammar's character-based JSON string matcher otherwise accepts
        // DSML's added ID as ordinary Unicode. Keep JSON syntax, but make the
        // marker's first character require normal token IDs inside strings.
        // Include the plain encoder's byte-fallback sequence as well as whole
        // normal tokens containing the character. No added ID enters this rule.
        let mut plain = tokenizer.clone();
        plain.with_added_vocabulary(AddedVocabulary::new());
        let pipe_ids = plain
            .encode("｜", false)
            .map_err(anyhow::Error::msg)?
            .get_ids()
            .to_vec();
        anyhow::ensure!(
            !pipe_ids.is_empty()
                && pipe_ids.iter().all(|id| !tokenizer
                    .get_added_vocabulary()
                    .get_added_tokens_decoder()
                    .contains_key(id)),
            "missing ordinary token encoding for the DSML marker character"
        );
        anyhow::ensure!(
            tokenizer
                .decode(&pipe_ids, false)
                .map_err(anyhow::Error::msg)?
                == "｜",
            "DSML marker character does not round-trip through ordinary tokens"
        );
        let mut pipe = vec![pipe_ids
            .iter()
            .map(|id| format!("Token({id})"))
            .collect::<Vec<_>>()
            .join(" ")];
        if let Some(ids) = token_set(&mut json_pipe_tokens) {
            pipe.push(ids);
        }
        let json_grammar = format!(
            r#"root ::= ws value ws
value ::= object | array | string | number | "true" | "false" | "null"
object ::= "{{" ws (string ws ":" ws value (ws "," ws string ws ":" ws value)*)? ws "}}"
array ::= "[" ws (value (ws "," ws value)*)? ws "]"
string ::= "\"" char* "\""
char ::= [^"\\\x00-\x1f｜] | "\\" escape | pipe
escape ::= ["\\/bfnrt] | "u" [0-9a-fA-F]{{4}}
pipe ::= {}
number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
ws ::= [ \t\r\n]*
"#,
            pipe.join(" | ")
        );
        tokens.json_value = json!({"type":"grammar", "grammar":json_grammar});
        Ok(tokens)
    }

    fn format(&self, suffix: &str) -> Result<Value> {
        let choices = |ids: &[u32]| json!({"type":"or", "elements":ids.iter().map(|id| json!({"type":"token","token":id})).collect::<Vec<_>>()});
        match suffix {
            "<" => Ok(choices(&self.open)),
            "</" => {
                let mut elements = Vec::new();
                if !self.close.is_empty() {
                    elements.push(choices(&self.close));
                }
                if !self.slash.is_empty() {
                    elements.push(json!({"type":"sequence", "elements":[choices(&self.open), choices(&self.slash)]}));
                }
                anyhow::ensure!(
                    !elements.is_empty(),
                    "tokenizer has no plain DSML closing prefix"
                );
                Ok(json!({"type":"or", "elements":elements}))
            }
            _ => anyhow::bail!("unsupported grammar token suffix {suffix:?}"),
        }
    }
}

fn token_set(ids: &mut Vec<u32>) -> Option<String> {
    ids.sort_unstable();
    ids.dedup();
    (!ids.is_empty()).then(|| {
        format!(
            "Token({})",
            ids.iter().map(u32::to_string).collect::<Vec<_>>().join(",")
        )
    })
}

fn resolve(tokenizer: &Tokenizer, value: &mut Value) -> Result<()> {
    if value.is_u64() {
        return Ok(());
    }
    let spelling = value
        .as_str()
        .context("expected a grammar token name or ID")?;
    *value = Value::from(control_token_id(tokenizer, spelling)?);
    Ok(())
}

fn control_token_id(tokenizer: &Tokenizer, spelling: &str) -> Result<u32> {
    let id = tokenizer.token_to_id(spelling).with_context(|| {
        format!("grammar control token {spelling:?} is missing from the tokenizer")
    })?;
    // Recipe identifies every added-vocabulary entry by ID, including DSML's
    // special=false entry. Do not infer control identity from decoded text.
    anyhow::ensure!(
        tokenizer
            .get_added_vocabulary()
            .get_added_tokens_decoder()
            .contains_key(&id),
        "grammar control token {spelling:?} is not in the added vocabulary"
    );
    Ok(id)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{
        cache::{CacheConfig, CachedTokenizer},
        HuggingFaceTokenizer, TokenizerTrait,
    };

    #[test]
    fn resolve_only_terminals_accept_special_false_and_forward_through_cache() {
        let raw = Tokenizer::from_bytes(
            serde_json::to_vec(&json!({
                "version":"1.0", "added_tokens":[{
                    "id":1,"content":"｜DSML｜","single_word":false,"lstrip":false,
                    "rstrip":false,"normalized":false,"special":false
                }],
                "model":{"type":"WordLevel","vocab":{"[UNK]":0,"｜DSML｜":1},"unk_token":"[UNK]"}
            }))
            .unwrap(),
        )
        .unwrap();
        let tokenizer = CachedTokenizer::new(
            Arc::new(HuggingFaceTokenizer::from_tokenizer(raw)),
            CacheConfig::default(),
        );
        let schema = json!({"type":"json_schema","json_schema":{"const":{"type":"token","token":"user data"}}});
        let mut format = json!({"type":"sequence","elements":[
            {"type":"token","token":"｜DSML｜"}, schema,
            {"type":"const_string","value":"｜DSML｜"}
        ]});
        tokenizer.resolve_grammar_tokens(&mut format).unwrap();
        assert_eq!(format["elements"][0]["token"], 1);
        assert_eq!(format["elements"][1], schema);
        assert_eq!(format["elements"][2]["value"], "｜DSML｜");
        let resolved = format.clone();
        tokenizer.resolve_grammar_tokens(&mut format).unwrap();
        assert_eq!(format, resolved);
        for text in ["missing", "[UNK]"] {
            assert!(tokenizer
                .resolve_grammar_tokens(&mut json!({"type":"token","token":text}))
                .is_err());
        }
    }

    #[test]
    fn unsupported_tokenizers_fail_only_for_unresolved_terminals() {
        let tokenizer = crate::MockTokenizer::new();
        let mut format =
            json!({"type":"json_schema","json_schema":{"const":{"type":"token_json"}}});
        tokenizer.resolve_grammar_tokens(&mut format).unwrap();
        for mut format in [
            json!({"type":"token_span"}),
            json!({"type":"token_json"}),
            json!({"type":"token","token":"｜DSML｜"}),
        ] {
            assert!(tokenizer.resolve_grammar_tokens(&mut format).is_err());
        }
    }
}
