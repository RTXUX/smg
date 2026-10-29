//! Public tokenizer rendering against the pinned deepseek-recipe oracle.
use std::{collections::HashMap, fs};

use llm_tokenizer::{
    chat_template::ChatTemplateParams, huggingface::HuggingFaceTokenizer, TokenizerTrait,
};
use serde_json::{json, Value};

#[test]
fn native_renderer_matches_recipe_through_the_public_tokenizer() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("tokenizer.json");
    fs::write(
        &path,
        json!({"version":"1.0", "added_tokens":[],
        "model":{"type":"WordLevel", "vocab":{"[UNK]":0}, "unk_token":"[UNK]"}})
        .to_string(),
    )
    .unwrap();
    fs::write(
        directory.path().join("config.json"),
        r#"{"model_type":"deepseek_v41"}"#,
    )
    .unwrap();
    let tokenizer = HuggingFaceTokenizer::from_file(path.to_str().unwrap()).unwrap();
    check_cases(&tokenizer, false);
}

#[expect(
    clippy::unwrap_used,
    clippy::panic,
    reason = "fixture harness fails on invalid oracle data or divergent rendering"
)]
fn check_cases(tokenizer: &HuggingFaceTokenizer, check_ids: bool) {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/deepseek_v41/recipe.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        // Compare the pinned recipe with its original pruning policy.
        let mut kwargs = HashMap::from([("drop_thinking".to_string(), json!(true))]);
        for (from, to) in [
            ("effort", "reasoning_effort"),
            ("response_format", "response_format"),
            ("tool_choice", "tool_choice"),
        ] {
            if let Some(value) = case.get(from) {
                kwargs.insert(to.to_string(), value.clone());
            }
        }
        let messages = case["messages"].as_array().unwrap();
        let prefix = messages
            .last()
            .is_some_and(|message| message["prefix"] == true);
        let rendered = tokenizer
            .apply_chat_template_with_encoding(
                messages,
                ChatTemplateParams {
                    add_generation_prompt: !prefix,
                    continue_final_message: prefix,
                    thinking: Some(case["thinking"] != false),
                    template_kwargs: Some(&kwargs),
                    tools: case["tools"].as_array().map(Vec::as_slice),
                    ..Default::default()
                },
                None,
            )
            .unwrap();
        assert_eq!(
            rendered.text,
            case["text"].as_str().unwrap(),
            "{}",
            case["name"]
        );
        if check_ids {
            let llm_tokenizer::PromptEncoding::Deferred(job) = rendered.encoding else {
                panic!("recipe encoding must preserve token provenance");
            };
            let ids = job.run().unwrap();
            assert_eq!(
                serde_json::to_value(ids.token_ids()).unwrap(),
                case["ids"],
                "{}",
                case["name"]
            );
        }
    }
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "diagnose the optional production tokenizer fixture"
)]
fn production_token_ids_match_the_recipe() {
    let directory = std::env::var_os("DEEPSEEK_V41_MODEL_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(".tokenizer_cache/deepseek_v41")
        });
    let path = directory.join("tokenizer.json");
    if !path.exists() {
        eprintln!("skipping production-tokenizer parity: set DEEPSEEK_V41_MODEL_DIR");
        return;
    }
    let tokenizer = HuggingFaceTokenizer::from_file(path.to_str().unwrap()).unwrap();
    check_cases(&tokenizer, true);
}
