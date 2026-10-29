//! Exercise recipe fixtures through the native gateway's typed API adapters.
use std::fs;

use llm_tokenizer::huggingface::HuggingFaceTokenizer;
use openai_protocol::{
    chat::ChatCompletionRequest, messages::CreateMessageRequest, validated::Normalizable,
};
use serde_json::{json, Value};
use smg::routers::grpc::utils::{
    chat_reasoning_starts_in_prefill, process_chat_messages_with_encoding,
    process_messages_with_encoding,
};

#[expect(clippy::unwrap_used, reason = "test tokenizer setup must succeed")]
fn tokenizer() -> (tempfile::TempDir, HuggingFaceTokenizer) {
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
    (directory, tokenizer)
}

#[test]
fn chat_requests_render_like_the_recipe() {
    let (_directory, tokenizer) = tokenizer();
    let fixture: Value = serde_json::from_str(include_str!(
        "../../crates/tokenizer/tests/fixtures/deepseek_v41/recipe.json"
    ))
    .unwrap();
    let mut checked = 0;
    for case in fixture["cases"].as_array().unwrap() {
        // These are core conversation forms, not SMG Chat DTO fields.
        if matches!(
            case["name"].as_str(),
            Some("system_tool_update" | "deferred_tool" | "reminders")
        ) {
            continue;
        }
        let mut body = json!({"model":"deepseek-v4.1-flash", "messages":case["messages"],
            "chat_template_kwargs":{"drop_thinking":case["drop_thinking"].as_bool().unwrap_or(false)},
            "thinking":{"type":if case["thinking"] == false {"disabled"} else {"enabled"}}});
        for (from, to) in [
            ("effort", "reasoning_effort"),
            ("tools", "tools"),
            ("response_format", "response_format"),
            ("tool_choice", "tool_choice"),
        ] {
            if let Some(value) = case.get(from) {
                body[to] = value.clone();
            }
        }
        if case["messages"]
            .as_array()
            .unwrap()
            .last()
            .is_some_and(|message| message["prefix"] == true)
        {
            body["continue_final_message"] = json!(true);
        }
        let mut request: ChatCompletionRequest = serde_json::from_value(body).unwrap();
        request.normalize();
        let (prompt, _) = process_chat_messages_with_encoding(&request, &tokenizer, None).unwrap();
        assert_eq!(
            prompt.text,
            case["text"].as_str().unwrap(),
            "{}",
            case["name"]
        );
        let last = case["messages"].as_array().unwrap().last().unwrap();
        let open_thought = case["thinking"] != false
            && (last["prefix"] != true
                || (last["content"].as_str().unwrap_or("").is_empty()
                    && last["tool_calls"].as_array().is_none_or(Vec::is_empty)));
        assert_eq!(
            chat_reasoning_starts_in_prefill(&request, &tokenizer),
            open_thought,
            "{}",
            case["name"]
        );
        checked += 1;
    }
    assert_eq!(checked, 31);
}

#[test]
fn named_calls_use_the_recipe_required_prefix_and_only_the_selected_tool() {
    let (_directory, tokenizer) = tokenizer();
    let fixture: Value = serde_json::from_str(include_str!(
        "../../crates/tokenizer/tests/fixtures/deepseek_v41/recipe.json"
    ))
    .unwrap();
    let case = fixture["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["name"] == "tools_required")
        .unwrap();
    let mut tools = case["tools"].as_array().unwrap().clone();
    tools.push(json!({"type":"function","function":{"name":"unused","parameters":{}}}));
    let mut request: ChatCompletionRequest = serde_json::from_value(json!({
        "model":"deepseek-v4.1-flash", "messages":case["messages"], "tools":tools,
        "thinking":{"type":"disabled"}, "tool_choice":{"type":"function","function":{"name":"lookup"}}
    })).unwrap();
    request.normalize();
    let (prompt, _) = process_chat_messages_with_encoding(&request, &tokenizer, None).unwrap();
    assert_eq!(prompt.text, case["text"].as_str().unwrap());
}

#[test]
fn messages_preserve_recipe_paragraphs_and_thinking_controls() {
    let (_directory, tokenizer) = tokenizer();
    let request: CreateMessageRequest = serde_json::from_value(json!({
        "model":"deepseek-v4.1-flash", "max_tokens":32,
        "thinking":{"type":"adaptive"}, "output_config":{"effort":"ultra"},
        "system":[{"type":"text","text":"first"},{"type":"text","text":"second"}],
        "messages":[{"role":"user","content":"q"},
            {"role":"assistant","content":[{"type":"thinking","thinking":"old"},
                {"type":"text","text":"a"},{"type":"text","text":"b"}]},
            {"role":"user","content":"next"}]
    }))
    .unwrap();
    let (prompt, _) = process_messages_with_encoding(&request, &tokenizer, None).unwrap();
    assert!(prompt.text.contains("Reasoning Effort: 100 (range"));
    assert!(prompt.text.contains("first\n\nsecond<｜User｜>q"));
    assert!(prompt
        .text
        .contains("<｜Assistant｜><think>old</think>a\n\nb<｜end▁of▁sentence｜>"));
    assert!(prompt
        .text
        .ends_with("<｜User｜>next<｜Assistant｜><think>"));
}

#[test]
fn chat_history_retention_can_be_explicitly_disabled() {
    let (_directory, tokenizer) = tokenizer();
    for drop in [None, Some(false), Some(true)] {
        let mut body = json!({"model":"deepseek-v4.1-flash",
            "messages":[{"role":"user","content":"q"},
                {"role":"assistant","reasoning_content":"old thought","content":"answer"},
                {"role":"user","content":"next"}]});
        if let Some(value) = drop {
            body["chat_template_kwargs"] = json!({"drop_thinking":value});
        }
        let mut request: ChatCompletionRequest = serde_json::from_value(body).unwrap();
        request.normalize();
        let (prompt, _) = process_chat_messages_with_encoding(&request, &tokenizer, None).unwrap();
        assert_eq!(
            prompt.text.contains("<think>old thought</think>"),
            drop != Some(true)
        );
    }
}

#[tokio::test]
#[expect(
    clippy::print_stderr,
    reason = "diagnose optional production tokenizer fixture"
)]
async fn parsed_tool_history_preserves_the_generated_token_prefix() {
    use llm_tokenizer::{Encoder, PromptEncoding};
    use tool_parser::{parsers::DeepSeekDsmlParser, ToolParser};

    let directory = std::env::var_os("DEEPSEEK_V41_MODEL_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../crates/tokenizer/.tokenizer_cache/deepseek_v41")
        });
    let path = directory.join("tokenizer.json");
    if !path.exists() {
        eprintln!("skipping prefix token check: set DEEPSEEK_V41_MODEL_DIR");
        return;
    }
    let tokenizer = HuggingFaceTokenizer::from_file(path.to_str().unwrap()).unwrap();
    let mut body = json!({"model":"deepseek-v4.1-flash", "thinking":{"type":"enabled"},
        "tools":[{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}],
        "messages":[{"role":"user","content":"call f"}]});
    let request: ChatCompletionRequest = serde_json::from_value(body.clone()).unwrap();
    let (initial, encoding) =
        process_chat_messages_with_encoding(&request, &tokenizer, None).unwrap();
    let PromptEncoding::Deferred(job) = encoding else {
        panic!("recipe encoding")
    };
    let mut expected = job.run().unwrap().token_ids().to_vec();
    let calls = concat!(
        "\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"f\">\n",
        "<｜DSML｜ parameter name=\"compact\" string=\"false\">[1,2,3]</｜DSML｜ parameter>\n",
        "<｜DSML｜ parameter name=\"nested\" string=\"false\">{ \"n\":1e+00, \"s\":\"\\u0061\" }</｜DSML｜ parameter>\n",
        "<｜DSML｜ parameter name=\"number\" string=\"false\">-0.00</｜DSML｜ parameter>\n",
        "</｜DSML｜ invoke>\n</｜DSML｜ calls>"
    );
    let generated = format!("Call the tool.</think>{calls}<｜end▁of▁sentence｜>");
    expected.extend_from_slice(tokenizer.encode(&generated, false).unwrap().token_ids());
    let (_, parsed) = DeepSeekDsmlParser::v41()
        .parse_complete(calls)
        .await
        .unwrap();
    body["messages"].as_array_mut().unwrap().extend([
        json!({"role":"assistant","content":"","reasoning_content":"Call the tool.",
            "tool_calls":[{"type":"function","id":"c1","function":parsed[0].function}]}),
        json!({"role":"tool","tool_call_id":"c1","content":"done"}),
    ]);
    let replay: ChatCompletionRequest = serde_json::from_value(body).unwrap();
    let (rendered, encoding) =
        process_chat_messages_with_encoding(&replay, &tokenizer, None).unwrap();
    assert!(rendered
        .text
        .starts_with(&format!("{}{generated}", initial.text)));
    let PromptEncoding::Deferred(job) = encoding else {
        panic!("recipe encoding")
    };
    let replay_ids = job.run().unwrap();
    assert_eq!(&replay_ids.token_ids()[..expected.len()], expected);
}
