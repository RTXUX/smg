//! Verify that `HuggingFaceTokenizer` selects the right chat-template renderer
//! based on `config.json::architectures`.
#[cfg(test)]
mod tests {
    use std::{collections::HashMap, fs};

    use llm_tokenizer::{
        chat_template::{
            ChatTemplateContentFormat, ChatTemplateParams, ThinkingKeyName, ThinkingToggle,
        },
        huggingface::HuggingFaceTokenizer,
        PromptEncoding, TokenizerTrait,
    };
    use openai_protocol::common::{Function, Tool};
    use serde_json::json;
    use tempfile::TempDir;
    use tokenizers::AddedVocabulary;
    /// A minimal tokenizer.json that loads cleanly. The only requirement is that
    /// it parses; the encoder logic does not call back into the tokenizer here.
    const MIN_TOKENIZER_JSON: &str = r#"{
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": { "type": "Whitespace" },
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "BPE",
            "vocab": { "hello": 0, "<s>": 1, "</s>": 2 },
            "merges": []
        }
    }"#;
    fn write_dir(architectures: Option<&[&str]>) -> (TempDir, String) {
        write_model_dir(None, architectures, None, None)
    }

    /// Build a model dir (optionally named, under the temp root) with the
    /// minimal tokenizer, an optional config (`architectures` and/or
    /// `model_type`), and an optional shipped Python encoder.
    fn write_model_dir(
        model_name: Option<&str>,
        architectures: Option<&[&str]>,
        model_type: Option<&str>,
        encoder_source: Option<&str>,
    ) -> (TempDir, String) {
        let temp = TempDir::new().unwrap();
        let model_dir = match model_name {
            Some(name) => {
                let dir = temp.path().join(name);
                fs::create_dir(&dir).unwrap();
                dir
            }
            None => temp.path().to_path_buf(),
        };
        let tok_path = model_dir.join("tokenizer.json");
        fs::write(&tok_path, MIN_TOKENIZER_JSON).unwrap();
        if architectures.is_some() || model_type.is_some() {
            let mut config = serde_json::Map::new();
            if let Some(archs) = architectures {
                config.insert("architectures".to_string(), json!(archs));
            }
            if let Some(model_type) = model_type {
                config.insert("model_type".to_string(), json!(model_type));
            }
            let body = serde_json::Value::Object(config).to_string();
            fs::write(model_dir.join("config.json"), body).unwrap();
        }
        if let Some(source) = encoder_source {
            let encoding_dir = model_dir.join("encoding");
            fs::create_dir(&encoding_dir).unwrap();
            fs::write(encoding_dir.join("encoding_dsv4.py"), source).unwrap();
        }
        (temp, tok_path.to_str().unwrap().to_string())
    }

    #[test]
    fn config_with_deepseek_v32_arch_uses_v32_renderer() {
        let (_tmp, tok) = write_dir(Some(&["DeepseekV32ForCausalLM"]));
        let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();
        let messages = vec![json!({ "role": "user", "content": "Hello" })];
        let kwargs: HashMap<String, serde_json::Value> = HashMap::new();
        let params = ChatTemplateParams {
            template_kwargs: Some(&kwargs),
            ..Default::default()
        };
        let out = tokenizer.apply_chat_template(&messages, params).unwrap();
        // V3.2 emits BOS + <｜User｜>Hello<｜Assistant｜></think> in chat mode.
        assert!(out.contains("<\u{FF5C}begin\u{2581}of\u{2581}sentence\u{FF5C}>"));
        assert!(out.contains("<\u{FF5C}User\u{FF5C}>Hello<\u{FF5C}Assistant\u{FF5C}>"));
        assert!(out.ends_with("</think>"));
    }
    #[test]
    fn config_with_deepseek_v4_arch_uses_v4_renderer() {
        let (_tmp, tok) = write_dir(Some(&["DeepseekV4ForCausalLM"]));
        let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();
        let messages = vec![json!({ "role": "user", "content": "Hello" })];
        let kwargs: HashMap<String, serde_json::Value> = HashMap::new();
        let params = ChatTemplateParams {
            template_kwargs: Some(&kwargs),
            ..Default::default()
        };
        let out = tokenizer.apply_chat_template(&messages, params).unwrap();
        // V4 renders the engine's default: BOS, the `high` effort preamble,
        // <｜User｜>Hello<｜Assistant｜><think> (thinking mode).
        assert!(
            out.starts_with("<\u{FF5C}begin\u{2581}of\u{2581}sentence\u{FF5C}>Reasoning Effort:")
        );
        assert!(out.contains("<\u{FF5C}User\u{FF5C}>Hello<\u{FF5C}Assistant\u{FF5C}>"));
        assert!(out.ends_with("<think>"));
        // `thinking: false` is chat mode: no preamble, `</think>` closed.
        let kwargs = HashMap::from([("thinking".to_string(), json!(false))]);
        let out = tokenizer
            .apply_chat_template(
                &messages,
                ChatTemplateParams {
                    template_kwargs: Some(&kwargs),
                    ..Default::default()
                },
            )
            .unwrap();
        assert!(out.starts_with(
            "<\u{FF5C}begin\u{2581}of\u{2581}sentence\u{FF5C}><\u{FF5C}User\u{FF5C}>Hello"
        ));
        assert!(out.ends_with("</think>"));
    }

    #[test]
    fn config_with_unrelated_arch_falls_back_to_jinja() {
        // A non-DeepSeek architecture should keep using the Jinja renderer; with
        // no chat_template set, applying the template should error rather than
        // silently picking a DeepSeek encoder.
        let (_tmp, tok) = write_dir(Some(&["LlamaForCausalLM"]));
        let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();
        let messages = vec![json!({ "role": "user", "content": "Hello" })];
        let result = tokenizer.apply_chat_template(&messages, ChatTemplateParams::default());
        assert!(
            result.is_err(),
            "expected error from missing Jinja template"
        );
    }
    #[test]
    fn no_config_json_falls_back_to_jinja() {
        // No sibling config.json — must still default to Jinja and not blow up.
        let (_tmp, tok) = write_dir(None);
        let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();
        let messages = vec![json!({ "role": "user", "content": "Hello" })];
        let result = tokenizer.apply_chat_template(&messages, ChatTemplateParams::default());
        // Without a chat template registered, the Jinja renderer surfaces an error.
        // The important thing is that we did NOT auto-select a DeepSeek encoder.
        assert!(result.is_err());
    }
    #[test]
    fn malformed_config_json_falls_back_to_jinja() {
        let temp = TempDir::new().unwrap();
        let tok_path = temp.path().join("tokenizer.json");
        fs::write(&tok_path, MIN_TOKENIZER_JSON).unwrap();
        fs::write(temp.path().join("config.json"), "{ this is not json").unwrap();
        let tokenizer = HuggingFaceTokenizer::from_file(tok_path.to_str().unwrap()).unwrap();
        let messages = vec![json!({ "role": "user", "content": "Hello" })];
        let result = tokenizer.apply_chat_template(&messages, ChatTemplateParams::default());
        assert!(result.is_err());
    }
    #[test]
    fn deepseek_v4_injects_tools_into_system_message() {
        // Client passes tools at the request level (params.tools), not embedded
        // in messages. The shim must attach them to a system message so the
        // encoder renders the tools block.
        let (_tmp, tok) = write_dir(Some(&["DeepseekV4ForCausalLM"]));
        let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();
        let messages = vec![json!({ "role": "user", "content": "Hello" })];
        let tools = vec![json!({
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": { "city": { "type": "string" } },
                    "required": ["city"]
                }
            }
        })];
        let params = ChatTemplateParams {
            tools: Some(&tools),
            ..Default::default()
        };
        let out = tokenizer.apply_chat_template(&messages, params).unwrap();
        assert!(out.contains("## Tools"), "tools block missing: {out}");
        assert!(
            out.contains("get_weather"),
            "tool name missing from prompt: {out}"
        );
        assert!(
            out.contains("<\u{FF5C}DSML\u{FF5C}tool_calls>"),
            "V4 DSML invocation grammar missing: {out}"
        );
    }

    #[test]
    fn deepseek_v32_injects_tools_into_system_message() {
        let (_tmp, tok) = write_dir(Some(&["DeepseekV32ForCausalLM"]));
        let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();
        let messages = vec![json!({ "role": "user", "content": "Hi" })];
        let tools = vec![json!({
            "type": "function",
            "function": { "name": "ping", "description": "ping", "parameters": {} }
        })];
        let params = ChatTemplateParams {
            tools: Some(&tools),
            ..Default::default()
        };
        let out = tokenizer.apply_chat_template(&messages, params).unwrap();
        assert!(out.contains("## Tools"), "tools block missing: {out}");
        assert!(out.contains("ping"), "tool name missing: {out}");
        // V3.2 uses function_calls (vs V4's tool_calls).
        assert!(
            out.contains("<\u{FF5C}DSML\u{FF5C}function_calls>"),
            "V3.2 DSML invocation grammar missing: {out}"
        );
    }

    #[test]
    fn deepseek_v4_attaches_tools_to_existing_system_message() {
        // When a system message is already present, tools should attach to it
        // rather than inserting a second system block.
        let (_tmp, tok) = write_dir(Some(&["DeepseekV4ForCausalLM"]));
        let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();
        let messages = vec![
            json!({ "role": "system", "content": "Be concise." }),
            json!({ "role": "user", "content": "Hi" }),
        ];
        let tools = vec![json!({
            "type": "function",
            "function": { "name": "ping", "description": "ping", "parameters": {} }
        })];
        let params = ChatTemplateParams {
            tools: Some(&tools),
            ..Default::default()
        };
        let out = tokenizer.apply_chat_template(&messages, params).unwrap();
        assert!(
            out.contains("Be concise."),
            "existing system content lost: {out}"
        );
        assert!(out.contains("ping"), "tool not attached: {out}");
    }

    #[test]
    fn deepseek_renderers_report_thinking_introspection() {
        // V3.2 / V4 inject `<think>` in the prefill when thinking is on, and
        // gate thinking on the `thinking` kwarg. The trait methods must
        // surface this so the gateway can call `mark_reasoning_started` on
        // the conditional reasoning parser (deepseek_v31 etc). V3.2 defaults
        // thinking off; V4 defaults it on like the engine's own server.
        for (arch, toggle) in &[
            ("DeepseekV32ForCausalLM", ThinkingToggle::DefaultOff),
            ("DeepseekV4ForCausalLM", ThinkingToggle::DefaultOn),
        ] {
            let (_tmp, tok) = write_dir(Some(&[*arch]));
            let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();
            assert_eq!(
                tokenizer.thinking_toggle(),
                *toggle,
                "{arch}: unexpected thinking toggle"
            );
            assert_eq!(
                tokenizer.thinking_key_name(),
                Some(ThinkingKeyName::Thinking),
                "{arch}: expected Thinking key name"
            );
            assert!(
                tokenizer.think_in_prefill(),
                "{arch}: expected think_in_prefill=true"
            );
        }
    }

    #[test]
    fn deepseek_v32_renderer_honors_thinking_kwarg_only() {
        // `thinking: true` → prompt ends with <think> (thinking mode).
        // `enable_thinking: true` alone → ignored (chat mode), matching
        // `thinking_key_name() == Some(Thinking)` and the serving engine's DeepSeek path.
        let (_tmp, tok) = write_dir(Some(&["DeepseekV32ForCausalLM"]));
        let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();
        let messages = vec![json!({ "role": "user", "content": "Hi" })];

        let mut thinking_kwargs: HashMap<String, serde_json::Value> = HashMap::new();
        thinking_kwargs.insert("thinking".to_string(), serde_json::Value::Bool(true));
        let out_thinking = tokenizer
            .apply_chat_template(
                &messages,
                ChatTemplateParams {
                    template_kwargs: Some(&thinking_kwargs),
                    ..Default::default()
                },
            )
            .unwrap();
        assert!(
            out_thinking.ends_with("<think>"),
            "thinking=true should enter thinking mode: {out_thinking}"
        );

        let mut enable_thinking_kwargs: HashMap<String, serde_json::Value> = HashMap::new();
        enable_thinking_kwargs.insert("enable_thinking".to_string(), serde_json::Value::Bool(true));
        let out_enable = tokenizer
            .apply_chat_template(
                &messages,
                ChatTemplateParams {
                    template_kwargs: Some(&enable_thinking_kwargs),
                    ..Default::default()
                },
            )
            .unwrap();
        assert!(
            out_enable.ends_with("</think>"),
            "enable_thinking alone must NOT enter thinking mode: {out_enable}"
        );
    }

    /// Render a V4 request the way the gateway's chat path does: the top-level
    /// `reasoning_effort` projected into the template kwargs under the
    /// request's own `chat_template_kwargs`, and into the protocol thinking
    /// preference (`none`/`minimal` off, a level no opinion).
    fn render_v4(
        tokenizer: &HuggingFaceTokenizer,
        chat_template_kwargs: Option<&HashMap<String, serde_json::Value>>,
        reasoning_effort: Option<&str>,
    ) -> String {
        let messages = vec![json!({ "role": "user", "content": "Hi" })];
        let mut kwargs: HashMap<String, serde_json::Value> = HashMap::new();
        if let Some(effort) = reasoning_effort {
            kwargs.insert("reasoning_effort".to_string(), json!(effort));
        }
        if let Some(explicit) = chat_template_kwargs {
            kwargs.extend(explicit.clone());
        }
        let params = ChatTemplateParams {
            add_generation_prompt: true,
            template_kwargs: (!kwargs.is_empty()).then_some(&kwargs),
            thinking: openai_protocol::chat::thinking_from_reasoning_effort(reasoning_effort),
            ..Default::default()
        };
        tokenizer.apply_chat_template(&messages, params).unwrap()
    }

    const V4_HIGH_PREFIX: &str = "Reasoning Effort: Absolute maximum";
    const V4_MAX_PREFIX: &str = "Reasoning Effort: Beyond maximum";

    #[test]
    fn deepseek_v4_defaults_to_thinking_with_the_engines_effort() {
        // A plain request renders what the engine's own server renders:
        // thinking mode with the `high` effort prompt.
        let (_tmp, tok) = write_dir(Some(&["DeepseekV4ForCausalLM"]));
        let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();
        let out = render_v4(&tokenizer, None, None);
        assert!(out.ends_with("<think>"), "{out}");
        assert!(out.contains(V4_HIGH_PREFIX), "{out}");
        assert_eq!(
            tokenizer.native_reasoning_effort_off_values(),
            &["none"],
            "the gateway disarms the reasoning parser on the renderer's off word"
        );
        assert!(tokenizer.renderer_capabilities().enable_thinking_alias);
    }

    #[test]
    fn deepseek_v4_thinking_switches_off_like_the_engine() {
        let (_tmp, tok) = write_dir(Some(&["DeepseekV4ForCausalLM"]));
        let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();
        let chat = "<｜User｜>Hi<｜Assistant｜></think>";
        // `thinking: false`, the engine's `enable_thinking: false` alias and
        // `reasoning_effort: "none"` (kwarg or top-level) render chat mode
        // without any effort prefix.
        for kwargs in [
            HashMap::from([("thinking".to_string(), json!(false))]),
            HashMap::from([("enable_thinking".to_string(), json!(false))]),
            HashMap::from([("reasoning_effort".to_string(), json!("none"))]),
            HashMap::from([
                ("thinking".to_string(), json!(false)),
                ("reasoning_effort".to_string(), json!("max")),
            ]),
        ] {
            let out = render_v4(&tokenizer, Some(&kwargs), None);
            assert!(out.ends_with(chat), "{kwargs:?}: {out}");
            assert!(!out.contains("Reasoning Effort"), "{kwargs:?}: {out}");
        }
        let out = render_v4(&tokenizer, None, Some("none"));
        assert!(out.ends_with(chat), "{out}");
        assert!(!out.contains("Reasoning Effort"), "{out}");
        // The typed `thinking.type: disabled` toggle arrives as `params.thinking`.
        let messages = vec![json!({ "role": "user", "content": "Hi" })];
        let out = tokenizer
            .apply_chat_template(
                &messages,
                ChatTemplateParams {
                    add_generation_prompt: true,
                    thinking: Some(false),
                    ..Default::default()
                },
            )
            .unwrap();
        assert!(out.ends_with(chat), "{out}");
        // The explicit toggle wins over the protocol preference, as it does
        // when the gateway arms the parser.
        let kwargs = HashMap::from([("thinking".to_string(), json!(true))]);
        let out = render_v4(&tokenizer, Some(&kwargs), Some("minimal"));
        assert!(out.ends_with("<think>"), "{out}");
    }

    #[test]
    fn deepseek_v4_maps_reasoning_effort_like_the_engine() {
        // The engine's mapping of the kwarg: `max` the top level, `low`/`minimal`/
        // `medium` the bottom one (no prefix), everything else `high`; the
        // top-level field counts the same as the kwarg.
        let (_tmp, tok) = write_dir(Some(&["DeepseekV4ForCausalLM"]));
        let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();
        for (effort, prefix) in [
            ("max", Some(V4_MAX_PREFIX)),
            ("high", Some(V4_HIGH_PREFIX)),
            ("xhigh", Some(V4_HIGH_PREFIX)),
            ("medium", None),
            ("minimal", None),
            ("low", None),
        ] {
            let kwargs = HashMap::from([("reasoning_effort".to_string(), json!(effort))]);
            for out in [
                render_v4(&tokenizer, Some(&kwargs), None),
                render_v4(&tokenizer, None, Some(effort)),
            ] {
                assert!(out.ends_with("<think>"), "{effort}: {out}");
                match prefix {
                    Some(prefix) => assert!(out.contains(prefix), "{effort}: {out}"),
                    None => assert!(!out.contains("Reasoning Effort"), "{effort}: {out}"),
                }
            }
        }
        // An explicit kwarg wins over the top-level field, as everywhere else
        // in the gateway.
        let kwargs = HashMap::from([("reasoning_effort".to_string(), json!("low"))]);
        let out = render_v4(&tokenizer, Some(&kwargs), Some("max"));
        assert!(!out.contains("Reasoning Effort"), "{out}");
    }

    #[test]
    fn deepseek_v4_renderer_passes_reasoning_effort() {
        let (_tmp, tok) = write_dir(Some(&["DeepseekV4ForCausalLM"]));
        let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();
        let messages = vec![json!({ "role": "user", "content": "Hello" })];
        let mut kwargs: HashMap<String, serde_json::Value> = HashMap::new();
        kwargs.insert(
            "reasoning_effort".to_string(),
            serde_json::Value::String("max".to_string()),
        );
        kwargs.insert("thinking".to_string(), serde_json::Value::Bool(true));
        let params = ChatTemplateParams {
            template_kwargs: Some(&kwargs),
            ..Default::default()
        };
        let out = tokenizer.apply_chat_template(&messages, params).unwrap();
        assert!(
            out.contains(V4_MAX_PREFIX),
            "expected the engine's max effort prefix in V4 output: {out}"
        );
        assert!(
            out.ends_with("<think>"),
            "thinking mode should leave a <think> token open"
        );
    }

    // -----------------------------------------------------------------------
    // 0731-vs-original effort-encoding identification
    // -----------------------------------------------------------------------

    /// Stand-ins for the `encoding/encoding_dsv4.py` each checkpoint ships;
    /// only the effort block differs between revisions, so the fingerprint is
    /// the 0731-only `max` prompt text.
    const BASE_ENCODER_SNIPPET: &str = r#"REASONING_EFFORT_MAX = (
    "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
)"#;
    const V0731_ENCODER_SNIPPET: &str = r#"REASONING_EFFORT_PROMPTS = {
    "low": "",
    "high": "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n",
    "max": "Reasoning Effort: Beyond maximum — exhaustive, relentless, and uncompromising.\n",
}"#;

    fn write_v4_model_dir(model_name: &str, encoder_source: Option<&str>) -> (TempDir, String) {
        write_model_dir(
            Some(model_name),
            Some(&["DeepseekV4ForCausalLM"]),
            None,
            encoder_source,
        )
    }

    fn render_with_native_effort(
        tokenizer: &HuggingFaceTokenizer,
        effort: &str,
    ) -> anyhow::Result<String> {
        let messages = vec![json!({ "role": "user", "content": "Hello" })];
        let kwargs = HashMap::from([("reasoning_effort".to_string(), json!(effort))]);
        let params = ChatTemplateParams {
            template_kwargs: Some(&kwargs),
            ..Default::default()
        };
        tokenizer.apply_chat_template(&messages, params)
    }

    #[test]
    fn shipped_encoder_fingerprint_beats_dir_name() {
        // Neutral dir name + 0731 encoder -> 0731 encoding.
        let (_tmp, tok) = write_v4_model_dir("my-local-v4-copy", Some(V0731_ENCODER_SNIPPET));
        let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();
        let out = render_with_native_effort(&tokenizer, "max").unwrap();
        assert!(out.contains("Reasoning Effort: Beyond maximum"), "{out}");

        // 0731 dir name + base encoder -> original encoding, where `max` is
        // the only level with a prompt; `low` is thinking mode without one.
        let (_tmp, tok) = write_v4_model_dir("DeepSeek-V4-Flash-0731", Some(BASE_ENCODER_SNIPPET));
        let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();
        let out = render_with_native_effort(&tokenizer, "max").unwrap();
        assert!(out.contains("Reasoning Effort: Absolute maximum"), "{out}");
        assert!(!out.contains("Beyond maximum"), "{out}");
        let out = render_with_native_effort(&tokenizer, "low").unwrap();
        assert!(!out.contains("Reasoning Effort:"), "{out}");
        assert!(out.ends_with("<think>"), "{out}");
    }

    #[test]
    fn missing_encoder_renders_the_engines_effort_table() {
        // Without the checkpoint's encoder (a tokenizer directory streamed from
        // a worker has none), whatever the directory is called, the effort
        // table is the one the engine's port renders for every V4 checkpoint.
        for name in ["DeepSeek-V4-Flash-0731", "DeepSeek-V4-Flash", "tmp8f2k1"] {
            let (_tmp, tok) = write_v4_model_dir(name, None);
            let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();
            let out = render_with_native_effort(&tokenizer, "max").unwrap();
            assert!(out.contains(V4_MAX_PREFIX), "{name}: {out}");
            let out = render_with_native_effort(&tokenizer, "high").unwrap();
            assert!(out.contains(V4_HIGH_PREFIX), "{name}: {out}");
            let out = render_with_native_effort(&tokenizer, "low").unwrap();
            assert!(!out.contains("Reasoning Effort:"), "{name}: {out}");
            assert!(out.ends_with("<think>"), "{name}: {out}");
        }
    }

    #[test]
    fn merged_public_effort_levels_render_thinking_without_a_prefix() {
        // `medium` and `minimal` are the bottom level to the engine: thinking
        // mode, no effort prompt.
        let (_tmp, tok) = write_v4_model_dir("DeepSeek-V4-Flash-0731", None);
        let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();
        for effort in ["medium", "minimal"] {
            let out = render_with_native_effort(&tokenizer, effort).unwrap();
            assert!(!out.contains("Reasoning Effort:"), "{effort}: {out}");
            assert!(out.ends_with("<think>"), "{effort}: {out}");
        }
    }

    #[test]
    fn native_effort_enables_thinking_unless_overridden() {
        let (_tmp, tok) = write_v4_model_dir("DeepSeek-V4-Flash-0731", None);
        let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();
        let messages = vec![json!({ "role": "user", "content": "Hello" })];

        // Effort alone implies thinking mode (the prefix only exists there).
        let out = render_with_native_effort(&tokenizer, "max").unwrap();
        assert!(out.ends_with("<think>"), "{out}");

        // An explicit thinking=false still wins and suppresses the prefix.
        let kwargs = HashMap::from([
            ("reasoning_effort".to_string(), json!("max")),
            ("thinking".to_string(), json!(false)),
        ]);
        let out = tokenizer
            .apply_chat_template(
                &messages,
                ChatTemplateParams {
                    template_kwargs: Some(&kwargs),
                    ..Default::default()
                },
            )
            .unwrap();
        assert!(out.ends_with("</think>"), "{out}");
        assert!(!out.contains("Reasoning Effort:"), "{out}");
    }

    // -----------------------------------------------------------------------
    // DeepSeek V4.1
    // -----------------------------------------------------------------------

    const V41_BOS: &str = "<\u{FF5C}begin\u{2581}of\u{2581}sentence\u{FF5C}>";
    const V41_EOS: &str = "<\u{FF5C}end\u{2581}of\u{2581}sentence\u{FF5C}>";
    /// The system header carrying the default budget (thinking mode only).
    const V41_DEFAULT_EFFORT_HEADER: &str = "<\u{FF5C}System\u{FF5C}>Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)\n\n";
    /// Tail of a single-user-turn prompt in thinking mode.
    const V41_THINKING_TAIL: &str = "<\u{FF5C}User\u{FF5C}>q<\u{FF5C}Assistant\u{FF5C}><think>";
    /// Tail of a single-user-turn prompt in chat mode.
    const V41_CHAT_TAIL: &str = "<\u{FF5C}User\u{FF5C}>q<\u{FF5C}Assistant\u{FF5C}></think>";

    fn v41_tokenizer() -> (TempDir, HuggingFaceTokenizer) {
        let (tmp, tok) = write_dir(Some(&["DeepseekV41ForCausalLM"]));
        let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();
        (tmp, tokenizer)
    }

    fn v41_tokenizer_with_added_tokens() -> (TempDir, HuggingFaceTokenizer, tokenizers::Tokenizer) {
        let (tmp, path) = write_dir(Some(&["DeepseekV41ForCausalLM"]));
        let mut document: serde_json::Value =
            serde_json::from_str(MIN_TOKENIZER_JSON).expect("minimal tokenizer JSON");
        let mut vocab = document["model"]["vocab"].as_object().unwrap().clone();
        let mut next_id = 128900;
        for id in vocab.values_mut() {
            *id = json!(next_id);
            next_id += 1;
        }
        for ch in
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 <>/\\\"{}[],:_-\n｜▁"
                .chars()
        {
            let text = ch.to_string();
            if !vocab.contains_key(&text) {
                vocab.insert(text, json!(next_id));
                next_id += 1;
            }
        }
        document["model"]["vocab"] = serde_json::Value::Object(vocab);
        let special = |id, content: &str| {
            json!({
                "id": id,
                "content": content,
                "special": true,
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
            })
        };
        document["added_tokens"] = json!([
            special(0, V41_BOS),
            special(1, V41_EOS),
            special(128803, "<｜User｜>"),
            special(128804, "<｜Assistant｜>"),
            special(128805, "<｜System｜>"),
            special(128806, "<｜latest_reminder｜>"),
            special(128821, "<think>"),
            special(128822, "</think>"),
            special(128825, "｜DSML｜"),
            special(129264, "<｜deepseek_image｜>"),
        ]);
        fs::write(&path, serde_json::to_vec(&document).unwrap()).unwrap();
        let tokenizer = HuggingFaceTokenizer::from_file(&path).unwrap();
        let raw = tokenizers::Tokenizer::from_file(&path).unwrap();
        (tmp, tokenizer, raw)
    }

    /// Render the single user turn `q` with a generation prompt.
    fn render_v41_turn(
        tokenizer: &HuggingFaceTokenizer,
        kwargs: Option<&HashMap<String, serde_json::Value>>,
        thinking: Option<bool>,
    ) -> anyhow::Result<String> {
        tokenizer.apply_chat_template(
            &[json!({"role": "user", "content": "q"})],
            ChatTemplateParams {
                add_generation_prompt: true,
                template_kwargs: kwargs,
                thinking,
                ..Default::default()
            },
        )
    }

    #[test]
    fn v41_architecture_selects_the_v41_renderer_with_thinking_on_by_default() {
        let (_tmp, tok) = write_dir(Some(&["DeepseekV41ForCausalLM"]));
        let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();
        assert_eq!(tokenizer.thinking_toggle(), ThinkingToggle::DefaultOn);
        assert_eq!(
            tokenizer.native_reasoning_effort_values(),
            &["low", "high", "xhigh", "max"]
        );
        assert!(tokenizer.think_in_prefill());
        let out = tokenizer
            .apply_chat_template(
                &[json!({"role": "user", "content": "q"})],
                ChatTemplateParams {
                    add_generation_prompt: true,
                    ..Default::default()
                },
            )
            .unwrap();
        assert!(
            out.starts_with("<\u{FF5C}begin\u{2581}of\u{2581}sentence\u{FF5C}><\u{FF5C}System\u{FF5C}>Reasoning Effort: 75 (range"),
            "{out}"
        );
        assert!(
            out.ends_with("<\u{FF5C}User\u{FF5C}>q<\u{FF5C}Assistant\u{FF5C}><think>"),
            "{out}"
        );
    }

    #[test]
    fn v41_model_type_alone_selects_the_v41_renderer() {
        // Some checkpoints identify themselves only through `model_type`,
        // with an unrelated `architectures` entry.
        let (_tmp, tok) = write_model_dir(
            None,
            Some(&["LlamaForCausalLM"]),
            Some("deepseek_v41"),
            None,
        );
        let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();
        assert_eq!(tokenizer.thinking_toggle(), ThinkingToggle::DefaultOn);
        assert_eq!(
            tokenizer.thinking_key_name(),
            Some(ThinkingKeyName::Thinking)
        );
        let out = render_v41_turn(&tokenizer, None, None).unwrap();
        assert_eq!(
            out,
            format!("{V41_BOS}{V41_DEFAULT_EFFORT_HEADER}{V41_THINKING_TAIL}")
        );
    }

    #[test]
    fn v41_reasoning_effort_none_disables_thinking_and_bad_values_error() {
        let (_tmp, tok) = write_dir(Some(&["DeepseekV41ForCausalLM"]));
        let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();
        let messages = vec![json!({"role": "user", "content": "q"})];

        let none_kwargs = HashMap::from([("reasoning_effort".to_string(), json!("none"))]);
        let off = tokenizer
            .apply_chat_template(
                &messages,
                ChatTemplateParams {
                    add_generation_prompt: true,
                    template_kwargs: Some(&none_kwargs),
                    ..Default::default()
                },
            )
            .unwrap();
        // Chat mode, and no effort line at all.
        assert_eq!(
            off,
            "<\u{FF5C}begin\u{2581}of\u{2581}sentence\u{FF5C}><\u{FF5C}User\u{FF5C}>q<\u{FF5C}Assistant\u{FF5C}></think>"
        );

        let medium_kwargs = HashMap::from([("reasoning_effort".to_string(), json!("invalid"))]);
        let err = tokenizer
            .apply_chat_template(
                &messages,
                ChatTemplateParams {
                    add_generation_prompt: true,
                    template_kwargs: Some(&medium_kwargs),
                    ..Default::default()
                },
            )
            .expect_err("expected an error for invalid reasoning_effort")
            .to_string();
        assert!(err.contains("Invalid reasoning effort"), "{err}");
    }

    #[test]
    fn v41_integer_effort_strings_are_restored_to_budgets() {
        // The gateway deserialises a top-level `"reasoning_effort": 42` into
        // the string "42" before forwarding it as a template kwarg; the shim
        // must hand `parse_reasoning_effort` the integer budget again.
        let (_tmp, tokenizer) = v41_tokenizer();
        let kw = HashMap::from([("reasoning_effort".to_string(), json!("42"))]);
        let out = render_v41_turn(&tokenizer, Some(&kw), None).unwrap();
        assert!(out.contains("Reasoning Effort: 42 (range"), "{out}");
        assert!(out.ends_with(V41_THINKING_TAIL), "{out}");
        // Out-of-range integers are still rejected by `parse_reasoning_effort`.
        for bad in ["0", "101"] {
            let kw = HashMap::from([("reasoning_effort".to_string(), json!(bad))]);
            let err = render_v41_turn(&tokenizer, Some(&kw), None)
                .expect_err(bad)
                .to_string();
            assert!(err.contains("Invalid reasoning effort"), "{bad}: {err}");
        }
        // Only the gateway's exact form, ASCII digits, is restored: a sign or
        // surrounding whitespace is not an integer budget and reaches the
        // strict parser as the string it is.
        for bad in ["+42", " 42 "] {
            let kw = HashMap::from([("reasoning_effort".to_string(), json!(bad))]);
            let err = render_v41_turn(&tokenizer, Some(&kw), None)
                .expect_err(bad)
                .to_string();
            assert_eq!(
                err,
                format!("DeepSeek V4.1 reasoning_effort invalid: Invalid reasoning effort `{bad}`: expected an integer within [1, 100] or one of low, high, xhigh, max"),
                "{bad:?}"
            );
        }
    }

    #[test]
    fn v41_explicit_thinking_true_overrides_reasoning_effort_none() {
        // Deliberate divergence from the engine's Python renderer, where "none" forces chat
        // mode: the gateway arms the reasoning parser from the explicit toggle
        // first, so the prompt must enter thinking mode as well. "none"
        // carries no effort level, so the default budget is rendered.
        let (_tmp, tokenizer) = v41_tokenizer();
        let kw = HashMap::from([
            ("reasoning_effort".to_string(), json!("none")),
            ("thinking".to_string(), json!(true)),
        ]);
        let out = render_v41_turn(&tokenizer, Some(&kw), None).unwrap();
        assert!(out.ends_with(V41_THINKING_TAIL), "{out}");
        assert!(out.contains("Reasoning Effort: 75 (range"), "{out}");
    }

    #[test]
    fn v41_reasoning_effort_minimal_enables_low_effort() {
        // Recipe maps minimal to low; only none switches thinking off.
        let (_tmp, tokenizer) = v41_tokenizer();
        let none_kw = HashMap::from([("reasoning_effort".to_string(), json!("none"))]);
        let minimal_kw = HashMap::from([("reasoning_effort".to_string(), json!("minimal"))]);
        let off = render_v41_turn(&tokenizer, Some(&none_kw), None).unwrap();
        let low = render_v41_turn(&tokenizer, Some(&minimal_kw), None).unwrap();
        assert!(low.ends_with(V41_THINKING_TAIL));
        assert!(low.contains("Reasoning Effort: 50 (range"));
        assert!(off.ends_with(V41_CHAT_TAIL), "{off}");
        // An explicit toggle wins over `minimal` exactly as it wins over `none`.
        let kw = HashMap::from([
            ("reasoning_effort".to_string(), json!("minimal")),
            ("thinking".to_string(), json!(true)),
        ]);
        let out = render_v41_turn(&tokenizer, Some(&kw), None).unwrap();
        assert!(out.ends_with(V41_THINKING_TAIL), "{out}");
        assert!(out.contains("Reasoning Effort: 50 (range"), "{out}");
    }

    #[test]
    fn v41_enable_thinking_alias_switches_the_mode_like_thinking() {
        // The engine's `enable_thinking` alias is honoured on both sides: the shim
        // reads it here and the gateway reads it when it arms the parser
        // (`renderer_capabilities().enable_thinking_alias`).
        let (_tmp, tokenizer) = v41_tokenizer();
        let kw = HashMap::from([("enable_thinking".to_string(), json!(false))]);
        let out = render_v41_turn(&tokenizer, Some(&kw), None).unwrap();
        assert!(out.ends_with("<｜Assistant｜></think>"), "{out}");
        // Agreeing duplicates are fine ...
        let kw = HashMap::from([
            ("enable_thinking".to_string(), json!(false)),
            ("thinking".to_string(), json!(false)),
        ]);
        let out = render_v41_turn(&tokenizer, Some(&kw), None).unwrap();
        assert!(out.ends_with("<｜Assistant｜></think>"), "{out}");
        // ... disagreeing ones are a request error naming both keys.
        let kw = HashMap::from([
            ("enable_thinking".to_string(), json!(true)),
            ("thinking".to_string(), json!(false)),
        ]);
        let err = render_v41_turn(&tokenizer, Some(&kw), None)
            .expect_err("disagreeing toggles must error")
            .to_string();
        assert!(err.contains("disagree"), "{err}");
        assert!(
            err.contains("enable_thinking") && err.contains("thinking"),
            "{err}"
        );
    }

    #[test]
    fn v41_reports_its_renderer_capabilities() {
        let (_tmp, tokenizer) = v41_tokenizer();
        let caps = tokenizer.renderer_capabilities();
        assert!(caps.enable_thinking_alias, "{caps:?}");
        assert!(caps.native_assistant_continuation, "{caps:?}");
        assert!(caps.raw_tool_call_arguments, "{caps:?}");
    }

    #[test]
    fn v41_non_boolean_thinking_kwarg_errors() {
        let (_tmp, tokenizer) = v41_tokenizer();
        let kw = HashMap::from([("thinking".to_string(), json!("yes"))]);
        let err = render_v41_turn(&tokenizer, Some(&kw), None)
            .expect_err("a non-boolean thinking kwarg must error")
            .to_string();
        assert!(err.contains("must be a boolean"), "{err}");
        // The message names the key and the offending value.
        assert!(err.contains("thinking") && err.contains("yes"), "{err}");
    }

    #[test]
    fn v41_non_boolean_drop_thinking_kwarg_errors() {
        // Same rule as `thinking`: present but not a JSON boolean is an error
        // naming the key and the value; JSON null counts as absent.
        let (_tmp, tokenizer) = v41_tokenizer();
        let kw = HashMap::from([("drop_thinking".to_string(), json!("false"))]);
        let err = render_v41_turn(&tokenizer, Some(&kw), None)
            .expect_err("a non-boolean drop_thinking kwarg must error")
            .to_string();
        assert_eq!(
            err,
            "DeepSeek V4.1: template_kwargs[\"drop_thinking\"] must be a boolean, got \"false\""
        );
        let kw = HashMap::from([("drop_thinking".to_string(), serde_json::Value::Null)]);
        assert_eq!(
            render_v41_turn(&tokenizer, Some(&kw), None).unwrap(),
            format!("{V41_BOS}{V41_DEFAULT_EFFORT_HEADER}{V41_THINKING_TAIL}")
        );
    }

    #[test]
    fn v41_params_thinking_false_renders_chat_mode() {
        // The gateway projects a top-level `reasoning_effort` of
        // `none`/`minimal` onto `params.thinking = Some(false)`.
        let (_tmp, tokenizer) = v41_tokenizer();
        let out = render_v41_turn(&tokenizer, None, Some(false)).unwrap();
        assert_eq!(out, format!("{V41_BOS}{V41_CHAT_TAIL}"));
    }

    #[test]
    fn v41_keeps_historical_reasoning_by_default() {
        let (_tmp, tokenizer) = v41_tokenizer();
        let messages = vec![
            json!({"role": "user", "content": "q1"}),
            json!({"role": "assistant", "reasoning_content": "r1", "content": "a1"}),
            json!({"role": "user", "content": "q2"}),
        ];
        let render = |kwargs: Option<&HashMap<String, serde_json::Value>>| {
            tokenizer
                .apply_chat_template(
                    &messages,
                    ChatTemplateParams {
                        add_generation_prompt: true,
                        template_kwargs: kwargs,
                        ..Default::default()
                    },
                )
                .unwrap()
        };
        assert!(render(None).contains("<think>r1</think>"));
        let drop = HashMap::from([("drop_thinking".to_string(), json!(true))]);
        assert!(!render(Some(&drop)).contains("r1"));
        let kw = HashMap::from([("drop_thinking".to_string(), json!(false))]);
        let out = render(Some(&kw));
        assert_eq!(
            out,
            format!(
                "{V41_BOS}{V41_DEFAULT_EFFORT_HEADER}<\u{FF5C}User\u{FF5C}>q1<\u{FF5C}Assistant\u{FF5C}><think>r1</think>a1{V41_EOS}<\u{FF5C}User\u{FF5C}>q2<\u{FF5C}Assistant\u{FF5C}><think>"
            )
        );
    }

    #[test]
    fn v4_history_retention_default_and_override_cover_both_revisions() {
        for source in [BASE_ENCODER_SNIPPET, V0731_ENCODER_SNIPPET] {
            let (_tmp, path) = write_v4_model_dir("deepseek-v4", Some(source));
            let tokenizer = HuggingFaceTokenizer::from_file(&path).unwrap();
            let messages = vec![
                json!({"role":"user", "content":"q1"}),
                json!({"role":"assistant", "reasoning_content":"old thought", "content":"a1"}),
                json!({"role":"user", "content":"q2"}),
            ];
            for (drop, retained) in [(None, true), (Some(false), true), (Some(true), false)] {
                let kwargs =
                    drop.map(|value| HashMap::from([("drop_thinking".to_string(), json!(value))]));
                let prompt = tokenizer
                    .apply_chat_template(
                        &messages,
                        ChatTemplateParams {
                            template_kwargs: kwargs.as_ref(),
                            ..Default::default()
                        },
                    )
                    .unwrap();
                assert_eq!(
                    prompt.contains("<think>old thought</think>"),
                    retained,
                    "{prompt}"
                );
            }
        }
    }

    #[test]
    fn v41_tools_are_in_a_leading_system_message() {
        // The engine's V4.1 rule: tools attach to the FIRST system message wherever
        // it is, even when the conversation opens with a user turn — unlike
        // V3.2/V4, which only rewrite a *leading* system/developer message.
        let (_tmp, tok) = write_dir(Some(&["DeepseekV41ForCausalLM"]));
        let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();
        let messages = vec![
            json!({"role": "user", "content": "hi"}),
            json!({"role": "assistant", "content": "hello"}),
            json!({"role": "system", "content": "S"}),
            json!({"role": "user", "content": "q"}),
        ];
        let tools = vec![json!({
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": { "city": { "type": "string" } },
                    "required": ["city"]
                }
            }
        })];
        let params = ChatTemplateParams {
            add_generation_prompt: true,
            tools: Some(&tools),
            ..Default::default()
        };
        let out = tokenizer.apply_chat_template(&messages, params).unwrap();
        assert!(out.contains("get_weather"), "tool name missing: {out}");
        assert!(out.find("## Tools").unwrap() < out.find("<｜User｜>hi").unwrap());
        assert!(out.contains("<｜System｜>S<｜User｜>q"));
        assert_eq!(out.matches("## Tools").count(), 1, "{out}");
    }

    #[test]
    fn v41_typed_function_without_optional_fields_renders_like_raw_json() {
        // D5: `openai_protocol::common::Function` skips its absent optional
        // fields (`description`, `strict`) when serialised, so a typed tool
        // reaches the schema block with the same bytes as the raw JSON a
        // client sends without those keys.
        let (_tmp, tokenizer) = v41_tokenizer();
        let parameters = json!({
            "type": "object",
            "properties": { "query": { "type": "string" } },
            "required": ["query"]
        });
        let typed = serde_json::to_value(Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "lookup".to_string(),
                description: None,
                parameters: parameters.clone(),
                strict: None,
                extra: Default::default(),
            },
        })
        .unwrap();
        let raw = json!({
            "type": "function",
            "function": { "name": "lookup", "parameters": parameters }
        });
        let render = |tool: serde_json::Value| {
            let tools = [tool];
            tokenizer
                .apply_chat_template(
                    &[json!({"role": "user", "content": "q"})],
                    ChatTemplateParams {
                        add_generation_prompt: true,
                        tools: Some(&tools),
                        ..Default::default()
                    },
                )
                .unwrap()
        };
        let from_typed = render(typed);
        assert_eq!(from_typed, render(raw));
        assert!(
            from_typed.contains(
                "\n\n{\"name\": \"lookup\", \"description\": \"\", \"parameters\": {\"type\": \"object\", \"properties\": {\"query\": {\"type\": \"string\"}}, \"required\": [\"query\"]}}\n\n"
            ),
            "{from_typed}"
        );
    }

    #[test]
    fn v41_add_generation_prompt_false_continues_the_final_assistant_message() {
        let (_tmp, tok) = write_dir(Some(&["DeepseekV41ForCausalLM"]));
        let tokenizer = HuggingFaceTokenizer::from_file(&tok).unwrap();
        let messages = vec![
            json!({"role": "user", "content": "q"}),
            json!({"role": "assistant", "reasoning_content": "r", "content": "Sure,"}),
        ];
        let out = tokenizer
            .apply_chat_template(
                &messages,
                ChatTemplateParams {
                    add_generation_prompt: false,
                    ..Default::default()
                },
            )
            .unwrap();
        // Thinking mode (default ON), reasoning kept (the assistant turn is
        // after the last user turn), and no trailing EOS since
        // `add_generation_prompt: false` continues the final message.
        assert_eq!(
            out,
            "<\u{FF5C}begin\u{2581}of\u{2581}sentence\u{FF5C}><\u{FF5C}System\u{FF5C}>Reasoning Effort: 75 (range 1-100, the higher the value, the more thorough the reasoning)\n\n<\u{FF5C}User\u{FF5C}>q<\u{FF5C}Assistant\u{FF5C}><think>r</think>Sure,"
        );
    }

    #[test]
    fn v41_chat_template_content_format_is_openai_v4_stays_string() {
        let (_tmp, tok41) = write_dir(Some(&["DeepseekV41ForCausalLM"]));
        let tokenizer41 = HuggingFaceTokenizer::from_file(&tok41).unwrap();
        assert_eq!(
            tokenizer41.chat_template_content_format(),
            ChatTemplateContentFormat::OpenAI
        );

        let (_tmp, tok4) = write_dir(Some(&["DeepseekV4ForCausalLM"]));
        let tokenizer4 = HuggingFaceTokenizer::from_file(&tok4).unwrap();
        assert_eq!(
            tokenizer4.chat_template_content_format(),
            ChatTemplateContentFormat::String
        );
    }

    #[test]
    fn deepseek_profile_thinking_matches_both_native_renderers() {
        use openai_protocol::{chat::ChatCompletionRequest, validated::Normalizable};
        for (model, architecture) in [
            ("deepseek-v4-pro", "DeepseekV4ForCausalLM"),
            ("deepseek-ai/DeepSeek-V4.1-Flash", "DeepseekV41ForCausalLM"),
        ] {
            let (_tmp, path) = write_dir(Some(&[architecture]));
            let tokenizer = HuggingFaceTokenizer::from_file(&path).unwrap();
            let v41 = model.contains("V4.1");
            for (extra, enabled) in [
                (json!({}), v41),
                (json!({"reasoning_effort":"minimal"}), v41),
                (json!({"reasoning_effort":"none"}), false),
                (
                    json!({"thinking":{"type":"disabled"},"reasoning_effort":"high"}),
                    false,
                ),
                (
                    json!({"thinking":{"type":"enabled"},"reasoning_effort":"none"}),
                    v41,
                ),
                (
                    json!({"chat_template_kwargs":{"enable_thinking":false},"reasoning_effort":"high"}),
                    false,
                ),
            ] {
                let mut body =
                    json!({"model":model,"messages":[{"role":"user","content":"hello"}]});
                body.as_object_mut()
                    .unwrap()
                    .extend(extra.as_object().unwrap().clone());
                let mut req: ChatCompletionRequest = serde_json::from_value(body.clone()).unwrap();
                req.normalize();
                // Match the gateway's merge: typed effort, then explicit kwargs.
                let mut kwargs = HashMap::new();
                if let Some(effort) = req.effective_reasoning_effort() {
                    kwargs.insert("reasoning_effort".to_string(), json!(effort));
                }
                if let Some(explicit) = &req.chat_template_kwargs {
                    kwargs.extend(explicit.clone());
                }
                let rendered = tokenizer
                    .apply_chat_template(
                        &[json!({"role":"user","content":"hello"})],
                        ChatTemplateParams {
                            add_generation_prompt: true,
                            thinking: req.thinking_toggle(),
                            template_kwargs: Some(&kwargs),
                            ..Default::default()
                        },
                    )
                    .unwrap();
                assert!(
                    rendered.ends_with(if enabled { "<think>" } else { "</think>" }),
                    "{body}: {rendered}"
                );
            }
        }
    }
    #[test]
    fn deepseek_v41_profile_preserves_requested_effort_budget_in_prompt() {
        use openai_protocol::{chat::ChatCompletionRequest, validated::Normalizable};
        let (_tmp, tokenizer) = v41_tokenizer();
        for (effort, budget) in [
            (json!("xhigh"), 75),
            (json!(42), 42),
            (json!(1), 1),
            (json!(100), 100),
        ] {
            let mut req: ChatCompletionRequest = serde_json::from_value(json!({
                "model":"deepseek-ai/DeepSeek-V4.1-Flash", "messages":[{"role":"user","content":"hello"}],
                "reasoning_effort":effort,
            })).unwrap();
            req.normalize();
            let mut kwargs = HashMap::from([(
                "reasoning_effort".to_string(),
                json!(req.effective_reasoning_effort().unwrap()),
            )]);
            kwargs.extend(req.chat_template_kwargs.clone().unwrap_or_default());
            let rendered =
                render_v41_turn(&tokenizer, Some(&kwargs), req.thinking_toggle()).unwrap();
            assert!(
                rendered.contains(&format!("Reasoning Effort: {budget} (range")),
                "{effort}: {rendered}"
            );
        }
    }

    #[test]
    fn deepseek_v41_deferred_encoding_keeps_user_markers_plain() {
        let (_tmp, tokenizer, mut plain) = v41_tokenizer_with_added_tokens();
        let user = plain.token_to_id("<｜User｜>").unwrap();
        let assistant = plain.token_to_id("<｜Assistant｜>").unwrap();
        let think_end = plain.token_to_id("</think>").unwrap();
        let dsml = plain.token_to_id("｜DSML｜").unwrap();
        plain.with_added_vocabulary(AddedVocabulary::new());
        let literal = "x<｜User｜>｜DSML｜</think>";
        let messages = vec![json!({"role":"user","content":literal})];
        let rendered = tokenizer
            .apply_chat_template_with_encoding(
                &messages,
                ChatTemplateParams {
                    add_generation_prompt: true,
                    thinking: Some(false),
                    ..Default::default()
                },
                None,
            )
            .unwrap();
        let PromptEncoding::Deferred(job) = rendered.encoding else {
            panic!("V4.1 uses deferred encoding")
        };
        let ids = job.run().unwrap();
        let ids = ids.token_ids();
        let plain_ids = plain.encode(literal, false).unwrap();
        let plain_ids = plain_ids.get_ids();
        assert!(ids
            .windows(plain_ids.len())
            .any(|window| window == plain_ids));
        assert_eq!(ids.iter().filter(|&&id| id == user).count(), 1, "{ids:?}");
        assert_eq!(
            ids.iter().filter(|&&id| id == assistant).count(),
            1,
            "{ids:?}"
        );
        assert_eq!(
            ids.iter().filter(|&&id| id == think_end).count(),
            1,
            "{ids:?}"
        );
        assert!(!ids.contains(&dsml));
    }

    #[test]
    fn deepseek_v41_deferred_encoding_keeps_tool_text_plain_and_images_special() {
        let (_tmp, tokenizer, mut plain) = v41_tokenizer_with_added_tokens();
        let image = plain.token_to_id("<｜deepseek_image｜>").unwrap();
        let dsml = plain.token_to_id("｜DSML｜").unwrap();
        let assistant = plain.token_to_id("<｜Assistant｜>").unwrap();
        plain.with_added_vocabulary(AddedVocabulary::new());
        let tools = vec![json!({
            "type":"function",
            "function":{"name":"f","description":"literal ｜DSML｜","parameters":{"type":"object"}}
        })];
        let messages = vec![
            json!({"role":"system","content":"s"}),
            json!({"role":"user","content":[
                {"type":"text","text":"before"},
                {"type":"image_url","image_url":{"url":"http://x/a.png"}},
            ]}),
            json!({"role":"assistant","content":"","tool_calls":[{
                "id":"c1","type":"function",
                "function":{"name":"f","arguments":"{\"x\":\"</think>\"}"}
            }]}),
            json!({"role":"tool","tool_call_id":"c1","content":"<｜User｜>"}),
        ];
        let rendered = tokenizer
            .apply_chat_template_with_encoding(
                &messages,
                ChatTemplateParams {
                    add_generation_prompt: true,
                    thinking: Some(false),
                    tools: Some(&tools),
                    ..Default::default()
                },
                None,
            )
            .unwrap();
        let PromptEncoding::Deferred(job) = rendered.encoding else {
            panic!("V4.1 uses deferred encoding")
        };
        let ids = job.run().unwrap();
        let ids = ids.token_ids();
        for literal in ["</think>", "<｜User｜>", "｜DSML｜"] {
            let expected = plain.encode(literal, false).unwrap();
            let expected = expected.get_ids();
            assert!(ids.windows(expected.len()).any(|window| window == expected));
        }
        assert_eq!(ids.iter().filter(|&&id| id == image).count(), 1, "{ids:?}");
        assert!(ids.contains(&dsml));
        assert!(ids.contains(&assistant));
    }
}
