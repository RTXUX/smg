//! Generated text -> Symphony -> Chat response -> typed request -> GLM prompt.

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, sync::Arc};

    use llm_tokenizer::{
        CacheConfig, CachedTokenizer, HuggingFaceTokenizer, TokenizerTrait as Tokenizer,
    };
    use openai_protocol::chat::ChatCompletionRequest;
    use serde_json::{json, Value};
    use smg::routers::grpc::utils::process_chat_messages;
    use symphony::{
        adapt::chat,
        engine::Engine,
        event::{Event, Events},
        formats,
        input::{EngineFinish, Input},
        parser::Parser,
        Declared,
    };

    const TEMPLATE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../crates/tokenizer/tests/fixtures/glm53_flash/chat_template.jinja"
    ));

    fn tokenizer() -> (Arc<dyn Tokenizer>, bool) {
        let directory = tempfile::tempdir().unwrap();
        let real = std::env::var("SMG_GLM53_TOKENIZER").ok();
        let path = if let Some(path) = &real {
            std::path::PathBuf::from(path)
        } else {
            let path = directory.path().join("tokenizer.json");
            std::fs::write(
                &path,
                r#"{"version":"1.0","model":{"type":"BPE","vocab":{"a":0},"merges":[]}}"#,
            )
            .unwrap();
            path
        };
        let mut tokenizer = HuggingFaceTokenizer::from_file(path.to_str().unwrap()).unwrap();
        tokenizer.set_chat_template(TEMPLATE.to_string()).unwrap();
        assert!(tokenizer.renderer_capabilities().raw_tool_call_arguments);
        (
            Arc::new(CachedTokenizer::new(
                Arc::new(tokenizer),
                CacheConfig {
                    enable_l1: true,
                    ..Default::default()
                },
            )),
            real.is_some(),
        )
    }

    fn request(messages: &[Value]) -> ChatCompletionRequest {
        // Exercise the ordinary wire JSON and typed protocol boundary as well.
        serde_json::from_str(
            &json!({
                "model":"glm-5.3-flash", "messages":messages,
                "tools":[{"type":"function", "function":{"name":"edit", "parameters":{
                    "type":"object", "properties":{
                        "code":{"type":"string"}, "blank":{"type":["string","null"]},
                        "quoted":{"$ref":"#/$defs/text"}, "array":{"type":"array"},
                        "object":{"type":"object"}, "number":{"type":"number"},
                        "integer":{"type":"integer"}, "nullable":{"type":["string","null"]}
                    }, "$defs":{"text":{"type":"string"}}
                }}}]
            })
            .to_string(),
        )
        .unwrap()
    }

    fn render(tokenizer: &dyn Tokenizer, request: &ChatCompletionRequest) -> String {
        process_chat_messages(request, tokenizer, None)
            .unwrap()
            .text
    }

    fn parse(
        prompt: &str,
        generated: &str,
        request: &ChatCompletionRequest,
        size: usize,
    ) -> Vec<Event<'static>> {
        let mut parser = Engine::new(
            formats::glm(),
            Declared::of(request.tools.as_deref().unwrap()),
        );
        let mut events = Events::new();
        parser
            .feed(
                Input::Prompt {
                    token_ids: &[],
                    text: prompt,
                },
                &mut events,
            )
            .unwrap();
        let chars: Vec<_> = generated
            .char_indices()
            .map(|(i, _)| i)
            .chain([generated.len()])
            .collect();
        for indices in chars[..chars.len() - 1].chunks(size) {
            let start = indices[0];
            let end = generated[indices[indices.len() - 1]..]
                .chars()
                .next()
                .unwrap()
                .len_utf8()
                + indices[indices.len() - 1];
            parser
                .feed(
                    Input::Delta {
                        token_ids: &[],
                        text: &generated[start..end],
                        spans: &[],
                    },
                    &mut events,
                )
                .unwrap();
        }
        parser
            .feed(
                Input::End {
                    finish: EngineFinish::Stop,
                },
                &mut events,
            )
            .unwrap();
        events.into_owned().drain()
    }

    fn assembled_stream(events: &[Event]) -> Value {
        let mut content = String::new();
        let mut reasoning = String::new();
        let mut calls = BTreeMap::<u32, Value>::new();
        for choice in chat::deltas(0, events) {
            let delta = choice.delta;
            content.push_str(delta.content.as_deref().unwrap_or_default());
            reasoning.push_str(delta.reasoning_content.as_deref().unwrap_or_default());
            for call in delta.tool_calls.unwrap_or_default() {
                let item = calls.entry(call.index).or_insert_with(|| {
                    json!({
                        "id":"", "type":"function", "function":{"name":"", "arguments":""}
                    })
                });
                if let Some(id) = call.id {
                    item["id"] = json!(id);
                }
                if let Some(function) = call.function {
                    if let Some(name) = function.name {
                        item["function"]["name"] = json!(name);
                    }
                    if let Some(arguments) = function.arguments {
                        let mut joined = item["function"]["arguments"].as_str().unwrap().to_owned();
                        joined.push_str(&arguments);
                        item["function"]["arguments"] = json!(joined);
                    }
                }
            }
        }
        json!({"role":"assistant", "content":content, "reasoning_content":reasoning,
        "tool_calls":calls.into_values().collect::<Vec<_>>()})
    }

    #[test]
    fn glm53_symphony_chat_replay_preserves_generated_prefix() {
        let (tokenizer, real_tokens) = tokenizer();
        let first_messages = vec![json!({"role":"user", "content":"Edit this Python function."})];
        let initial = request(&first_messages);
        let prompt = render(tokenizer.as_ref(), &initial);
        let tool = concat!(
        "<tool_call>edit",
        "<arg_key>code</arg_key><arg_value>    if ready:\n\t    return \"你好\"  \n</arg_value>",
        "<arg_key>blank</arg_key><arg_value> \t\n</arg_value>",
        "<arg_key>quoted</arg_key><arg_value>\"literal\"</arg_value>",
        "<arg_key>array</arg_key><arg_value>[1,2]</arg_value>",
        "<arg_key>object</arg_key><arg_value>{ \"z\":1e+02,\"a\":\"\\u0061\" }</arg_value>",
        "<arg_key>number</arg_key><arg_value>1.00e+02</arg_value>",
        "<arg_key>integer</arg_key><arg_value>-0</arg_value>",
        "<arg_key>nullable</arg_key><arg_value>null</arg_value></tool_call>"
    );
        for generated in [
            "\n plan \n</think>\n\n    return 1\n\t ".to_owned(),
            "</think> \t\n".to_owned(),
            format!("R</think>\n \t{tool}"),
            format!("R</think>\nEditing now. \n{tool}{tool}"),
        ] {
            for size in [1, 7, usize::MAX] {
                let events = parse(&prompt, &generated, &initial, size);
                assert!(!events.iter().any(|e| matches!(e, Event::Malformed { .. })));
                let whole = serde_json::to_value(chat::message(0, &events).message).unwrap();
                let streaming = assembled_stream(&events);
                assert_eq!(whole["content"], streaming["content"]);
                for assistant in [whole, streaming] {
                    let mut messages = first_messages.clone();
                    messages.push(assistant.clone());
                    let calls = assistant["tool_calls"]
                        .as_array()
                        .cloned()
                        .unwrap_or_default();
                    if calls.is_empty() {
                        messages.push(json!({"role":"user", "content":"Continue."}));
                    } else {
                        // Results arrive in reverse order; the template consistently orders a
                        // complete block by its calls, without changing the generated prefix.
                        for call in calls.iter().rev() {
                            let arguments: Value = serde_json::from_str(
                                call["function"]["arguments"].as_str().unwrap(),
                            )
                            .unwrap();
                            assert_eq!(
                                arguments["code"],
                                "    if ready:\n\t    return \"你好\"  \n"
                            );
                            assert_eq!(arguments["blank"], " \t\n");
                            assert_eq!(arguments["quoted"], "\"literal\"");
                            messages.push(
                                json!({"role":"tool", "tool_call_id":call["id"], "content":"done"}),
                            );
                        }
                    }
                    let next = render(tokenizer.as_ref(), &request(&messages));
                    assert!(
                        next.starts_with(&format!("{prompt}{generated}")),
                        "size={size}, generated={generated:?}, next={next:?}"
                    );
                    if real_tokens {
                        let mut executed = tokenizer
                            .encode(&prompt, false)
                            .unwrap()
                            .token_ids()
                            .to_vec();
                        executed.extend_from_slice(
                            tokenizer.encode(&generated, false).unwrap().token_ids(),
                        );
                        assert!(tokenizer
                            .encode(&next, false)
                            .unwrap()
                            .token_ids()
                            .starts_with(&executed));
                    }
                    // Complete another assistant turn, then append a user: retained reasoning
                    // and both previous calls must stay byte-identical across the whole cycle.
                    messages.push(json!({"role":"assistant", "reasoning_content":"Done.", "content":"\n Fixed. \n"}));
                    messages.push(json!({"role":"user", "content":"Check it."}));
                    let final_prompt = render(tokenizer.as_ref(), &request(&messages));
                    assert!(final_prompt
                        .starts_with(&format!("{next}Done.</think>\n Fixed. \n<|user|>")));
                }
            }
        }
    }
}
