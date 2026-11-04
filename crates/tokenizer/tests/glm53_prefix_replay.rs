//! The edited GLM template and its argument-preserving renderer.

#[cfg(test)]
mod tests {
    use llm_tokenizer::{
        chat_template::{ChatTemplateParams, ChatTemplateState},
        HuggingFaceTokenizer, TokenizerTrait,
    };
    use serde_json::json;

    const TEMPLATE: &str = include_str!("fixtures/glm53_flash/chat_template.jinja");

    fn tokenizer() -> HuggingFaceTokenizer {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("tokenizer.json");
        std::fs::write(
            &path,
            r#"{"version":"1.0","model":{"type":"BPE","vocab":{"a":0},"merges":[]}}"#,
        )
        .unwrap();
        std::fs::write(directory.path().join("chat_template.jinja"), TEMPLATE).unwrap();
        HuggingFaceTokenizer::from_file(path.to_str().unwrap()).unwrap()
    }

    #[test]
    fn glm53_replays_content_without_trimming() {
        let state = ChatTemplateState::new(Some(TEMPLATE.to_string())).unwrap();
        for content in ["\n\n    return 1\n\t ", " \t\n", "", "你好\n"] {
            let prompt = state
                .apply(
                    &[json!({"role":"user", "content":"Hi"})],
                    ChatTemplateParams {
                        add_generation_prompt: true,
                        ..Default::default()
                    },
                )
                .unwrap();
            let next = state.apply(&[
            json!({"role":"user", "content":"Hi"}),
            json!({"role":"assistant", "reasoning_content":"\n plan \n", "content":content}),
            json!({"role":"user", "content":"Next"}),
        ], ChatTemplateParams { add_generation_prompt: true, ..Default::default() }).unwrap();
            assert!(
                next.starts_with(&format!("{prompt}\n plan \n</think>{content}<|user|>")),
                "{next:?}"
            );
        }
    }

    #[test]
    fn glm53_renders_argument_values_as_written() {
        let tokenizer = tokenizer();
        let raw = r#"{"code":"    if ready:\n\t    return \"你好\"  \n", "empty":" \t\n", "quoted":"\"literal\"", "array":[1,2], "object":{ "z":1e+02,"a":"\u0061" }, "negative_zero":-0}"#;
        let messages = [
            json!({"role":"assistant", "content":"\n ", "reasoning_content":"R", "tool_calls":[{
                "id":"c", "type":"function", "function":{"name":"edit", "arguments":raw}
            }]}),
        ];
        let rendered = tokenizer
            .apply_chat_template(&messages, ChatTemplateParams::default())
            .unwrap();
        assert!(
            rendered.ends_with(concat!(
        "<|assistant|><think>R</think>\n <tool_call>edit",
        "<arg_key>code</arg_key><arg_value>    if ready:\n\t    return \"你好\"  \n</arg_value>",
        "<arg_key>empty</arg_key><arg_value> \t\n</arg_value>",
        "<arg_key>quoted</arg_key><arg_value>\"literal\"</arg_value>",
        "<arg_key>array</arg_key><arg_value>[1,2]</arg_value>",
        "<arg_key>object</arg_key><arg_value>{ \"z\":1e+02,\"a\":\"\\u0061\" }</arg_value>",
        "<arg_key>negative_zero</arg_key><arg_value>-0</arg_value></tool_call>"
    )),
            "{rendered:?}"
        );
        let mut object_form = messages;
        object_form[0]["tool_calls"][0]["function"]["arguments"] =
            serde_json::from_str(raw).unwrap();
        let rendered = tokenizer
            .apply_chat_template(&object_form, ChatTemplateParams::default())
            .unwrap();
        assert!(
            rendered.contains("<arg_value>    if ready:\n\t    return \"你好\"  \n</arg_value>")
        );
    }

    #[test]
    fn glm53_invalid_argument_json_is_rejected() {
        let tokenizer = tokenizer();
        for arguments in ["{", "[]", "null", "42"] {
            let messages = [json!({"role":"assistant", "content":"", "tool_calls":[{
                "function":{"name":"edit", "arguments":arguments}
            }]})];
            assert!(tokenizer
                .apply_chat_template(&messages, ChatTemplateParams::default())
                .is_err());
        }
    }

    #[test]
    fn raw_argument_helper_works_without_model_metadata_and_does_not_rewrite_templates() {
        let mut tokenizer = tokenizer();
        assert!(tokenizer.renderer_capabilities().raw_tool_call_arguments);
        tokenizer
            .set_chat_template(
                concat!(
        "{% for k, v in tool_argument_items(messages[0].tool_calls[0].function.arguments) %}",
        "{{k}}={{v}};{% endfor %}"
    )
                .to_string(),
            )
            .unwrap();
        assert!(tokenizer.renderer_capabilities().raw_tool_call_arguments);
        assert_eq!(
            tokenizer
                .apply_chat_template(
                    &[json!({"role":"assistant", "tool_calls":[{
                        "function":{"name":"f", "arguments":r#"{"array":[1,2],"text":"  code\n"}"#}
                    }]})],
                    ChatTemplateParams::default()
                )
                .unwrap(),
            "array=[1,2];text=  code\n;"
        );
        tokenizer
            .set_chat_template(
                concat!(
                    "{# tool_argument_items(arguments) #}",
                    "{{ 'tool_argument_items' if false else '' }}",
                    "{{ messages[0].content.strip() }}"
                )
                .to_string(),
            )
            .unwrap();
        assert!(!tokenizer.renderer_capabilities().raw_tool_call_arguments);
        assert_eq!(
            tokenizer
                .apply_chat_template(
                    &[json!({"role":"user", "content":"  hi  "})],
                    ChatTemplateParams::default()
                )
                .unwrap(),
            "hi"
        );
    }
}
