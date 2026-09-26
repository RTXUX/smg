//! RTXUX frontend ports: Qwen implicit boundaries (7661150) and strict V4 boundaries (2920a09).
use reasoning_parser::ParserFactory;

#[test]
fn tool_boundaries_match_batch_at_every_transport_split() {
    for (name, opener) in [
        ("qwen3", "<tool_call>"),
        ("qwen_thinking", "<tool_call>"),
        (
            "deepseek_v4",
            "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"f\">",
        ),
        ("deepseek_v4", "<｜DSML｜invoke name=\"f\">"),
    ] {
        let text = format!("计划\n{opener}value <think> inside arguments");
        let mut batch = ParserFactory::new().create(name);
        batch.mark_reasoning_started();
        batch.mark_think_start_stripped();
        let expected = batch.detect_and_parse_reasoning(&text).unwrap();
        assert_eq!(expected.reasoning_text, "计划\n");
        assert_eq!(
            expected.normal_text,
            format!("{opener}value <think> inside arguments")
        );
        for (index, _) in text.char_indices() {
            let mut parser = ParserFactory::new().create(name);
            parser.mark_reasoning_started();
            parser.mark_think_start_stripped();
            let mut reasoning = String::new();
            let mut normal = String::new();
            for chunk in [&text[..index], &text[index..]] {
                let delta = parser.parse_reasoning_streaming_incremental(chunk).unwrap();
                reasoning.push_str(&delta.reasoning_text);
                normal.push_str(&delta.normal_text);
            }
            let delta = parser.flush().unwrap();
            reasoning.push_str(&delta.reasoning_text);
            normal.push_str(&delta.normal_text);
            assert_eq!(
                (reasoning, normal),
                (
                    expected.reasoning_text.clone(),
                    expected.normal_text.clone()
                ),
                "{name}, split {index}"
            );
        }
    }
}

#[test]
fn literal_and_unconfirmed_dsml_openers_remain_reasoning() {
    for text in [
        "Documentation mentions <｜DSML｜tool_calls> literally.",
        "<｜DSML｜tool_calls>\nnot an invoke</｜DSML｜tool_calls>",
        "<｜DSML｜tool_calls>",
        "<｜DSML｜tool_calls>\n<｜DSML｜inv",
    ] {
        let mut parser = ParserFactory::new().create("deepseek_v4");
        parser.mark_reasoning_started();
        assert_eq!(
            parser
                .detect_and_parse_reasoning(text)
                .unwrap()
                .reasoning_text,
            text
        );
        let mut reasoning = String::new();
        for ch in text.chars() {
            let delta = parser
                .parse_reasoning_streaming_incremental(&ch.to_string())
                .unwrap();
            assert!(delta.normal_text.is_empty());
            reasoning.push_str(&delta.reasoning_text);
        }
        reasoning.push_str(&parser.flush().unwrap().reasoning_text);
        assert_eq!(reasoning, text);
    }
}

#[test]
fn prefilled_reasoning_leaves_subsequent_tool_argument_markers_verbatim() {
    for (name, opener) in [
        ("qwen3", "<tool_call>"),
        ("deepseek_v4", "<｜DSML｜tool_calls>\n<｜DSML｜invoke name="),
    ] {
        let mut parser = ParserFactory::new().create(name);
        parser.mark_reasoning_started();
        let text = format!("plan{opener}\"f\">literal <think> argument</think>");
        let mut reasoning = String::new();
        let mut normal = String::new();
        for ch in text.chars() {
            let delta = parser
                .parse_reasoning_streaming_incremental(&ch.to_string())
                .unwrap();
            reasoning.push_str(&delta.reasoning_text);
            normal.push_str(&delta.normal_text);
        }
        let delta = parser.flush().unwrap();
        reasoning.push_str(&delta.reasoning_text);
        normal.push_str(&delta.normal_text);
        assert_eq!(reasoning, "plan");
        assert_eq!(normal, text.strip_prefix("plan").unwrap());
    }
}
