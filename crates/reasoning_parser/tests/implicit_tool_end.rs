use reasoning_parser::{ParserFactory, ReasoningParser};

fn parser(model: &str) -> Box<dyn ReasoningParser> {
    let mut parser = ParserFactory::new().create(model);
    parser.mark_reasoning_started();
    parser
}

#[test]
fn tool_openers_end_reasoning_at_every_chunk_boundary() {
    for (model, tool) in [
        ("deepseek_v4", "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"f\">{}</｜DSML｜invoke></｜DSML｜tool_calls>"),
        ("deepseek_v4", "<｜DSML｜invoke name=\"f\">{}</｜DSML｜invoke>"),
    ] {
        let text = format!("<think>Plan. {tool}");
        let full = parser(model).detect_and_parse_reasoning(&text).unwrap();
        assert_eq!(full.reasoning_text, "Plan. ", "{model}");
        assert_eq!(full.normal_text, tool, "{model}");
        for cut in text.char_indices().map(|(index, _)| index).chain([text.len()]) {
            let mut parser = parser(model);
            let (mut reasoning, mut content) = (String::new(), String::new());
            for piece in [&text[..cut], &text[cut..]] {
                let delta = parser.parse_reasoning_streaming_incremental(piece).unwrap();
                reasoning.push_str(&delta.reasoning_text);
                content.push_str(&delta.normal_text);
            }
            let tail = parser.flush().unwrap();
            reasoning.push_str(&tail.reasoning_text);
            content.push_str(&tail.normal_text);
            assert_eq!(reasoning, full.reasoning_text, "{model} cut={cut}");
            assert_eq!(content, full.normal_text, "{model} cut={cut}");
            parser.reset();
            assert!(parser.detect_and_parse_reasoning("plain").unwrap().reasoning_text.is_empty());
        }
    }
}

#[test]
fn a_literal_or_unconfirmed_dsml_block_stays_reasoning() {
    for text in [
        "The format uses <｜DSML｜tool_calls> followed by invokes.",
        "Plan <｜DSML｜tool_calls>\n<｜DSML｜inv",
        "Plan <｜DSML｜tool_calls>\n",
    ] {
        let mut parser = parser("deepseek_v4");
        let mut reasoning = String::new();
        for character in text.chars() {
            let part = parser
                .parse_reasoning_streaming_incremental(&character.to_string())
                .unwrap();
            reasoning.push_str(&part.reasoning_text);
            assert!(part.normal_text.is_empty());
        }
        let tail = parser.flush().unwrap();
        reasoning.push_str(&tail.reasoning_text);
        assert!(tail.normal_text.is_empty());
        assert_eq!(reasoning, text);
    }
}
