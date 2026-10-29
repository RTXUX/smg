//! Record prompt text and control-token provenance from the pinned recipe.
use std::{env, fs};

use deepseek_recipe_core::{
    conversation::{Conversation, ReasoningEffort, ResponseFormat},
    messages::{InputMessage, ToolCall},
    prompt::Prompt,
    special_tokens::SpecialTokensMap,
    tools::{ToolChoice, ToolDefinition},
};
use deepseek_recipe_encoding::{v4::dsv41::DeepseekV41Encoding, PromptEncoding, PromptTokenizer};
use serde_json::{json, Value};

fn tools(value: &Value) -> Vec<ToolDefinition> {
    value
        .as_array()
        .into_iter()
        .flatten()
        .map(|tool| {
            let function = tool.get("function").unwrap_or(tool);
            ToolDefinition {
                name: function["name"].as_str().unwrap().into(),
                description: function["description"].as_str().map(str::to_owned),
                parameters: function.get("parameters").cloned().unwrap_or(json!({})),
                strict: function["strict"].as_bool(),
                defer_loading: tool["defer_loading"].as_bool().unwrap_or(false),
            }
        })
        .collect()
}

fn content(value: &Value) -> Prompt {
    if let Some(text) = value.as_str() {
        return Prompt::plain(text);
    }
    let mut prompt = Prompt::new();
    for (i, part) in value.as_array().into_iter().flatten().enumerate() {
        if i > 0 {
            prompt.push_plain("\n\n");
        }
        match part["type"].as_str().unwrap() {
            "text" => prompt.push_plain(part["text"].as_str().unwrap()),
            "image_url" | "image" => prompt.push_special("<｜deepseek_image｜>"),
            other => panic!("unsupported fixture part {other}"),
        }
    }
    prompt
}

fn message(value: &Value) -> InputMessage {
    match value["role"].as_str().unwrap() {
        "system" => InputMessage::System {
            content: content(&value["content"]),
            image_sources: vec![],
            tools: value.get("tools").map(tools),
        },
        "user" | "developer" => InputMessage::User {
            content: content(&value["content"]),
            image_sources: vec![],
        },
        "tool" => InputMessage::Tool {
            content: content(&value["content"]),
            image_sources: vec![],
            tool_call_id: value["tool_call_id"].as_str().map(str::to_owned),
        },
        "latest_reminder" => InputMessage::LatestReminder {
            content: value["content"].as_str().unwrap().into(),
        },
        "assistant" => InputMessage::Assistant {
            content: content(&value["content"]).into_parts().0,
            reasoning_content: value["reasoning_content"].as_str().map(str::to_owned),
            prefix: value["prefix"].as_bool().unwrap_or(false),
            tool_calls: value["tool_calls"].as_array().map(|calls| {
                calls
                    .iter()
                    .map(|call| ToolCall {
                        id: call["id"].as_str().unwrap_or("").into(),
                        name: call["function"]["name"].as_str().unwrap().into(),
                        arguments: call["function"]["arguments"].as_str().unwrap().into(),
                    })
                    .collect()
            }),
        },
        other => panic!("unsupported fixture role {other}"),
    }
}

fn main() {
    let args: Vec<_> = env::args().collect();
    let mut input: Value = serde_json::from_slice(&fs::read(&args[1]).unwrap()).unwrap();
    let encoder = DeepseekV41Encoding::new().with_special_tokens(SpecialTokensMap {
        image: "<｜deepseek_image｜>".into(),
        ..Default::default()
    });
    let tokenizer = tokenizers::Tokenizer::from_file(&args[3]).unwrap();
    let encoder = encoder.with_tokenizer(PromptTokenizer::new(&tokenizer));
    for case in input["cases"].as_array_mut().unwrap() {
        let conversation = Conversation {
            messages: case["messages"]
                .as_array()
                .unwrap()
                .iter()
                .map(message)
                .collect(),
            thinking_mode: case["thinking"].as_bool().unwrap_or(true),
            reasoning_effort: case["effort"].as_str().map(|effort| match effort {
                "low" => ReasoningEffort::Low,
                "high" => ReasoningEffort::High,
                "xhigh" => ReasoningEffort::Xhigh,
                "max" => ReasoningEffort::Max,
                other => panic!("unsupported fixture effort {other}"),
            }),
            tools: tools(&case["tools"]),
            tool_choice: match case["tool_choice"].as_str() {
                Some("none") => ToolChoice::None,
                Some("required") => ToolChoice::Required,
                _ => ToolChoice::Auto,
            },
            response_format: match case["response_format"]["type"].as_str() {
                Some("json_object") => ResponseFormat::JsonObject,
                Some("json_schema") => ResponseFormat::JsonSchema(
                    case["response_format"]["json_schema"]["schema"].clone(),
                ),
                _ => ResponseFormat::Text,
            },
        };
        let rendered = encoder.render_conversation(&conversation);
        case["ids"] = json!(encoder.encode(&conversation).unwrap());
        case["text"] = json!(rendered.prompt);
        case["special_ranges"] = json!(rendered.special_ranges);
    }
    fs::write(
        &args[2],
        serde_json::to_string_pretty(&input).unwrap() + "\n",
    )
    .unwrap();
}
