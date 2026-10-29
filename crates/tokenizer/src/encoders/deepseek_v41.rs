// DeepSeek V4.1 prompt encoding, aligned with deepseek-recipe at
// 935817c66f5d6e64838f419692545f06787e485f. See scripts/deepseek_recipe
// for the independent oracle, provenance, and upstream MIT license.

use std::collections::HashMap;

use serde_json::Value;

// `DsEncodingError` lives in `deepseek_common` (shared with V4); re-exported
// here because it is part of this module's public API, and `pub(super)` items
// are not reachable from outside `encoders` on their own.
pub use super::deepseek_common::DsEncodingError;
// Reuse the public ThinkingMode enum from the V3.2 module to keep the
// "thinking" / "chat" mode invariant identical across DeepSeek versions.
pub use super::deepseek_v32::ThinkingMode;
// Message-shape preprocessing, drop-thinking, DSML argument rendering and the
// tools template are V4's too, apart from the tag strings and the last-turn
// definition; see `deepseek_common` for the shared implementation and its
// own doc comments.
use super::{
    deepseek_common::{to_json, tool_calls_from_openai_format, write_v4_tools_template, DsmlTags},
    Prompt, RendererProfile,
};
use crate::{
    chat_template::{ChatTemplateContentFormat, ThinkingKeyName, ThinkingToggle},
    traits::RendererCapabilities,
};

// ---------------------------------------------------------------------------
// Special-token constants — copied verbatim from the Python source.
// ---------------------------------------------------------------------------
pub const BOS_TOKEN: &str = "<｜begin▁of▁sentence｜>";
pub const EOS_TOKEN: &str = "<｜end▁of▁sentence｜>";
pub const THINKING_START_TOKEN: &str = "<think>";
pub const THINKING_END_TOKEN: &str = "</think>";
pub const DSML_TOKEN: &str = "｜DSML｜";
/// What every image content part renders as; the image payload itself is the
/// serving layer's business, the prompt only carries this marker.
pub const IMAGE_PLACEHOLDER: &str = "<｜deepseek_image｜>";
const USER_SP_TOKEN: &str = "<｜User｜>";
const ASSISTANT_SP_TOKEN: &str = "<｜Assistant｜>";
const SYSTEM_SP_TOKEN: &str = "<｜System｜>";
const LATEST_REMINDER_SP_TOKEN: &str = "<｜latest_reminder｜>";
/// V4.1's DSML dialect: a leading space on the block name and on both tags
/// (V4 uses bare `tool_calls`/`invoke`/`parameter`).
const TOOL_CALLS_BLOCK_NAME: &str = " calls";
const DSML_TAGS: DsmlTags = DsmlTags {
    invoke: " invoke",
    parameter: " parameter",
};

// ---------------------------------------------------------------------------
// Reasoning effort
// ---------------------------------------------------------------------------
/// The effort names the Python encoder accepts (`REASONING_EFFORT_MAPPINGS`).
pub const NATIVE_EFFORT_VALUES: &[&str] = &["low", "high", "xhigh", "max"];

/// What this renderer answers around a render (see [`RendererProfile`]).
pub(crate) const PROFILE: RendererProfile = RendererProfile {
    // Message parts are preserved (OpenAI wire format) so the gateway passes
    // `image_url` / `image` parts through as a list instead of flattening to
    // a string; the encoder turns each part into a placeholder at its
    // position. Part *order* is still the gateway's media-part order (the
    // multimodal spec declares `Authored`; without a spec the gateway moves
    // media first).
    content_format: Some(ChatTemplateContentFormat::OpenAI),
    // Thinking defaults ON: `reasoning_effort: "none"` or an explicit
    // `thinking: false` (or the engine's `enable_thinking` alias, see the
    // capabilities) turns it off.
    thinking_toggle: ThinkingToggle::DefaultOn,
    thinking_key_name: ThinkingKeyName::Thinking,
    native_effort_values: NATIVE_EFFORT_VALUES,
    native_effort_off_values: &["none"],
    // `<｜Assistant｜><think>` ends the prompt when thinking is on.
    think_in_prefill: true,
    // The shim honours the engine's `enable_thinking` alias, renders a
    // trailing assistant message itself when `add_generation_prompt` is
    // false, and parses tool-call `arguments` strings with the reference's
    // tolerance.
    capabilities: RendererCapabilities {
        required_tool_prefix: Some("\n\n<｜DSML｜ calls>\n"),
        message_text_separator: Some("\n\n"),
        authored_media_order: true,
        enable_thinking_alias: true,
        native_assistant_continuation: true,
        raw_tool_call_arguments: true,
        declares_tools_under_tool_choice_none: false,
    },
    thinking_mode: Some(resolve_thinking),
};

/// The recipe's explicit thinking control outranks effort. Template aliases
/// are supported for callers rendering a prompt without a protocol request.
pub(crate) fn resolve_thinking(
    kwargs: Option<&HashMap<String, Value>>,
    thinking: Option<bool>,
) -> bool {
    thinking
        .or_else(|| {
            kwargs.and_then(|kwargs| {
                kwargs
                    .get("thinking")
                    .and_then(Value::as_bool)
                    .or_else(|| kwargs.get("enable_thinking").and_then(Value::as_bool))
                    .or_else(|| {
                        kwargs
                            .get("thinking_mode")
                            .and_then(Value::as_str)
                            .and_then(|mode| match mode {
                                "enabled" | "thinking" | "adaptive" => Some(true),
                                "disabled" | "chat" => Some(false),
                                _ => None,
                            })
                    })
            })
        })
        .unwrap_or_else(|| {
            kwargs
                .and_then(|k| k.get("reasoning_effort"))
                .and_then(Value::as_str)
                != Some("none")
        })
}
pub const REASONING_EFFORT_TEMPLATE: &str =
    "Reasoning Effort: {budget} (range 1-100, the higher the value, the more thorough the reasoning)\n\n";
const DEFAULT_REASONING_EFFORT: ReasoningEffort = ReasoningEffort::High;

/// Reasoning effort for the V4.1 numeric prompt prefix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReasoningEffort {
    Low,
    High,
    XHigh,
    Max,
    /// A raw budget, as accepted by the API in `1..=100`.
    Budget(u8),
}

impl ReasoningEffort {
    /// deepseek-recipe's effort table; xhigh and high share the same budget.
    pub const fn budget(self) -> u8 {
        match self {
            Self::Low => 50,
            Self::High => 75,
            Self::XHigh => 75,
            Self::Max => 100,
            Self::Budget(n) => n,
        }
    }

    /// Parse one of [`NATIVE_EFFORT_VALUES`].
    fn from_native(value: &str) -> Option<Self> {
        match value {
            "minimal" | "low" => Some(Self::Low),
            "medium" | "high" => Some(Self::High),
            "xhigh" => Some(Self::XHigh),
            "ultra" | "max" => Some(Self::Max),
            _ => None,
        }
    }
}

/// Parse the request's `reasoning_effort` field.
///
/// Recipe effort names and aliases, or an explicit integer budget in 1..=100.
/// The caller handles `none` as a thinking switch. Floats, booleans, unknown
/// names and out-of-range integers are rejected.
pub fn parse_reasoning_effort(value: &Value) -> Result<Option<ReasoningEffort>, DsEncodingError> {
    match value {
        Value::Null => Ok(None),
        Value::String(name) => ReasoningEffort::from_native(name)
            .map(Some)
            .ok_or_else(|| DsEncodingError::InvalidReasoningEffort(name.clone())),
        // Python's `type(effort) is int` rejects floats and booleans alike.
        Value::Number(number) => number
            .as_u64()
            .and_then(|budget| u8::try_from(budget).ok())
            .filter(|budget| (1..=100).contains(budget))
            .map(|budget| Some(ReasoningEffort::Budget(budget)))
            .ok_or_else(|| DsEncodingError::InvalidReasoningEffort(number.to_string())),
        other => Err(DsEncodingError::InvalidReasoningEffort(other.to_string())),
    }
}

/// Mirrors `render_reasoning_effort`: the prefix exists in thinking mode, at
/// index 0, only.
fn render_reasoning_effort(
    index: usize,
    thinking_mode: ThinkingMode,
    effort: Option<ReasoningEffort>,
) -> String {
    if index != 0 || thinking_mode != ThinkingMode::Thinking {
        return String::new();
    }
    let budget = effort.unwrap_or(DEFAULT_REASONING_EFFORT).budget();
    REASONING_EFFORT_TEMPLATE.replace("{budget}", &budget.to_string())
}

// ---------------------------------------------------------------------------
// Templates
// ---------------------------------------------------------------------------
/// Mirrors V4.1's `TOOLS_TEMPLATE`: the V4 family's text with this dialect's
/// block name and spaced tags.
fn render_tools(tools: &[Value]) -> Prompt {
    let mut prompt = Prompt::default();
    write_v4_tools_template(
        &mut prompt,
        TOOL_CALLS_BLOCK_NAME,
        &DSML_TAGS,
        &tools.iter().map(to_json).collect::<Vec<_>>().join("\n"),
    );
    prompt
}

fn render_response_format(schema: &Value) -> String {
    format!(
        "## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n{}",
        to_json(schema)
    )
}

/// Read a string field, treating missing/null content as empty.
fn text<'a>(value: &'a Value, key: &str) -> &'a str {
    value.get(key).and_then(Value::as_str).unwrap_or("")
}

fn normalized_tools(tools: &[Value], hide_deferred: bool) -> Vec<Value> {
    tools.iter().filter(|tool| !hide_deferred || tool["defer_loading"] != true)
        .map(|tool| {
            let function = tool.get("function").unwrap_or(tool);
            serde_json::json!({
                "name": function["name"],
                "description": function["description"].as_str().unwrap_or(""),
                "parameters": function.get("parameters").filter(|p| !p.is_null()).cloned().unwrap_or_else(|| serde_json::json!({})),
            })
        }).collect()
}

/// Preserve image identity while rendering content parts. An identical
/// spelling inside client text remains ordinary text, including image markers.
fn push_content(prompt: &mut Prompt, content: &Value) -> Result<(), DsEncodingError> {
    if let Some(text) = content.as_str() {
        prompt.push_plain(text);
    } else if let Some(parts) = content.as_array() {
        for (index, part) in parts.iter().enumerate() {
            if index > 0 {
                prompt.push_plain("\n\n");
            }
            match text(part, "type") {
                "text" | "input_text" | "output_text" => prompt.push_plain(text(part, "text")),
                "image" | "image_url" | "input_image" | "image_pil" => {
                    prompt.push_special(IMAGE_PLACEHOLDER);
                }
                other => return Err(DsEncodingError::UnsupportedContentPart(other.into())),
            }
        }
    }
    Ok(())
}

/// Only grounding markers in assistant text have special-token identity.
fn push_assistant_content(prompt: &mut Prompt, content: &str) {
    const GROUNDING: [&str; 8] = [
        "<｜ref｜>",
        "<｜/ref｜>",
        "<｜box｜>",
        "<｜/box｜>",
        "<｜point｜>",
        "<｜/point｜>",
        "<｜polygon｜>",
        "<｜/polygon｜>",
    ];
    let mut cursor = 0;
    while let Some((index, token)) = GROUNDING
        .iter()
        .filter_map(|token| {
            content[cursor..]
                .find(token)
                .map(|index| (index + cursor, *token))
        })
        .min_by_key(|(index, _)| *index)
    {
        prompt.push_plain(&content[cursor..index]);
        prompt.push_special(token);
        cursor = index + token.len();
    }
    prompt.push_plain(&content[cursor..]);
}

fn write_recipe_arguments(prompt: &mut Prompt, arguments: &Value) {
    let original = arguments
        .as_str()
        .map(str::to_owned)
        .unwrap_or_else(|| to_json(arguments));
    let parsed =
        serde_json::from_str::<serde_json::Map<String, Value>>(&original).unwrap_or_else(|_| {
            serde_json::Map::from_iter([("arguments".into(), Value::String(original))])
        });
    for (index, (name, value)) in parsed.iter().enumerate() {
        if index > 0 {
            prompt.push_plain("\n");
        }
        let (is_string, value) = match value {
            Value::String(text) => ("true", text.clone()),
            value => ("false", to_json(value)),
        };
        prompt.push_plain("<");
        prompt.push_special(DSML_TOKEN);
        prompt.push_plain(format!(
            " parameter name=\"{name}\" string=\"{is_string}\">{value}</"
        ));
        prompt.push_special(DSML_TOKEN);
        prompt.push_plain(" parameter>");
    }
}

fn tool_calls_block(tool_calls: &[Value]) -> Prompt {
    let mut prompt = Prompt::default();
    prompt.push_plain("\n\n<");
    prompt.push_special(DSML_TOKEN);
    prompt.push_plain(format!("{TOOL_CALLS_BLOCK_NAME}>\n"));
    for (index, tool_call) in tool_calls.iter().enumerate() {
        if index > 0 {
            prompt.push_plain("\n");
        }
        let name = text(tool_call, "name");
        let arguments = tool_call.get("arguments").unwrap_or(&Value::Null);
        prompt.push_plain("<");
        prompt.push_special(DSML_TOKEN);
        prompt.push_plain(format!("{} name=\"{name}\">\n", DSML_TAGS.invoke));
        write_recipe_arguments(&mut prompt, arguments);
        prompt.push_plain("\n</");
        prompt.push_special(DSML_TOKEN);
        prompt.push_plain(format!("{}>", DSML_TAGS.invoke));
    }
    prompt.push_plain("\n</");
    prompt.push_special(DSML_TOKEN);
    prompt.push_plain(format!("{TOOL_CALLS_BLOCK_NAME}>"));
    prompt
}

/// Parameters for [`encode_messages`].
///
/// `context` is intentionally omitted: SMG always renders from scratch, so the
/// Python default of `context=None` always applies.
#[derive(Debug, Clone, Copy)]
pub struct EncodeParams {
    pub add_default_bos_token: bool,
    pub drop_thinking: bool,
    pub reasoning_effort: Option<ReasoningEffort>,
}
impl Default for EncodeParams {
    fn default() -> Self {
        Self {
            add_default_bos_token: true,
            drop_thinking: true,
            reasoning_effort: None,
        }
    }
}

/// Encode a list of OpenAI-style messages into a DeepSeek V4.1 prompt string.
///
/// The signature mirrors the Python `encode_messages` function; `context` is
/// omitted because SMG always renders from scratch.
#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "public API mirrors the documented Rust signature with a borrow"
)]
pub fn encode_messages(
    messages: &[Value],
    thinking_mode: ThinkingMode,
    params: &EncodeParams,
) -> Result<String, DsEncodingError> {
    render_messages(messages, thinking_mode, params).map(|prompt| prompt.into_parts().0)
}

/// Render with the provenance of every explicit added token preserved for deferred encoding.
#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "matches the existing encode_messages signature"
)]
pub(crate) fn render_messages(
    messages: &[Value],
    thinking_mode: ThinkingMode,
    params: &EncodeParams,
) -> Result<Prompt, DsEncodingError> {
    render_conversation(messages, thinking_mode, *params, None, None, None)
}

/// Render the recipe's conversation envelope without folding tool results into
/// user text or inferring special-token identity from client-controlled bytes.
pub(crate) fn render_conversation(
    messages: &[Value],
    thinking_mode: ThinkingMode,
    params: EncodeParams,
    tools: Option<&[Value]>,
    response_format: Option<&Value>,
    tool_choice: Option<&Value>,
) -> Result<Prompt, DsEncodingError> {
    let messages = messages.to_vec();
    let mut normalized: Vec<Value> = Vec::new();
    for mut message in messages {
        if message["role"] == "developer" {
            if message["content"].as_str() == Some("")
                || message["content"].as_array().is_some_and(Vec::is_empty)
            {
                return Err(DsEncodingError::InvalidMessage {
                    role: "developer".into(),
                    msg: message.to_string(),
                });
            }
            message["role"] = Value::String("user".into());
        }
        match (normalized.last_mut(), text(&message, "role")) {
            (Some(last), "latest_reminder") if last["role"] == "latest_reminder" => {}
            _ => normalized.push(message),
        }
    }
    sort_recipe_tool_results(&mut normalized);
    let mut tools = tools.unwrap_or_default().to_vec();
    let mut required = tool_choice.is_some_and(|choice| choice == "required");
    if let Some(choice) = tool_choice {
        match text(choice, "type") {
            "function" => {
                let name = choice.get("function").unwrap_or(choice).get("name");
                tools.retain(|tool| tool.get("function").unwrap_or(tool).get("name") == name);
                required = true;
            }
            "allowed_tools" => {
                required = choice["mode"] == "required";
                tools.retain(|tool| {
                    choice["tools"].as_array().is_some_and(|allowed| {
                        allowed.iter().any(|allowed| {
                            allowed.get("name") == tool.get("function").unwrap_or(tool).get("name")
                        })
                    })
                });
            }
            _ => {}
        }
    }
    let renders_tools = !tools.is_empty() && tool_choice != Some(&Value::String("none".into()));
    let schema = response_format.and_then(|format| match text(format, "type") {
        "json_object" => Some(serde_json::json!({"type":"json_object"})),
        "json_schema" => format
            .get("json_schema")
            .and_then(|config| config.get("schema"))
            .cloned(),
        _ => None,
    });
    if (renders_tools || schema.is_some())
        && normalized
            .first()
            .is_none_or(|message| message["role"] != "system")
    {
        normalized.insert(0, serde_json::json!({"role":"system","content":""}));
    }
    let mut prompt = Prompt::default();
    if params.add_default_bos_token {
        prompt.push_special(BOS_TOKEN);
    }
    let thinking = thinking_mode == ThinkingMode::Thinking;
    let keep_history = renders_tools || !params.drop_thinking;
    for (index, message) in normalized.iter().enumerate() {
        let role = text(message, "role");
        if index == 0 && (thinking || role == "system") {
            prompt.push_special(SYSTEM_SP_TOKEN);
        }
        prompt.push_plain(render_reasoning_effort(
            index,
            thinking_mode,
            params.reasoning_effort,
        ));
        match role {
            "system" => {
                if index > 0 {
                    prompt.push_special(SYSTEM_SP_TOKEN);
                }
                push_content(&mut prompt, &message["content"])?;
                if let Some(tools) = message["tools"]
                    .as_array()
                    .filter(|tools| !tools.is_empty())
                {
                    prompt.push_plain("\n\n### Updated Tool Schemas\n\n");
                    prompt.push_plain(
                        normalized_tools(tools, false)
                            .iter()
                            .map(to_json)
                            .collect::<Vec<_>>()
                            .join("\n"),
                    );
                }
                if index == 0 {
                    if renders_tools {
                        prompt.push_plain("\n\n");
                        prompt.append(render_tools(&normalized_tools(&tools, true)));
                    }
                    if let Some(schema) = &schema {
                        prompt.push_plain("\n\n");
                        prompt.push_plain(render_response_format(schema));
                    }
                }
            }
            "user" | "tool" => {
                if index > 0 && matches!(text(&normalized[index - 1], "role"), "user" | "tool") {
                    prompt.push_plain("\n\n");
                } else {
                    prompt.push_special(USER_SP_TOKEN);
                }
                if role == "tool" {
                    prompt.push_plain("<tool_result>");
                }
                push_content(&mut prompt, &message["content"])?;
                if role == "tool" {
                    prompt.push_plain("</tool_result>");
                }
            }
            "latest_reminder" => {
                prompt.push_special(LATEST_REMINDER_SP_TOKEN);
                prompt.push_plain(text(message, "content"));
            }
            "assistant" => {
                let mut content = Prompt::default();
                push_content(&mut content, &message["content"])?;
                let content = content.into_parts().0;
                let prefix = message["prefix"] == true || message["wo_eos"] == true;
                let calls = message["tool_calls"]
                    .as_array()
                    .filter(|calls| !calls.is_empty());
                let reasoning = thinking && index > 0 && (keep_history || prefix);
                prompt.push_special(ASSISTANT_SP_TOKEN);
                prompt.push_special(if reasoning {
                    THINKING_START_TOKEN
                } else {
                    THINKING_END_TOKEN
                });
                if reasoning {
                    push_assistant_content(&mut prompt, text(message, "reasoning_content"));
                    if !(prefix && content.is_empty() && calls.is_none()) {
                        prompt.push_special(THINKING_END_TOKEN);
                    }
                }
                push_assistant_content(&mut prompt, &content);
                if let Some(calls) = calls {
                    prompt.append(tool_calls_block(&tool_calls_from_openai_format(calls)));
                }
                if !prefix {
                    prompt.push_special(EOS_TOKEN);
                }
            }
            other => return Err(DsEncodingError::UnknownRole(other.into())),
        }
    }
    let prefix = normalized.last().is_some_and(|message| {
        message["role"] == "assistant" && (message["prefix"] == true || message["wo_eos"] == true)
    });
    if !prefix {
        prompt.push_special(ASSISTANT_SP_TOKEN);
        prompt.push_special(if thinking {
            THINKING_START_TOKEN
        } else {
            THINKING_END_TOKEN
        });
    }
    if renders_tools && required {
        prompt.push_plain("\n\n<");
        prompt.push_special(DSML_TOKEN);
        prompt.push_plain(" calls>\n");
    }
    Ok(prompt)
}

fn sort_recipe_tool_results(messages: &mut [Value]) {
    let mut call_ids: Vec<String> = Vec::new();
    let mut index = 0;
    while index < messages.len() {
        if let Some(calls) = messages[index]["tool_calls"]
            .as_array()
            .filter(|calls| !calls.is_empty())
        {
            call_ids = calls
                .iter()
                .map(|call| text(call, "id").to_owned())
                .collect();
        }
        if matches!(text(&messages[index], "role"), "user" | "tool") {
            let start = index;
            while index < messages.len()
                && matches!(text(&messages[index], "role"), "user" | "tool")
            {
                index += 1;
            }
            let positions: Vec<_> = (start..index)
                .filter(|i| {
                    messages[*i]["role"] == "tool" && messages[*i]["tool_call_id"].is_string()
                })
                .collect();
            let mut results: Vec<_> = positions.iter().map(|i| messages[*i].clone()).collect();
            results.sort_by_key(|message| {
                call_ids
                    .iter()
                    .position(|id| id == text(message, "tool_call_id"))
                    .unwrap_or(0)
            });
            for (position, result) in positions.into_iter().zip(results) {
                messages[position] = result;
            }
        } else {
            index += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn recipe_prompts_and_special_token_ranges_match_the_pinned_oracle() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/deepseek_v41/recipe.json"
        ))
        .unwrap();
        assert_eq!(
            fixture["revision"],
            "935817c66f5d6e64838f419692545f06787e485f"
        );
        for case in fixture["cases"].as_array().unwrap() {
            let params = EncodeParams {
                reasoning_effort: parse_reasoning_effort(&case["effort"]).unwrap(),
                ..Default::default()
            };
            let mode = if case["thinking"] == false {
                ThinkingMode::Chat
            } else {
                ThinkingMode::Thinking
            };
            let prompt = render_conversation(
                case["messages"].as_array().unwrap(),
                mode,
                params,
                case["tools"].as_array().map(Vec::as_slice),
                case.get("response_format"),
                case.get("tool_choice"),
            )
            .unwrap();
            let (text, ranges) = prompt.into_parts();
            assert_eq!(text, case["text"].as_str().unwrap(), "{}", case["name"]);
            assert_eq!(
                serde_json::to_value(ranges).unwrap(),
                case["special_ranges"],
                "{}",
                case["name"]
            );
        }
    }

    #[test]
    fn numeric_budgets_remain_an_explicit_extension() {
        for budget in [1, 42, 100] {
            let params = EncodeParams {
                reasoning_effort: parse_reasoning_effort(&json!(budget)).unwrap(),
                ..Default::default()
            };
            let text = encode_messages(
                &[json!({"role":"user","content":"q"})],
                ThinkingMode::Thinking,
                &params,
            )
            .unwrap();
            assert!(text.contains(&format!("Reasoning Effort: {budget} (range")));
        }
        for invalid in [
            json!(0),
            json!(101),
            json!(1.5),
            json!(true),
            json!("bogus"),
        ] {
            assert!(parse_reasoning_effort(&invalid).is_err());
        }
    }

    #[test]
    fn literal_image_tokens_are_plain_and_unknown_roles_are_rejected() {
        let prompt = render_messages(
            &[json!({"role":"user","content":IMAGE_PLACEHOLDER})],
            ThinkingMode::Chat,
            &EncodeParams::default(),
        )
        .unwrap();
        let (text, ranges) = prompt.into_parts();
        assert!(text.contains(IMAGE_PLACEHOLDER));
        assert!(!ranges
            .iter()
            .any(|range| &text[range.clone()] == IMAGE_PLACEHOLDER));
        assert!(encode_messages(
            &[json!({"role":"moderator","content":"x"})],
            ThinkingMode::Chat,
            &EncodeParams::default()
        )
        .is_err());
        assert!(encode_messages(
            &[json!({"role":"user","content":[{"type":"video"}]})],
            ThinkingMode::Chat,
            &EncodeParams::default()
        )
        .is_err());
    }
}
