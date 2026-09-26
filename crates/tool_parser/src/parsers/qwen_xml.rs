use std::borrow::Cow;

use async_trait::async_trait;
use openai_protocol::common::Tool;
use regex::Regex;
use serde_json::Value;

use crate::{
    errors::{ParserError, ParserResult},
    parsers::helpers,
    traits::ToolParser,
    types::{FunctionCall, StreamingParseResult, ToolCall, ToolCallItem},
};

/// Qwen XML format parser for tool calls
///
/// Handles the Qwen XML specific XML format:
/// `<tool_call>\n<function=name>\n<parameter=key>value</parameter>\n</function>\n</tool_call>`
///
/// Features:
/// - Tool Call Tags: `<tool_call>` and `</tool_call>` wrap each individual call
/// - XML-style function declaration: `<function=name>`
/// - XML-style parameters: `<parameter=key>value</parameter>`
///
/// Reference: https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8?chat_template=default
pub struct QwenXmlParser {
    /// Regex for extracting tool calls in parse_complete
    extractor: Regex,

    /// Buffer for accumulating incomplete patterns across chunks
    buffer: String,

    /// Stores complete tool call info (name and arguments) for each tool being parsed
    prev_tool_call_arr: Vec<Value>,

    /// Index of currently streaming tool call (-1 means no active tool)
    current_tool_id: i32,

    /// Flag for whether current tool's name has been sent to client
    current_tool_name_sent: bool,

    /// Tracks raw JSON string content streamed to client for each tool's arguments
    streamed_args_for_tool: Vec<String>,

    /// Token configuration
    tool_call_start_token: &'static str,
    tool_call_end_token: &'static str,

    /// XML format streaming state
    in_tool_call: bool,
    awaiting_wrapper_close: bool,
    current_function_name: String,
    current_parameters: serde_json::Map<String, Value>,

    /// Precompiled regex patterns for XML format parsing
    xml_function_pattern: Regex,
    xml_param_pattern: Regex,
}

fn next_unclosed_xml_marker(tail: &str) -> Option<(&'static str, usize)> {
    [
        "<parameter=",
        "</parameter>",
        "</function>",
        "<function=",
        "</tool_call>",
        "<tool_call>",
    ]
    .into_iter()
    .filter_map(|marker| tail.find(marker).map(|idx| (marker, idx)))
    .min_by_key(|(_, idx)| *idx)
}

/// Close missing XML fences at the next parameter, function or wrapper
/// boundary. At EOF an unfinished value is held rather than invented.
fn normalize_lenient_xml_block(block: &str) -> Cow<'_, str> {
    let mut out = String::with_capacity(block.len());
    let mut rest = block;
    let mut changed = false;
    let mut in_tool_call = false;
    let mut in_function = false;
    let mut in_parameter = false;

    while !rest.is_empty() {
        let Some((marker, idx)) = next_unclosed_xml_marker(rest) else {
            out.push_str(rest);
            break;
        };

        out.push_str(&rest[..idx]);
        rest = &rest[idx..];

        match marker {
            "<parameter=" => {
                if in_parameter {
                    out.push_str("</parameter>");
                    changed = true;
                }
                in_parameter = true;
            }
            "</parameter>" => in_parameter = false,
            "</function>" => {
                if in_parameter {
                    out.push_str("</parameter>");
                    in_parameter = false;
                    changed = true;
                }
                in_function = false;
            }
            "<function=" => {
                if in_parameter {
                    out.push_str("</parameter>");
                    in_parameter = false;
                    changed = true;
                }
                if in_function {
                    out.push_str("</function>");
                    changed = true;
                }
                in_tool_call = true;
                in_function = true;
            }
            "</tool_call>" | "<tool_call>" => {
                if in_parameter {
                    out.push_str("</parameter>");
                    in_parameter = false;
                    changed = true;
                }
                if in_function {
                    out.push_str("</function>");
                    in_function = false;
                    changed = true;
                }
                if marker == "<tool_call>" && in_tool_call {
                    out.push_str("</tool_call>");
                    changed = true;
                }
                in_tool_call = marker == "<tool_call>";
            }
            _ => {}
        }
        out.push_str(marker);
        rest = &rest[marker.len()..];
    }

    if changed {
        Cow::Owned(out)
    } else {
        Cow::Borrowed(block)
    }
}

/// Strip at most one structural line break at each edge of an XML value.
fn strip_structural_line_breaks(raw: &str) -> &str {
    let raw = raw
        .strip_prefix("\r\n")
        .or_else(|| raw.strip_prefix('\n'))
        .unwrap_or(raw);
    raw.strip_suffix("\r\n")
        .or_else(|| raw.strip_suffix('\n'))
        .unwrap_or(raw)
}

/// Parse a raw parameter value, similar to Python's `_safe_val`.
///
/// Argument values are treated **literally** — no HTML-entity decoding. This
/// matches Qwen's own official API (DashScope), verified for both Qwen3-Coder
/// and Qwen3.5: a tool argument whose value contains `&amp;`, `&lt;`, `&#39;`
/// is returned with those entities intact. The Qwen XML tool format is not
/// HTML-escaped on render either (the chat template emits values via
/// `| tojson | safe` / `| string`), so parsing must not unescape it. vLLM's
/// `Qwen3CoderToolParser`, SGLang's `qwen3_coder_detector`, and Qwen-Agent all
/// agree, passing argument values through verbatim.
///
/// 1. Try to parse as JSON (numbers, booleans, null, objects, arrays)
/// 2. Fall back to string if JSON parsing fails
fn safe_val(raw: &str) -> Value {
    let trimmed = raw.trim();

    // Try JSON parsing first
    if let Ok(v) = serde_json::from_str::<Value>(trimmed) {
        return v;
    }

    // Handle Python-style literals (True, False, None)
    match trimmed {
        "True" => return Value::Bool(true),
        "False" => return Value::Bool(false),
        "None" => return Value::Null,
        _ => {}
    }

    // Fall back to string
    Value::String(raw.to_string())
}

/// Coerce an XML parameter value by its declared schema type, falling back to
/// [`safe_val`] inference when the type is unknown.
///
/// Values are treated literally; see [`safe_val`] for why the format is not
/// HTML-unescaped.
fn coerce_value(raw: &str, declared_type: Option<&str>) -> Value {
    let raw = strip_structural_line_breaks(raw);
    helpers::coerce_by_schema_type(raw, declared_type).unwrap_or_else(|| safe_val(raw))
}

impl QwenXmlParser {
    /// Constrain the native Qwen3 Coder XML format using xgrammar's schema style.
    pub fn build_structural_tag(tools: &[Tool], at_least_one: bool) -> Value {
        let tags: Vec<Value> = tools
            .iter()
            .filter(|tool| !tool.function.name.is_empty())
            .map(|tool| {
                serde_json::json!({
                    "type": "tag",
                    "begin": format!("<tool_call>\n<function={}>\n", tool.function.name),
                    "content": {
                        "type": "json_schema",
                        "json_schema": tool.function.parameters,
                        "style": "qwen_xml",
                    },
                    "end": "\n</function>\n</tool_call>",
                })
            })
            .collect();
        serde_json::json!({
            "type": "structural_tag",
            "format": {
                "type": "triggered_tags",
                "triggers": ["<tool_call>\n<function="],
                "tags": tags,
                "at_least_one": at_least_one,
            }
        })
    }

    /// Allow a prompt-prefilled thinking block to finish before a forced call.
    pub fn reasoning_prefix() -> Value {
        serde_json::json!({
            "type": "tag",
            "begin": "",
            "content": {"type": "any_text", "excludes": ["<think>", "</think>", "<tool_call>"]},
            "end": "</think>",
        })
    }

    /// Create a new Qwen XML parser
    #[expect(
        clippy::expect_used,
        reason = "regex patterns are compile-time string literals"
    )]
    pub fn new() -> Self {
        // Support XML format: <tool_call>\n<function=name>\n<parameter=key>value</parameter>\n</function>\n</tool_call>
        let pattern = r"(?s)<function=[^>]+>.*?</function>";
        let extractor = Regex::new(pattern).expect("Valid regex pattern");

        // Precompile XML format regex patterns for performance
        let xml_function_pattern =
            Regex::new(r"<function=([^>]+)>").expect("Valid XML function pattern");
        let xml_param_pattern = Regex::new(r"(?s)<parameter=([^>]+)>(.*?)</parameter>")
            .expect("Valid XML parameter pattern");

        Self {
            extractor,
            buffer: String::new(),
            prev_tool_call_arr: Vec::new(),
            current_tool_id: -1,
            current_tool_name_sent: false,
            streamed_args_for_tool: Vec::new(),
            tool_call_start_token: "<tool_call>",
            tool_call_end_token: "</tool_call>",
            in_tool_call: false,
            awaiting_wrapper_close: false,
            current_function_name: String::new(),
            current_parameters: serde_json::Map::new(),
            xml_function_pattern,
            xml_param_pattern,
        }
    }

    /// Parse XML format tool call: <function=name><parameter=key>value</parameter></function>
    fn parse_xml_format(&self, content: &str, tools: &[Tool]) -> ParserResult<Option<ToolCall>> {
        let function_captures = self
            .xml_function_pattern
            .captures(content)
            .ok_or_else(|| ParserError::ParsingFailed("No function name found".to_string()))?;

        let function_name = function_captures
            .get(1)
            .ok_or_else(|| ParserError::ParsingFailed("Function name capture failed".to_string()))?
            .as_str()
            .trim()
            .to_string();

        if function_name.is_empty() {
            return Ok(None);
        }

        let param_types = helpers::param_types_for_function(tools, &function_name);
        let mut parameters = serde_json::Map::new();

        for cap in self.xml_param_pattern.captures_iter(content) {
            if let (Some(key_match), Some(value_match)) = (cap.get(1), cap.get(2)) {
                let key = key_match.as_str().trim().to_string();
                let value = value_match.as_str();
                let json_value = coerce_value(value, param_types.get(&key).map(String::as_str));
                parameters.insert(key, json_value);
            }
        }

        let arguments = serde_json::to_string(&parameters)
            .map_err(|e| ParserError::ParsingFailed(e.to_string()))?;

        Ok(Some(ToolCall {
            function: FunctionCall {
                name: function_name,
                arguments,
            },
        }))
    }

    /// Parse and stream complete parameters from buffer
    /// Returns tool call items to emit (similar to Python's _parse_and_stream_parameters)
    fn parse_and_stream_parameters(
        &mut self,
        tools: &[Tool],
        parameter_end: usize,
    ) -> Vec<ToolCallItem> {
        let mut calls: Vec<ToolCallItem> = vec![];
        let param_types = helpers::param_types_for_function(tools, &self.current_function_name);

        // Leave parameters from subsequent coalesced calls for their own iteration.
        let mut new_params = serde_json::Map::new();
        for cap in self
            .xml_param_pattern
            .captures_iter(&self.buffer[..parameter_end])
        {
            if let (Some(key_match), Some(value_match)) = (cap.get(1), cap.get(2)) {
                let key = key_match.as_str().trim().to_string();
                let value = value_match.as_str();
                let json_value = coerce_value(value, param_types.get(&key).map(String::as_str));
                new_params.insert(key, json_value);
            }
        }

        // Calculate parameter diff and stream updates
        if new_params != self.current_parameters {
            let current_args = &mut self.streamed_args_for_tool[self.current_tool_id as usize];

            if self.current_parameters.is_empty() {
                // First parameter(s) - build JSON fragment (without closing brace)
                let mut items = Vec::new();
                for (key, value) in &new_params {
                    let key_json =
                        serde_json::to_string(key).unwrap_or_else(|_| format!("\"{key}\""));
                    let value_json = serde_json::to_string(value).unwrap_or_default();
                    items.push(format!("{key_json}: {value_json}"));
                }
                let json_fragment = format!("{{{}", items.join(", "));

                calls.push(ToolCallItem {
                    tool_index: self.current_tool_id as usize,
                    name: None,
                    parameters: json_fragment.clone(),
                });
                *current_args = json_fragment;
            } else {
                // Additional parameters - add them incrementally
                let new_keys: Vec<_> = new_params
                    .keys()
                    .filter(|k| !self.current_parameters.contains_key(*k))
                    .collect();

                if !new_keys.is_empty() {
                    let mut continuation_parts = Vec::new();
                    for key in new_keys {
                        if let Some(value) = new_params.get(key) {
                            let key_json =
                                serde_json::to_string(key).unwrap_or_else(|_| format!("\"{key}\""));
                            let value_json = serde_json::to_string(value).unwrap_or_default();
                            continuation_parts.push(format!("{key_json}: {value_json}"));
                        }
                    }

                    let json_fragment = format!(", {}", continuation_parts.join(", "));

                    calls.push(ToolCallItem {
                        tool_index: self.current_tool_id as usize,
                        name: None,
                        parameters: json_fragment.clone(),
                    });
                    current_args.push_str(&json_fragment);
                }
            }

            // Update current state
            self.current_parameters.clone_from(&new_params);
            if let Some(tool_obj) =
                self.prev_tool_call_arr[self.current_tool_id as usize].as_object_mut()
            {
                tool_obj.insert("arguments".to_string(), Value::Object(new_params));
            }
        }

        calls
    }

    /// Shared non-streaming parse, schema-aware when `tools` are provided.
    fn parse_complete_inner(
        &self,
        text: &str,
        tools: &[Tool],
    ) -> ParserResult<(String, Vec<ToolCall>)> {
        // Check if text contains Qwen XML format
        if !self.has_tool_markers(text) {
            return Ok((text.to_string(), vec![]));
        }

        // Find where the first tool call begins
        // Safe: has_tool_markers() already confirmed the marker exists
        let idx = text
            .find(self.tool_call_start_token)
            .into_iter()
            .chain(text.find("<function="))
            .min()
            .ok_or_else(|| ParserError::ParsingFailed("tool call marker not found".to_string()))?;
        let normal_text = text[..idx].to_string();

        // Recover missing inner fences and complete bare functions.
        let normalized = normalize_lenient_xml_block(text);
        let mut parsed = Vec::new();
        for captures in self.extractor.captures_iter(&normalized) {
            if let Some(content_str) = captures.get(0) {
                let content = content_str.as_str().trim();

                match self.parse_xml_format(content, tools) {
                    Ok(Some(tool)) => parsed.push(tool),
                    Ok(None) => continue,
                    Err(e) => {
                        tracing::warn!("Failed to parse XML tool call: {:?}", e);
                        continue;
                    }
                }
            }
        }

        // If no tools were successfully parsed despite having markers, return entire text
        if parsed.is_empty() {
            return Ok((text.to_string(), vec![]));
        }

        Ok((normal_text, parsed))
    }

    /// Reset streaming state for next tool call
    fn reset_streaming_state(&mut self) {
        self.in_tool_call = false;
        self.current_tool_name_sent = false;
        self.current_function_name.clear();
        self.current_parameters.clear();
    }
}

impl Default for QwenXmlParser {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ToolParser for QwenXmlParser {
    async fn parse_complete(&self, text: &str) -> ParserResult<(String, Vec<ToolCall>)> {
        self.parse_complete_inner(text, &[])
    }

    async fn parse_complete_with_tools(
        &self,
        text: &str,
        tools: &[Tool],
    ) -> ParserResult<(String, Vec<ToolCall>)> {
        self.parse_complete_inner(text, tools)
    }

    async fn parse_incremental(
        &mut self,
        chunk: &str,
        tools: &[Tool],
    ) -> ParserResult<StreamingParseResult> {
        self.buffer.push_str(chunk);
        self.buffer = normalize_lenient_xml_block(&self.buffer).into_owned();

        let mut normal_text = String::new();
        let mut calls: Vec<ToolCallItem> = vec![];

        // Build tool indices for validation
        let tool_indices = helpers::get_tool_indices(tools);

        loop {
            // A complete bare function is recoverable without an outer wrapper.
            if !self.in_tool_call {
                if self.awaiting_wrapper_close {
                    let tail = self.buffer.trim_start();
                    if tail.starts_with(self.tool_call_end_token) {
                        let end = self.buffer.len() - tail.len() + self.tool_call_end_token.len();
                        self.buffer.drain(..end);
                        self.awaiting_wrapper_close = false;
                    } else if [
                        self.tool_call_end_token,
                        "<function=",
                        self.tool_call_start_token,
                    ]
                    .into_iter()
                    .any(|marker| marker.starts_with(tail))
                    {
                        break;
                    } else if tail.starts_with("<function=")
                        || tail.starts_with(self.tool_call_start_token)
                    {
                        let whitespace = self.buffer.len() - tail.len();
                        self.buffer.drain(..whitespace);
                        self.awaiting_wrapper_close = false;
                    } else {
                        self.awaiting_wrapper_close = false;
                    }
                }
                let start = self
                    .buffer
                    .find(self.tool_call_start_token)
                    .map(|index| (index, self.tool_call_start_token.len()))
                    .into_iter()
                    .chain(self.buffer.find("<function=").map(|index| (index, 0)))
                    .min_by_key(|(index, _)| *index);
                if let Some((index, marker_len)) = start {
                    let prefix = &self.buffer[..index];
                    if self.current_tool_id >= 0 {
                        normal_text.push_str(&prefix.replace(self.tool_call_end_token, ""));
                    } else {
                        normal_text.push_str(prefix);
                    }
                    self.buffer.drain(..index + marker_len);
                    self.in_tool_call = true;
                    continue;
                }
                // Hold partial openers/closers across token and transport splits.
                let keep_from = [
                    self.tool_call_start_token,
                    "<function=",
                    self.tool_call_end_token,
                ]
                .into_iter()
                .filter_map(|token| helpers::ends_with_partial_token(&self.buffer, token))
                .map(|suffix| self.buffer.len() - suffix)
                .min()
                .unwrap_or(self.buffer.len());
                let prefix = &self.buffer[..keep_from];
                if self.current_tool_id >= 0 {
                    normal_text.push_str(&prefix.replace(self.tool_call_end_token, ""));
                } else {
                    normal_text.push_str(prefix);
                }
                self.buffer.drain(..keep_from);
                break;
            }

            // We're in a tool call, try to parse function name if not sent yet
            if !self.current_tool_name_sent {
                if let Some(captures) = self.xml_function_pattern.captures(&self.buffer) {
                    if let Some(name_match) = captures.get(1) {
                        let function_name = name_match.as_str().trim().to_string();

                        // Validate function name
                        if tool_indices.contains_key(&function_name) {
                            self.current_function_name.clone_from(&function_name);
                            self.current_tool_name_sent = true;

                            // Initialize tool call tracking
                            if self.current_tool_id == -1 {
                                self.current_tool_id = 0;
                            }

                            // Ensure tracking arrays are large enough
                            helpers::ensure_capacity(
                                self.current_tool_id,
                                &mut self.prev_tool_call_arr,
                                &mut self.streamed_args_for_tool,
                            );

                            // Store tool call info
                            self.prev_tool_call_arr[self.current_tool_id as usize] = serde_json::json!({
                                "name": function_name,
                                "arguments": {}
                            });

                            // Send tool name
                            calls.push(ToolCallItem {
                                tool_index: self.current_tool_id as usize,
                                name: Some(function_name),
                                parameters: String::new(),
                            });

                            // Keep the header so later boundary normalization can close
                            // this function before a following function or wrapper.
                            continue;
                        } else {
                            // Invalid function name, reset state
                            tracing::warn!("Invalid function name: {}", function_name);
                            self.reset_streaming_state();
                            normal_text.push_str(&self.buffer);
                            self.buffer.clear();
                            break;
                        }
                    }
                } else {
                    // Function name not complete yet, wait for more text
                    break;
                }
            }

            // Parse parameters (only complete ones)
            if self.current_tool_name_sent {
                let end_pos = self
                    .buffer
                    .find("</function>")
                    .map(|index| (index, "</function>".len()))
                    .into_iter()
                    .chain(
                        self.buffer
                            .find(self.tool_call_end_token)
                            .map(|index| (index, self.tool_call_end_token.len())),
                    )
                    .min_by_key(|(index, _)| *index);
                let parameter_end = end_pos.map_or(self.buffer.len(), |(index, _)| index);
                let param_calls = self.parse_and_stream_parameters(tools, parameter_end);
                calls.extend(param_calls);

                // Check if tool call is complete
                if let Some((end_pos, marker_len)) = end_pos {
                    // Parameter fragments leave the root open; braces in values are data.
                    let current_args =
                        &mut self.streamed_args_for_tool[self.current_tool_id as usize];
                    let closing = if current_args.is_empty() { "{}" } else { "}" };
                    calls.push(ToolCallItem {
                        tool_index: self.current_tool_id as usize,
                        name: None,
                        parameters: closing.to_string(),
                    });
                    current_args.push_str(closing);

                    // Complete the tool call
                    self.buffer = self.buffer[end_pos + marker_len..].to_string();
                    self.reset_streaming_state();
                    self.awaiting_wrapper_close = marker_len == "</function>".len();
                    self.current_tool_id += 1;
                    continue;
                } else {
                    // Tool call not complete yet, wait for more text
                    break;
                }
            }

            break;
        }

        Ok(StreamingParseResult { normal_text, calls })
    }

    fn has_tool_markers(&self, text: &str) -> bool {
        text.contains(self.tool_call_start_token) || text.contains("<function=")
    }

    fn get_unstreamed_tool_args(&self) -> Option<Vec<ToolCallItem>> {
        let tool_index = self.prev_tool_call_arr.len().checked_sub(1)?;
        let actual = self.streamed_args_for_tool.get(tool_index)?;
        let expected = self.prev_tool_call_arr[tool_index].get("arguments")?;
        // XML parameters use spaced JSON, unlike the generic compact-prefix recovery.
        // Append only the root brace, and only for exactly the already parsed values.
        if !actual.is_empty()
            && serde_json::from_str::<Value>(&format!("{actual}}}"))
                .ok()
                .as_ref()
                == Some(expected)
        {
            return Some(vec![ToolCallItem {
                tool_index,
                name: None,
                parameters: "}".to_string(),
            }]);
        }
        helpers::get_unstreamed_args(&self.prev_tool_call_arr, &self.streamed_args_for_tool)
    }

    fn reset(&mut self) {
        helpers::reset_parser_state(
            &mut self.buffer,
            &mut self.prev_tool_call_arr,
            &mut self.current_tool_id,
            &mut self.current_tool_name_sent,
            &mut self.streamed_args_for_tool,
        );
        self.reset_streaming_state();
        self.awaiting_wrapper_close = false;
    }
}

#[cfg(test)]
mod tests {
    use openai_protocol::common::Function;

    use super::*;

    #[test]
    fn test_safe_val_json() {
        assert_eq!(safe_val("42"), Value::Number(42.into()));
        assert_eq!(safe_val("1.5"), serde_json::json!(1.5));
        assert_eq!(safe_val("true"), Value::Bool(true));
        assert_eq!(safe_val("false"), Value::Bool(false));
        assert_eq!(safe_val("null"), Value::Null);
        assert_eq!(
            safe_val(r#"{"key": "value"}"#),
            serde_json::json!({"key": "value"})
        );
        assert_eq!(safe_val(r"[1, 2, 3]"), serde_json::json!([1, 2, 3]));
    }

    #[test]
    fn test_safe_val_python_literals() {
        assert_eq!(safe_val("True"), Value::Bool(true));
        assert_eq!(safe_val("False"), Value::Bool(false));
        assert_eq!(safe_val("None"), Value::Null);
    }

    #[test]
    fn test_safe_val_string_fallback() {
        assert_eq!(
            safe_val("hello world"),
            Value::String("hello world".to_string())
        );
        assert_eq!(
            safe_val("  spaces  "),
            Value::String("  spaces  ".to_string())
        );
    }

    // Values are treated literally: entity-like substrings must NOT be decoded
    // (parity with Qwen's official API, which returns them intact).
    #[test]
    fn test_safe_val_preserves_html_entities() {
        assert_eq!(
            safe_val("&lt;div&gt;"),
            Value::String("&lt;div&gt;".to_string())
        );
        assert_eq!(
            safe_val("Tom &amp; Jerry"),
            Value::String("Tom &amp; Jerry".to_string())
        );
        // Numeric/hex entities are likewise left untouched.
        assert_eq!(safe_val("it&#39;s"), Value::String("it&#39;s".to_string()));
        assert_eq!(safe_val("&#x3C;"), Value::String("&#x3C;".to_string()));
    }

    fn tool_with_props(props: Value) -> Vec<Tool> {
        vec![Tool {
            tool_type: "function".to_string(),
            function: Function {
                name: "f".to_string(),
                description: None,
                parameters: serde_json::json!({"type": "object", "properties": props}),
                strict: None,
            },
        }]
    }

    // String-typed params stay strings even when they look numeric/bool/array/object.
    #[tokio::test]
    async fn test_schema_aware_coercion_keeps_strings() {
        let tools = tool_with_props(serde_json::json!({
            "limit": {"type": "string"},
            "flag": {"type": "string"},
            "coords": {"type": "string"},
            "cfg": {"type": "string"},
            "count": {"type": "integer"},
        }));
        let text = "<tool_call>\n<function=f>\n\
            <parameter=limit>4</parameter>\n\
            <parameter=flag>true</parameter>\n\
            <parameter=coords>[60,30]</parameter>\n\
            <parameter=cfg>{\"a\": 1}</parameter>\n\
            <parameter=count>5</parameter>\n\
            </function>\n</tool_call>";
        let (_, calls) = QwenXmlParser::new()
            .parse_complete_with_tools(text, &tools)
            .await
            .unwrap();
        assert_eq!(calls.len(), 1);
        let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["limit"], Value::String("4".to_string()));
        assert_eq!(args["flag"], Value::String("true".to_string()));
        assert_eq!(args["coords"], Value::String("[60,30]".to_string()));
        assert_eq!(args["cfg"], Value::String("{\"a\": 1}".to_string()));
        assert_eq!(args["count"], Value::Number(5.into()));
    }

    // The streaming path threads `tools` separately, so cover it too.
    #[tokio::test]
    async fn test_streaming_schema_aware_coercion() {
        let tools = tool_with_props(serde_json::json!({
            "limit": {"type": "string"},
            "count": {"type": "integer"},
        }));
        let text = "<tool_call>\n<function=f>\n\
            <parameter=limit>4</parameter>\n\
            <parameter=count>5</parameter>\n\
            </function>\n</tool_call>";
        let result = QwenXmlParser::new()
            .parse_incremental(text, &tools)
            .await
            .unwrap();
        let args: String = result.calls.iter().map(|c| c.parameters.as_str()).collect();
        assert!(
            args.contains(r#""limit": "4""#),
            "string param must stay string: {args}"
        );
        assert!(
            args.contains(r#""count": 5"#),
            "int param must coerce: {args}"
        );
    }

    // Golden conformance test (regression guard for #1888): a tool argument
    // whose value contains HTML entities must round-trip UNCHANGED, matching
    // Qwen's official API (verified on Qwen3-Coder and Qwen3.5). Covers both the
    // schema-typed `string` path and the schema-less inference fallback.
    #[tokio::test]
    async fn test_arg_values_with_entities_roundtrip_unchanged() {
        let literal = "<a>Tom &amp; Jerry</a> &lt;x&gt; it&#39;s";

        // Function name matches `tool_with_props` (`f`) so the schema-typed
        // branch below actually resolves the `content` param's type.
        let text = format!(
            "<tool_call>\n<function=f>\n\
             <parameter=content>\n{literal}\n</parameter>\n\
             </function>\n</tool_call>"
        );

        // Schema-less inference path (`safe_val`): not valid JSON -> stays a
        // literal string with entities intact.
        let (_, calls) = QwenXmlParser::new().parse_complete(&text).await.unwrap();
        assert_eq!(calls.len(), 1);
        let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["content"], Value::String(literal.to_string()));

        // Schema-typed `string` path (`coerce_value` -> `coerce_by_schema_type`).
        let tools = tool_with_props(serde_json::json!({
            "content": {"type": "string"},
        }));
        let (_, calls) = QwenXmlParser::new()
            .parse_complete_with_tools(&text, &tools)
            .await
            .unwrap();
        let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["content"], Value::String(literal.to_string()));
    }
}
