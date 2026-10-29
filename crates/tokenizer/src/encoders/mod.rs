//! The chat renderers with an encoder of their own, and what each answers
//! for the template questions the gateway asks around a render.

use std::{collections::HashMap, ops::Range};

use serde_json::Value;

use crate::{
    chat_template::{ChatTemplateContentFormat, ThinkingKeyName, ThinkingToggle},
    traits::RendererCapabilities,
};

mod deepseek_common;
pub mod deepseek_v32;
pub mod deepseek_v4;
pub mod deepseek_v41;
pub mod kimi_k25_tools;
pub mod kimi_k3_xtml;

/// Prompt text with the byte ranges that are explicit added tokens. Everything outside those
/// ranges is encoded without the added vocabulary, including text taken from the request.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Prompt {
    text: String,
    special_ranges: Vec<Range<usize>>,
}

impl Prompt {
    pub(crate) fn push_plain(&mut self, text: impl AsRef<str>) {
        self.text.push_str(text.as_ref());
    }

    pub(crate) fn push_special(&mut self, token: &str) {
        let start = self.text.len();
        self.text.push_str(token);
        self.special_ranges.push(start..self.text.len());
    }

    pub(crate) fn append(&mut self, other: Self) {
        let offset = self.text.len();
        self.text.push_str(&other.text);
        self.special_ranges.extend(
            other
                .special_ranges
                .into_iter()
                .map(|range| range.start + offset..range.end + offset),
        );
    }

    pub(crate) fn into_parts(self) -> (String, Vec<Range<usize>>) {
        (self.text, self.special_ranges)
    }
}

/// A renderer's own rule for the thinking mode a request renders with, from
/// the template kwargs and the request's `thinking` field.
pub(crate) type ThinkingModeRule = fn(Option<&HashMap<String, Value>>, Option<bool>) -> bool;

/// What a renderer with its own encoder answers for the questions the
/// gateway asks a tokenizer around a render (the `Tokenizer` trait's
/// `chat_template_content_format`, `thinking_toggle`, `thinking_key_name`,
/// the `reasoning_effort` words, `think_in_prefill`, `renderer_capabilities`,
/// `native_thinking_mode`): one table per renderer, kept beside it. The Jinja
/// renderer answers them from the template instead
/// ([`crate::chat_template::ChatTemplateState`]), so a wrapper reads the
/// table when a native renderer applies and asks the template otherwise.
#[derive(Debug, Clone, Copy)]
pub(crate) struct RendererProfile {
    /// The renderer's own content format; `None` for a renderer with no
    /// opinion on content parts, which reports the template's (usually
    /// absent, so the default).
    pub(crate) content_format: Option<ChatTemplateContentFormat>,
    /// The thinking switch and its default. The Jinja processor knows
    /// nothing of a native renderer, so the renderer states it.
    pub(crate) thinking_toggle: ThinkingToggle,
    /// The kwarg the switch reads.
    pub(crate) thinking_key_name: ThinkingKeyName,
    /// `chat_template_kwargs.reasoning_effort` values that switch thinking
    /// on, so the gateway's parser is armed from the kwarg as the prompt is;
    /// empty when the renderer takes only the protocol's own words.
    pub(crate) native_effort_values: &'static [&'static str],
    /// The values that switch it off; empty when the renderer switches off
    /// on the protocol's `none` / `minimal` and declares no words of its own.
    pub(crate) native_effort_off_values: &'static [&'static str],
    /// The prompt ends inside an open think block when thinking is on, so a
    /// completion starts mid-reasoning and the parser must be told. Every
    /// native renderer reports a thinking switch, so the switchless-template
    /// question `prefill_opens_think_block` mirrors this.
    pub(crate) think_in_prefill: bool,
    /// The behaviours the gateway mirrors when it prepares the request.
    pub(crate) capabilities: RendererCapabilities,
    /// The renderer's own rule for the thinking mode a request renders with,
    /// when the parser must be armed from the same answer; `None` leaves the
    /// mode to the gateway's reading of the kwargs.
    pub(crate) thinking_mode: Option<ThinkingModeRule>,
}

impl RendererProfile {
    /// The behaviours no native renderer has unless its table says so.
    pub(crate) const NO_CAPABILITIES: RendererCapabilities = RendererCapabilities {
        enable_thinking_alias: false,
        native_assistant_continuation: false,
        raw_tool_call_arguments: false,
        declares_tools_under_tool_choice_none: false,
    };

    /// The behaviours of a renderer without a profile: the Jinja template,
    /// which continues a trailing assistant message the way transformers'
    /// `continue_final_message` does.
    pub(crate) const TEMPLATE_CAPABILITIES: RendererCapabilities = RendererCapabilities {
        native_assistant_continuation: true,
        ..Self::NO_CAPABILITIES
    };
}
