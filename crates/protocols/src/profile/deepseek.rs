//! DeepSeek frontend controls: Dynamo V4 opt-in thinking and the V4.1 recipe.
//! Explicit thinking controls and normalized effort are shared by rendering and parsing.
//!
//! Older V3/R1 models and opaque aliases retain the baseline. This profile
//! does not claim the vendor's context window, sampling behavior or JSON
//! constraints for a self-hosted worker, nor enforce documented history and
//! identifier restrictions the official API did not enforce in calibration.

use serde_json::Value;

use crate::{
    chat::{ChatCompletionRequest, ThinkingType},
    common::{ToolChoice, ToolChoiceValue},
};

/// Whether a model uses the V4.1 recipe, including its public Flash alias.
pub fn is_v41_model(model: &str) -> bool {
    model.split('/').any(|part| {
        let part = part.to_ascii_lowercase();
        [
            "deepseek-v4.1",
            "deepseek_v4.1",
            "deepseek-v41",
            "deepseek_v41",
            "deepseekv41",
            "deepseek_v4_1",
            "deepseek-flash",
            "deepseek_flash",
        ]
        .iter()
        .any(|prefix| {
            part.strip_prefix(prefix)
                .is_some_and(|tail| tail.is_empty() || tail.starts_with(['-', '_']))
        })
    })
}

pub(super) fn matches_model(segment: &str) -> bool {
    is_v41_model(segment)
        || ["deepseek-v4", "deepseek_v4", "deepseekv4"]
            .iter()
            .any(|prefix| {
                segment
                    .to_ascii_lowercase()
                    .strip_prefix(prefix)
                    .is_some_and(|tail| tail.is_empty() || tail.starts_with(['-', '_']))
            })
}

/// The effort names passed to native renderers. V4.1 follows deepseek-recipe;
/// numeric 1..=100 budgets remain a supported SMG extension.
pub fn normalize_effort<'a>(model: &str, effort: &'a str) -> &'a str {
    match effort {
        "minimal" if is_v41_model(model) => "low",
        "minimal" => "none",
        "medium" => "high",
        "xhigh" if !is_v41_model(model) => "max",
        "ultra" => "max",
        _ => effort,
    }
}

/// Resolve protocol controls without template overrides (Messages and counts).
pub fn thinking_enabled(model: &str, explicit: Option<bool>, effort: Option<&str>) -> bool {
    if is_v41_model(model) {
        explicit.unwrap_or_else(|| effort != Some("none"))
    } else {
        explicit != Some(false)
            && !matches!(effort, Some("none" | "minimal"))
            && (explicit == Some(true) || effort.is_some())
    }
}

pub(super) fn normalize_chat(req: &mut ChatCompletionRequest) {
    // Invalid override types must survive normalization for validation.
    if req.chat_template_kwargs.as_ref().is_some_and(|kwargs| {
        ["thinking", "enable_thinking"].iter().any(|key| {
            kwargs
                .get(*key)
                .is_some_and(|value| !value.is_null() && !value.is_boolean())
        })
    }) {
        return;
    }
    let explicit = req
        .thinking
        .as_ref()
        .and_then(|thinking| thinking.r#type)
        .map(|kind| kind != ThinkingType::Disabled);
    let kwargs = req.chat_template_kwargs.as_ref();
    let mode = kwargs
        .and_then(|k| k.get("thinking_mode"))
        .and_then(Value::as_str)
        .and_then(|mode| match mode {
            "enabled" | "thinking" | "adaptive" => Some(true),
            "disabled" | "chat" => Some(false),
            _ => None,
        });
    let toggle = |key| kwargs.and_then(|k| k.get(key)).and_then(Value::as_bool);
    let effort = req
        .effective_reasoning_effort()
        .map(str::to_owned)
        .or_else(|| {
            kwargs.and_then(|k| k.get("reasoning_effort")).map(|value| {
                value
                    .as_str()
                    .map(str::to_owned)
                    .unwrap_or_else(|| value.to_string())
            })
        });
    let enabled = if is_v41_model(&req.model) {
        thinking_enabled(
            &req.model,
            explicit
                .or_else(|| toggle("thinking"))
                .or_else(|| toggle("enable_thinking"))
                .or(mode),
            effort.as_deref(),
        )
    } else {
        let disabled = [
            explicit,
            toggle("thinking"),
            toggle("enable_thinking"),
            mode,
        ]
        .contains(&Some(false));
        thinking_enabled(
            &req.model,
            if disabled {
                Some(false)
            } else if [
                explicit,
                toggle("thinking"),
                toggle("enable_thinking"),
                mode,
            ]
            .contains(&Some(true))
            {
                Some(true)
            } else {
                None
            },
            effort.as_deref(),
        )
    };
    for value in [
        req.reasoning_effort.as_mut(),
        req.thinking.as_mut().and_then(|t| t.effort.as_mut()),
    ]
    .into_iter()
    .flatten()
    {
        *value = normalize_effort(&req.model, value).to_owned();
        if enabled && value == "none" {
            *value = "high".into();
        }
    }
    let effort = effort
        .as_deref()
        .map(|effort| normalize_effort(&req.model, effort));
    let effort = match effort {
        None | Some("none") if enabled => Some("high"),
        other => other,
    };
    let kwargs = req.chat_template_kwargs.get_or_insert_default();
    kwargs.insert("thinking".into(), Value::Bool(enabled));
    kwargs.insert("enable_thinking".into(), Value::Bool(enabled));
    kwargs.insert(
        "thinking_mode".into(),
        Value::String(if enabled { "enabled" } else { "disabled" }.into()),
    );
    if let Some(effort) = effort {
        kwargs.insert("reasoning_effort".into(), Value::String(effort.into()));
    }
}

/// Validate normalized thinking controls without imposing Chat history rules
/// on a Responses replay reconstructed by the gateway.
pub fn validate_chat(req: &ChatCompletionRequest) -> Result<(), validator::ValidationError> {
    // Never let a malformed override block projection and make validation
    // disagree with a renderer that ignores that value.
    if let Some(kwargs) = &req.chat_template_kwargs {
        for key in ["thinking", "enable_thinking"] {
            if kwargs
                .get(key)
                .is_some_and(|value| !value.is_null() && !value.is_boolean())
            {
                return Err(error(
                    "invalid_thinking_override",
                    "DeepSeek thinking template overrides must be booleans or null",
                ));
            }
        }
        if let (Some(thinking), Some(alias)) = (
            kwargs.get("thinking").and_then(Value::as_bool),
            kwargs.get("enable_thinking").and_then(Value::as_bool),
        ) {
            if thinking != alias {
                return Err(error(
                    "conflicting_thinking_overrides",
                    "DeepSeek thinking and enable_thinking template overrides must agree",
                ));
            }
        }
    }
    // Validate both spellings, including one hidden by the other's precedence.
    for effort in [
        req.reasoning_effort.as_deref(),
        req.thinking
            .as_ref()
            .and_then(|thinking| thinking.effort.as_deref()),
    ]
    .into_iter()
    .flatten()
    {
        // The V4.1 native renderer restores ASCII-digit strings to 1..=100
        // budgets. Other V4 renderers do not support that extension.
        let is_v41 = is_v41_model(&req.model);
        let native_budget = is_v41
            && effort.bytes().all(|byte| byte.is_ascii_digit())
            && effort
                .parse::<u8>()
                .is_ok_and(|budget| (1..=100).contains(&budget));
        if !["none", "minimal", "low", "medium", "high", "xhigh", "max"].contains(&effort)
            && !native_budget
        {
            return Err(error(
                "reasoning_effort_not_allowed",
                if is_v41 {
                    "DeepSeek V4.1 reasoning effort must be none, low, high, xhigh, max or an integer budget from 1 to 100 (minimal and medium aliases are accepted)"
                } else {
                    "DeepSeek V4 reasoning effort must be none, low, high or max (minimal, medium and xhigh aliases are accepted)"
                },
            ));
        }
    }
    let forced = match req.tool_choice.as_ref() {
        Some(ToolChoice::Value(ToolChoiceValue::Required) | ToolChoice::Function { .. }) => true,
        Some(ToolChoice::AllowedTools { mode, .. }) => mode == "required",
        _ => false,
    };
    let enabled = req
        .chat_template_kwargs
        .as_ref()
        .and_then(|kwargs| kwargs.get("thinking"))
        .and_then(Value::as_bool)
        .unwrap_or_else(|| {
            thinking_enabled(
                &req.model,
                req.thinking_toggle(),
                req.effective_reasoning_effort(),
            )
        });
    if enabled && forced {
        return Err(error(
            "tool_choice_not_supported",
            "DeepSeek V4 requires thinking to be disabled for forced tool choice",
        ));
    }
    Ok(())
}

fn error(code: &'static str, message: &'static str) -> validator::ValidationError {
    let mut error = validator::ValidationError::new(code);
    error.message = Some(message.into());
    error
}
