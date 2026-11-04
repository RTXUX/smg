This is the GLM-5.3-Flash chat template from the local
`zai-org/GLM-5.3-Flash` checkpoint, with the assistant `content.strip()`
condition and expression replaced by an unconditional `{{- content -}}`.
The same edit was applied directly to the model directory's template.

The argument loop uses the shared SMG Jinja helper `tool_argument_items`.
It accepts the client's original `function.arguments` JSON string, decodes
string values without trimming, and gives the template each other value's
original JSON spelling. The template writes those values verbatim. Any model's
template can call the same helper; no model-specific branch or template source
replacement is used. This helper is an SMG Jinja extension.

The integration test covers argument spacing, numeric spelling, escapes,
and content whitespace. The gateway test additionally exercises Symphony,
streaming/non-streaming Chat Completions, and replay through typed requests.
Set `SMG_GLM53_TOKENIZER` to a real `tokenizer.json` when running
`glm53_symphony_chat_replay_preserves_generated_prefix` to also assert the
canonical token prefix; without it the test checks the complete text path.
