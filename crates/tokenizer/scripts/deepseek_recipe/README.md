# DeepSeek V4.1 recipe oracle

The authoritative V4.1 rendering fixture is
`../../tests/fixtures/deepseek_v41/recipe.json`, recorded from the clean
`deepseek-recipe` checkout at `3c63f864a2e3b9b564e3de8a6ff409cea6be9933`.
`inputs.json` contains 34 conversations covering effort, message boundaries,
history, assistant prefixes, tool results, required calls, response formats,
images, grounding, and literal control-token spellings. Expected prompt text,
special-token ranges, and token IDs come from the recipe, not SMG.

Regenerate, then verify reproducibility:

```sh
DEEPSEEK_V41_MODEL_DIR=/path/to/production/checkpoint \
  bash record.sh /path/to/deepseek-recipe
DEEPSEEK_V41_MODEL_DIR=/path/to/production/checkpoint \
  bash record.sh /path/to/deepseek-recipe --check
```

The tokenizer defaults to `../../.tokenizer_cache/deepseek_v41/tokenizer.json`.
The recorder uses the production `<｜deepseek_image｜>` image token instead of
`<｜image｜>`, enables preserved tool-argument spelling, and defaults
`drop_thinking` to false, matching SMG's adapter policy.
The adjacent MIT license accompanies the recipe-derived encoding behavior.

The tokenizer tests replay all core conversations. The gateway test also
replays the 31 conversations representable by its Chat DTO; dynamic system
tool declarations, deferred tools, and the internal latest-reminder role are
renderer fixtures, not claims of new API support. Messages tests cover its
block separators and thinking controls. SMG retains explicit numeric effort
budgets (1–100) as an extension. V4 and V4.1 retain historical reasoning by
default (`drop_thinking: false`); `chat_template_kwargs.drop_thinking: true`
opts into pruning. The oracle covers both retention and explicit pruning, plus raw nested JSON
spacing, escapes, large integers, and numeric spelling in tool history.

The older `render_fixtures.json`, `render_ids_fixtures.json`, and
`generate_deepseek_v41_fixtures.py` retain the previous engine-format reference;
they are not the acceptance oracle for the recipe renderer.

From the repository root, export the native and Symphony grammars before
and after the prefilled block opener, then check them with XGrammar:

```sh
SMG_V41_GRAMMARS_OUT=/tmp/v41-grammars.json \
  cargo test --locked -p smg --lib prefilled_dsml_constraints
python crates/tokenizer/scripts/deepseek_recipe/check_grammars.py \
  /tmp/v41-grammars.json /path/to/production/checkpoint/tokenizer.json
```
