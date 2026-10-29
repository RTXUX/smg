#!/usr/bin/env bash
set -euo pipefail
recipe_dir=$(realpath "${1:?usage: record.sh /path/to/deepseek-recipe [--check]}")
fixture_dir=$(cd "$(dirname "$0")" && pwd)
revision=3c63f864a2e3b9b564e3de8a6ff409cea6be9933
test "$(git -C "$recipe_dir" rev-parse HEAD)" = "$revision"
test -z "$(git -C "$recipe_dir" status --porcelain)"
record_dir=$(mktemp -d)
trap 'rm -rf "$record_dir"' EXIT
python3 - "$recipe_dir" "$fixture_dir" "$record_dir" <<'PY'
import json
import pathlib
import shutil
import sys
recipe, fixture, work = map(pathlib.Path, sys.argv[1:])
manifest = '[package]\nname="smg-recipe-recorder"\nversion="0.1.0"\nedition="2024"\n[dependencies]\nserde_json={version="1",features=["preserve_order"]}\ntokenizers="0.23.2"\n'
for crate in ('deepseek-recipe-core', 'deepseek-recipe-encoding'):
    manifest += f'{crate}={{path={json.dumps(str(recipe / crate))}}}\n'
(work / 'Cargo.toml').write_text(manifest)
(work / 'src').mkdir()
shutil.copyfile(fixture / 'record.rs', work / 'src/main.rs')
if (fixture / 'Cargo.lock').exists():
    shutil.copyfile(fixture / 'Cargo.lock', work / 'Cargo.lock')
PY
tokenizer_file=${DEEPSEEK_V41_MODEL_DIR:-"$fixture_dir/../../.tokenizer_cache/deepseek_v41"}/tokenizer.json
cargo run --locked --offline --quiet --manifest-path "$record_dir/Cargo.toml" -- "$fixture_dir/inputs.json" "$record_dir/fixture.json" "$tokenizer_file"
python3 - "$record_dir/fixture.json" "$tokenizer_file" <<'PY'
import hashlib
import json
import pathlib
import sys
output, tokenizer = map(pathlib.Path, sys.argv[1:])
fixture = json.loads(output.read_text())
fixture['tokenizer_sha256'] = hashlib.sha256(tokenizer.read_bytes()).hexdigest()
output.write_text(json.dumps(fixture, ensure_ascii=False, indent=2) + '\n')
PY
destination="$fixture_dir/../../tests/fixtures/deepseek_v41/recipe.json"
if [[ ${2:-} == --check ]]; then
    cmp "$record_dir/fixture.json" "$destination"
else
    cp "$record_dir/fixture.json" "$destination"
    cp "$record_dir/Cargo.lock" "$fixture_dir/Cargo.lock"
fi
