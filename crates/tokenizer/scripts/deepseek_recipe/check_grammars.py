"""Check V4.1 grammar acceptance and sampling masks with the production vocabulary."""

import json
import sys
from pathlib import Path

import xgrammar as xgr
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

source = Path(sys.argv[2])
raw = Tokenizer.from_file(str(source))
document = json.loads(source.read_text())
document["added_tokens"] = []
plain = Tokenizer.from_str(json.dumps(document, ensure_ascii=False))
dsml = "｜DSML｜"
dsml_id = raw.token_to_id(dsml)
info = xgr.TokenizerInfo.from_huggingface(
    PreTrainedTokenizerFast(tokenizer_object=raw),
    stop_token_ids=[raw.token_to_id("<｜end▁of▁sentence｜>")],
)
compiler = xgr.GrammarCompiler(info, max_threads=2)
mask = xgr.allocate_token_bitmask(1, info.vocab_size)
checks = 0


def encode(parts):
    """Merge adjacent plain regions, including header/value/close boundaries.

    Only explicitly structural parts may contain DSML control IDs. Literal
    marker spellings in values go through the ordinary vocabulary, as recipe
    replay encodes them. This exercises real BPE merges, including empty values.
    """
    ids, pending = [], ""
    for text, structural in parts:
        pieces = text.split(dsml) if structural else [text]
        for index, piece in enumerate(pieces):
            if index:
                ids.extend(plain.encode(pending, add_special_tokens=False).ids)
                ids.append(dsml_id)
                pending = ""
            pending += piece
    ids.extend(plain.encode(pending, add_special_tokens=False).ids)
    return ids


def check(compiled, ids, expected, label):
    global checks
    # Check both APIs independently: accept_token alone previously missed
    # sampling masks that denied the next delimiter token of valid strings.
    for masks in [False, True]:
        matcher = xgr.GrammarMatcher(compiled)
        accepted = True
        for index, token in enumerate(ids):
            if masks:
                matcher.fill_next_token_bitmask(mask)
                allowed = bool(int(mask[0, token // 32]) & (1 << (token % 32)))
                if not allowed:
                    accepted = False
                    break
            if not matcher.accept_token(token):
                accepted = False
                break
        accepted = accepted and matcher.is_completed()
        assert accepted == expected, (label, "mask" if masks else "accept", index, token, expected)
        checks += 1


for case in json.loads(Path(sys.argv[1]).read_text()):
    serialized = json.dumps(case["grammar"])
    # Backends can validate before loading a tokenizer; no symbolic terminals
    # may survive gateway preparation.
    xgr.Grammar.from_structural_tag(serialized)
    grammar = compiler.compile_structural_tag(serialized)
    prefix = "" if case["prefilled"] else "\n\n<｜DSML｜ calls>\n"
    begin = '<｜DSML｜ invoke name="lookup">\n'
    end = "</｜DSML｜ invoke>\n"
    close = "</｜DSML｜ calls>"
    one = prefix + begin + end + close
    label = (case["source"], case["required"], case["one_call"], case["prefilled"])
    for text, expected in [
        (one, True),
        (prefix + begin + end + begin + end + close, not case["one_call"]),
        (prefix + close, False),
        (one + "ignored", False),
        (one + "\n\n<｜DSML｜ calls>\n" + begin + end + close, False),
        (one.replace('name="lookup"', 'name="unknown"'), False),
    ]:
        # Full production tokenization must work, not just separately encoded tags.
        check(grammar, raw.encode(text, add_special_tokens=False).ids, expected, (label, text))
    check(grammar, encode([(one, False)]), not case["required"], (label, "ordinary fake call"))
    if not case["required"]:
        check(grammar, encode([("ordinary response", False)]), True, (label, "no call"))
        check(grammar, encode([("answer before", False), (one, True)]), True, (label, "text then call"))

    invalid_head = prefix + begin + '<｜DSML｜ parameter name="q" string="false">'
    invalid_tail = "</｜DSML｜ parameter>\n" + end + close
    check(grammar, encode([(invalid_head, True), ("{broken}", False), (invalid_tail, True)]), False, (label, "invalid JSON"))

    for string, value in [
        (True, ""),
        (True, "hello"),
        (True, "</not a delimiter"),
        (True, 'quoted "unicode 中文"\nnext line'),
        (True, "literal ｜DSML｜ text"),
        (True, "literal </｜DSML｜ parameter> text"),
        (True, "literal </｜DSML｜ invoke> text"),
        (True, "literal </｜DSML｜ calls> text"),
        (False, "42"),
        (False, "-0.00"),
        (False, "1e+00"),
        (False, "[1,2,3]"),
        (False, '{ "n":1e+00, "s":"\\u0061" }'),
        (False, '"hello"'),
        (False, 'true'),
        (False, 'null'),
        (False, '{"literal ｜DSML｜ key":1}'),
        (False, '{"text":"literal </｜DSML｜ parameter>"}'),
        (False, '{"text":"literal </｜DSML｜ invoke>"}'),
        (False, '{"text":"literal ｜DSML｜ text"}'),
    ]:
        head = prefix + begin + '<｜DSML｜ parameter name="q" string="' + str(string).lower() + '">'
        tail = "</｜DSML｜ parameter>\n" + end + close
        check(grammar, encode([(head, True), (value, False), (tail, True)]), True, (label, string, value))
        # Also cover streams that split the header and value at token boundaries.
        split = encode([(head, True)]) + encode([(value, False)]) + encode([(tail, True)])
        check(grammar, split, True, (label, "split", string, value))
        if dsml in value:
            check(grammar, encode([(head + value + tail, True)]), False, (label, "real control in value", string, value))

print(f"{checks} XGrammar acceptance and sampling-mask checks passed")
