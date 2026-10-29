"""Check exported full/prefilled V4.1 grammars with XGrammar and a real tokenizer."""

import json
import sys

import xgrammar as xgr
from transformers import PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast(tokenizer_file=sys.argv[2])
compiler = xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(tokenizer))
checks = 0
for case in json.load(open(sys.argv[1])):
    grammar = compiler.compile_structural_tag(json.dumps(case["grammar"]))
    call = '<｜DSML｜ invoke name="lookup">\n</｜DSML｜ invoke>\n'
    prefix = "" if case["prefilled"] else "\n\n<｜DSML｜ calls>\n"
    close = "</｜DSML｜ calls>"
    one = prefix + call + close
    for text, expected in [
        (one, True),
        (prefix + call + call + close, True),
        (prefix + close, False),
        (one + "ignored", False),
        (one + "\n\n<｜DSML｜ calls>\n" + call + close, False),
        (one.replace('name="lookup"', 'name="unknown"'), False),
    ]:
        matcher = xgr.GrammarMatcher(grammar)
        accepted = all(
            matcher.accept_token(token)
            for token in tokenizer.encode(text, add_special_tokens=False)
        ) and matcher.is_completed()
        assert accepted == expected, (case["prefilled"], text, expected)
        checks += 1
print(f"{checks} XGrammar acceptance/rejection checks passed")
