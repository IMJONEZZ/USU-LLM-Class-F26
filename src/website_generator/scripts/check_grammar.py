"""Run: uv run --group inference python -m src.website_generator.scripts.check_grammar."""

from pathlib import Path

import xgrammar as xgr


def main():
    source = (Path(__file__).resolve().parents[1] / "backend/grammar.gbnf").read_text()
    grammar = xgr.Grammar.from_ebnf(source)
    # Byte-level tokenizer lets the compiler/matcher run without model downloads or GPU.
    info = xgr.TokenizerInfo(
        [bytes([i]) for i in range(256)], vocab_type=xgr.VocabType.RAW
    )
    compiled = xgr.GrammarCompiler(info).compile_grammar(grammar)
    for text, expected in [
        ("<p>Hello</p>", True),
        (
            (
                '<section><h1>Books</h1><p><a href="https://example.org">Read</a></p>'
                "<ul><li>One</li></ul></section>"
            ),
            True,
        ),
        ("<script>alert(1)</script>", False),
        ('<p onclick="x">Hi</p>', False),
        ('<p><a href="javascript:x">Hi</a></p>', False),
        ("<p>unfinished", False),
    ]:
        matcher = xgr.GrammarMatcher(compiled)
        accepted = matcher.accept_string(text) and matcher.is_completed()
        assert accepted == expected, text
    print("XGrammar compiled; all six acceptance/rejection checks passed.")


if __name__ == "__main__":  # pragma: no cover
    main()
