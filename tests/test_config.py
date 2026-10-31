from src import config

# words the schema cannot express: a prompt asking for them fails both methods for the wrong reason
OUT_OF_SCHEMA_WORDS = ["table", "list", "image", "picture", "chart", "form", "video"]


def test_there_are_exactly_ten_prompts():
    assert len(config.PROMPTS) == 10


def test_prompts_are_non_empty_strings():
    assert all(isinstance(p, str) and p.strip() for p in config.PROMPTS)


def test_prompts_are_unique_ignoring_case_and_whitespace():
    normalized = {p.strip().lower() for p in config.PROMPTS}
    assert len(normalized) == len(config.PROMPTS)


def test_prompts_do_not_ask_for_things_the_schema_cannot_express():
    for prompt in config.PROMPTS:
        words = prompt.lower().replace(".", " ").replace(",", " ").split()
        assert not [w for w in OUT_OF_SCHEMA_WORDS if w in words], prompt


def test_base_prompt_template_wraps_a_request():
    filled = config.BASE_PROMPT_TEMPLATE.format(request="a page about cats")
    assert "a page about cats" in filled
    assert "{" not in filled


def test_every_prompt_fits_the_template():
    for prompt in config.PROMPTS:
        assert prompt in config.BASE_PROMPT_TEMPLATE.format(request=prompt)
