import pytest

from src.crawlers.startups import is_ai_startup


@pytest.mark.parametrize("h,expected", [
    ({"one_liner": "AI teammates for logistics"}, True),
    ({"tags": ["Generative AI"]}, True),
    ({"industries": ["B2B"], "subindustry": "B2B -> Engineering, Product and Design", "tags": ["Computer Vision"]}, True),
    ({"one_liner": "Fine-tune LLMs in one click"}, True),
    ({"one_liner": "Pay instantly with your fingerprint."}, False),
    ({"one_liner": "Decarbonizing the chemicals industry.", "tags": ["Climate"]}, False),
    ({"one_liner": "Maintain your railway", "tags": []}, False),  # 'ai' inside a word must not match
])
def test_is_ai(h, expected):
    assert is_ai_startup(h) is expected
