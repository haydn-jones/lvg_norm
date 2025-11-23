import pytest

from lvg_norm.norm import NormNormalizer


@pytest.fixture
def normer() -> NormNormalizer:
    return NormNormalizer()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("HNF1A p.Q125*", ["hnf1a p q125", "hnf1on p q125", "hnf1um p q125"]),
        ("Zea mays B73", ["b73 may zea"]),
        ("lactone compounds", ["compound lactone"]),
        ("Mus musculus C57BL/6", ["6 c57bl mu musculus", "6 c57bl mus musculus"]),
        ("scleróses", ["scleros", "sclerose", "scleroses", "sclerosis"]),
        ("β-lactam antibiotics", ["antibiotic beta lactam"]),
    ],
)
def test_examples_from_design(normer: NormNormalizer, text: str, expected: list[str]) -> None:
    assert normer.normalize(text) == expected


def test_max_combinations_guard() -> None:
    normer = NormNormalizer(max_combinations=2)
    # First token has multiple variants; guard collapses to a single variant per token.
    result = normer.normalize("scleroses running")
    assert result == ["run scleros"]
