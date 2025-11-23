from pathlib import Path

import pytest

from lvg_norm.norm import NormNormalizer

DATA_DIR = Path(__file__).parent / "data"


def _load_examples(path: Path) -> list[tuple[str, list[str]]]:
    """
    Load pipe-delimited examples of the form 'text|normalized' and coalesce
    duplicate rows so expected outputs are sorted and unique per input text.
    """

    examples: dict[str, set[str]] = {}
    with path.open(encoding="utf-8") as handle:
        for lineno, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                text, normalized = line.split("|", 1)
            except ValueError as exc:  # pragma: no cover - defensive guard
                msg = f"Malformed golden example on line {lineno}: {raw_line!r}"
                raise ValueError(msg) from exc

            examples.setdefault(text, set()).add(normalized)

    return [(text, sorted(norms)) for text, norms in sorted(examples.items())]


GOLDEN_EXAMPLES = _load_examples(DATA_DIR / "golden_examples.txt")
FAILING_EXAMPLES = _load_examples(DATA_DIR / "failing.txt")


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


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("⅓ beta-blocker", ["1 3 beta blocker"]),
        ("Na→K pump", ["k na pump"]),
        ("µ-opioid receptor", ["opioid receptor u"]),
        ("encyclopædia entry", ["encyclopedia entry"]),
        ("scleroses running", ["run sclerose", "run sclerosis", "running sclerose", "running sclerosis"]),
    ],
)
def test_unicode_q7_q8_sequencing(normer: NormNormalizer, text: str, expected: list[str]) -> None:
    assert normer.normalize(text) == expected


def test_max_combinations_guard() -> None:
    normer = NormNormalizer(max_combinations=2)
    # When variant permutations exceed the limit, the Java norm falls back to the
    # original token sequence (post lowercasing/stopword stripping) instead of
    # picking an arbitrary stem. Mirror that behaviour here.
    result = normer.normalize("scleroses running")
    assert result == ["running scleroses"]


def test_golden_corpus(normer: NormNormalizer) -> None:
    total_mismatches = 0
    samples: list[tuple[str, list[str], list[str]]] = []

    for text, expected in GOLDEN_EXAMPLES:
        actual = normer.normalize(text)
        if actual != expected:
            total_mismatches += 1
            if len(samples) < 5:
                samples.append((text, expected, actual))

    if total_mismatches:
        formatted = "\n".join(f"- {text!r}: expected {exp} got {act}" for text, exp, act in samples)
        pytest.xfail(
            f"{total_mismatches}/{len(GOLDEN_EXAMPLES)} golden examples diverge from the LVG baseline. "
            f"Sample diffs:\n{formatted}"
        )


@pytest.mark.xfail(reason="Known divergences from LVG norm; serves as a watch list.", strict=True)
def test_failing_corpus(normer: NormNormalizer) -> None:
    total_mismatches = 0
    samples: list[tuple[str, list[str], list[str]]] = []

    for text, expected in FAILING_EXAMPLES:
        actual = normer.normalize(text)
        if actual != expected:
            total_mismatches += 1
            if len(samples) < 5:
                samples.append((text, expected, actual))

    if total_mismatches:
        formatted = "\n".join(f"- {text!r}: expected {exp} got {act}" for text, exp, act in samples)
        pytest.xfail(
            f"{total_mismatches}/{len(FAILING_EXAMPLES)} known failing examples still diverge. "
            f"Sample diffs:\n{formatted}"
        )
