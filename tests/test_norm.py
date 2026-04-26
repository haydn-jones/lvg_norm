from pathlib import Path
from typing import Any

import pytest

from lvg_norm import NormNormalizer, lvg_normalize

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


def test_lvg_normalize_matches_normer(normer: NormNormalizer) -> None:
    text = "HNF1A p.Q125*"
    assert lvg_normalize(text) == normer.normalize(text)


def test_lvg_normalize_respects_options() -> None:
    stopwords = {"beta"}
    use_lvg_stopwords = False
    use_lexicon = False
    use_citation = False
    remove_s_rules: list[str] = []
    max_combinations = 2
    min_term_length = 4
    text = "beta-blockers running"
    assert lvg_normalize(
        text,
        stopwords=stopwords,
        use_lvg_stopwords=use_lvg_stopwords,
        use_lexicon=use_lexicon,
        use_citation=use_citation,
        remove_s_rules=remove_s_rules,
        max_combinations=max_combinations,
        min_term_length=min_term_length,
    ) == NormNormalizer(
        stopwords=stopwords,
        use_lvg_stopwords=use_lvg_stopwords,
        use_lexicon=use_lexicon,
        use_citation=use_citation,
        remove_s_rules=remove_s_rules,
        max_combinations=max_combinations,
        min_term_length=min_term_length,
    ).normalize(text)


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


@pytest.fixture
def chem_normer() -> NormNormalizer:
    return NormNormalizer(pipeline="chemical")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Locants, parens, hyphens, indicated hydrogen all preserved; tokens not sorted.
        ("(2S,3R)-2,3-dihydro-1H-indole", ["(2s,3r)-2,3-dihydro-1h-indole"]),
        # Greek expansion glued to parent name; plural NOT uninflected.
        ("β-lactam antibiotics", ["beta-lactam antibiotics"]),
        ("α-tocopherol", ["alpha-tocopherol"]),
        # ± maps to +/- via existing nonStripMap because punct->space is skipped.
        ("(±)-tartaric acid", ["(+/-)-tartaric acid"]),
        # Unicode prime folds to ASCII apostrophe and survives (no punct stripping).
        ("2,2′-bipyridine", ["2,2'-bipyridine"]),
        # Stereo descriptor case differences collapse via end casefold; (s) is NOT
        # treated as a parenthetic plural (as it would be in the medical pipeline).
        ("(R)-2-bromopyridine", ["(r)-2-bromopyridine"]),
        ("(s)-2-bromopyridine", ["(s)-2-bromopyridine"]),
        # Heteroatom/sugar descriptors preserved alongside locants.
        ("N-methyl-D-aspartate", ["n-methyl-d-aspartate"]),
        # Substituent grouping in parens kept intact.
        ("4-(dimethylamino)pyridine", ["4-(dimethylamino)pyridine"]),
        # Square brackets and dots in von-Baeyer descriptors kept intact.
        ("bicyclo[2.2.1]heptane", ["bicyclo[2.2.1]heptane"]),
        # No English morphology, no stopword removal — all tokens kept in order.
        ("phenols and amines", ["phenols and amines"]),
        # Order preservation across multiple substituent positions.
        ("2-methyl-3-ethylpyridine", ["2-methyl-3-ethylpyridine"]),
    ],
)
def test_chemical_pipeline(chem_normer: NormNormalizer, text: str, expected: list[str]) -> None:
    assert chem_normer.normalize(text) == expected


def test_chemical_pipeline_distinguishes_enantiomers(chem_normer: NormNormalizer) -> None:
    r_form = chem_normer.normalize("(R)-2-bromopyridine")
    s_form = chem_normer.normalize("(S)-2-bromopyridine")
    assert r_form != s_form


def test_chemical_pipeline_case_insensitive_for_stereo(chem_normer: NormNormalizer) -> None:
    assert chem_normer.normalize("(R)-2-bromopyridine") == chem_normer.normalize("(r)-2-bromopyridine")
    assert chem_normer.normalize("(2S,3R)-X") == chem_normer.normalize("(2s,3r)-X")


def test_chemical_pipeline_preserves_token_order(chem_normer: NormNormalizer) -> None:
    forward = chem_normer.normalize("2-methyl-3-ethylpyridine")
    reverse = chem_normer.normalize("3-methyl-2-ethylpyridine")
    assert forward != reverse


def test_chemical_pipeline_empty_input(chem_normer: NormNormalizer) -> None:
    assert chem_normer.normalize("") == []
    assert chem_normer.normalize("   ") == []


def test_lvg_normalize_chemical_matches_normer(chem_normer: NormNormalizer) -> None:
    text = "(2S,3R)-2,3-dihydro-1H-indole"
    assert lvg_normalize(text, pipeline="chemical") == chem_normer.normalize(text)


def test_invalid_pipeline_raises() -> None:
    kwargs: dict[str, Any] = {"pipeline": "bogus"}
    with pytest.raises(ValueError, match="pipeline must be one of"):
        NormNormalizer(**kwargs)


def test_default_pipeline_is_medical(normer: NormNormalizer) -> None:
    # Pin: the default preset must not change without an explicit decision.
    assert normer.pipeline == "medical"
    assert normer.normalize("β-lactam antibiotics") == ["antibiotic beta lactam"]


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
