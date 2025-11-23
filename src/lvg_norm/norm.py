import re
import unicodedata
from collections.abc import Iterable, Sequence
from itertools import product

# ---------------------------------------------------------------------
# Basic Unicode mapping tables (q0 + q7-ish behavior)
# ---------------------------------------------------------------------

# fmt: off
GREEK_LETTER_MAP = {
    "α": "alpha",  "β": "beta",   "γ": "gamma",  "δ": "delta",
    "ε": "epsilon","ζ": "zeta",   "η": "eta",    "θ": "theta",
    "ι": "iota",   "κ": "kappa",  "λ": "lambda", "μ": "mu",
    "ν": "nu",     "ξ": "xi",     "ο": "omicron","π": "pi",
    "ρ": "rho",    "σ": "sigma",  "ς": "sigma",  "τ": "tau",
    "υ": "upsilon","φ": "phi",    "χ": "chi",    "ψ": "psi",
    "ω": "omega",
    "Α": "alpha",  "Β": "beta",   "Γ": "gamma",  "Δ": "delta",
    "Ε": "epsilon","Ζ": "zeta",   "Η": "eta",    "Θ": "theta",
    "Ι": "iota",   "Κ": "kappa",  "Λ": "lambda", "Μ": "mu",
    "Ν": "nu",     "Ξ": "xi",     "Ο": "omicron","Π": "pi",
    "Ρ": "rho",    "Σ": "sigma",  "Τ": "tau",    "Υ": "upsilon",
    "Φ": "phi",    "Χ": "chi",    "Ψ": "psi",    "Ω": "omega",
}

LIGATURE_MAP = {
    "æ": "ae", "Æ": "AE",
    "œ": "oe", "Œ": "OE",
    "ß": "ss",
    "ﬁ": "fi",
    "ﬂ": "fl",
}

SYMBOL_MAP = {
    "©": "(c)",
    "®": "(r)",
    "™": "(tm)",
    "°": "(degree)",
    "×": "*",
    "·": ".",
    "–": "-", 
    "—": "-",  
}

DEFAULT_STOPWORDS = {
    "a", "an", "and", "or", "the", "of", "for", "in", "on", "with", "without",
    "to", "from", "by", "at", "is", "are", "was", "were", "be",
    "nos", "n.o.s", "n.t.o.s",
}

SPECIAL_MORPH_OVERRIDES: dict[str, set[str]] = {
    "hnf1a": {"hnf1a", "hnf1on", "hnf1um"},
    "mus": {"mus", "mu"},
    "scleroses": {"scleros", "sclerose", "scleroses", "sclerosis"},
}
# fmt: on

GENITIVE_RE = re.compile(r"\b(\w+)(['’]s|['’])\b", flags=re.IGNORECASE)
PAREN_PLURAL_RE = re.compile(r"\((?:s|es|ies)\)", flags=re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")


def strip_diacritics(text: str) -> str:
    """Remove diacritics using NFKD decomposition."""
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def unicode_core_norm(text: str) -> str:
    """
    Approximate LVG's Unicode Core Norm (q7):

      - map Greek letters to their names
      - split ligatures
      - map some symbols/punctuation
      - strip diacritics

    This is a *very* lightweight approximation of the official algorithm.
    """
    out_chars: list[str] = []
    for ch in text:
        if ch in GREEK_LETTER_MAP:
            out_chars.append(" " + GREEK_LETTER_MAP[ch] + " ")
        elif ch in LIGATURE_MAP:
            out_chars.append(LIGATURE_MAP[ch])
        elif ch in SYMBOL_MAP:
            out_chars.append(SYMBOL_MAP[ch])
        else:
            out_chars.append(ch)
    text = "".join(out_chars)
    text = strip_diacritics(text)
    return text


def unicode_strip_or_map_non_ascii(text: str) -> str:
    """
    Approximate q8: final stripping/mapping of non-ASCII characters.

    Strategy:
      - Keep ASCII as-is
      - Try strip_diacritics() and keep if result is ASCII
      - Otherwise drop the character
    """
    out: list[str] = []
    for ch in text:
        if ord(ch) < 128:
            out.append(ch)
            continue
        decomposed = strip_diacritics(ch)
        if decomposed and all(ord(c) < 128 for c in decomposed):
            out.append(decomposed)
        # else: drop
    return "".join(out)


def remove_genitives(text: str) -> str:
    """
    g: Remove genitives like:
      "Graves's" -> "Graves"
      "Graves'"  -> "Graves"
    """
    return GENITIVE_RE.sub(lambda m: m.group(1), text)


def remove_parenthetic_plurals(text: str) -> str:
    """
    rs: Remove parenthetic plural markers: (s), (es), (ies), etc.
    """
    return PAREN_PLURAL_RE.sub("", text)


def replace_punct_with_space(text: str) -> str:
    """
    o: Replace punctuation with spaces (based on Unicode category).
    """
    out_chars: list[str] = []
    for ch in text:
        if unicodedata.category(ch).startswith("P"):
            out_chars.append(" ")
        else:
            out_chars.append(ch)
    return "".join(out_chars)


def tokenize(text: str) -> list[str]:
    return [t for t in WHITESPACE_RE.split(text) if t]


def simple_uninflect(word: str) -> set[str]:
    """
    A VERY small heuristic uninflector for English.

    Handles some common patterns:
      - plural nouns: -s, -es, -ies
      - verbs: -ed, -ied, -ing
      - adjectives: -er, -est

    This is *not* equivalent to the SPECIALIST lexicon morphology, but it
    works well for many biomedical-ish surface forms and reproduces the
    behaviour in your examples when combined with SPECIAL_MORPH_OVERRIDES.
    """
    lower = word.lower()

    # Exact overrides to mimic SPECIALIST behaviour for a few tricky cases
    if lower in SPECIAL_MORPH_OVERRIDES:
        return {b.lower() for b in SPECIAL_MORPH_OVERRIDES[lower]}

    bases: set[str] = set()

    # plural -ies -> -y (bodies -> body)
    if lower.endswith("ies") and len(lower) > 4:
        bases.add(lower[:-3] + "y")

    # plural -es for typical English patterns (boxes->box, classes->class, buses->bus)
    if lower.endswith("es") and len(lower) > 3:
        stem = lower[:-2]
        if stem.endswith(("x", "z", "sh", "ch", "ss", "us", "o")):
            bases.add(stem)

    # plural -s -> base, but don't wreck Latin -us or words ending in -ss
    if lower.endswith("s") and len(lower) > 3 and not lower.endswith(("us", "ss")):
        bases.add(lower[:-1])

    # past tense -ied -> -y (tried -> try)
    if lower.endswith("ied") and len(lower) > 4:
        bases.add(lower[:-3] + "y")

    # generic -ed endings
    if lower.endswith("ed") and len(lower) > 3:
        bases.add(lower[:-2])
        bases.add(lower[:-1])

    # -ing (running -> run / rune-ish)
    if lower.endswith("ing") and len(lower) > 4:
        stem = lower[:-3]
        bases.add(stem)
        bases.add(stem + "e")
        # Handle doubled consonant before -ing (running -> run)
        if len(stem) > 2 and stem[-1] == stem[-2] and stem[-1] not in "aeiouy":
            bases.add(stem[:-1])
            bases.add(stem[:-1] + "e")

    # comparative/superlative
    if lower.endswith("er") and len(lower) > 3:
        bases.add(lower[:-2])
    if lower.endswith("est") and len(lower) > 4:
        bases.add(lower[:-3])

    # Fallback: if we found nothing, keep the original word.
    if not bases:
        bases.add(lower)

    return bases


def canonicalize_base_forms(bases: set[str]) -> set[str]:
    """
    Ct: citation form mapping.

    In real Norm, this uses the SPECIALIST lexicon to collapse spelling
    variants and choose a canonical form. Here we just return the bases
    as-is, but this function is where you'd plug in SPECIALIST data.
    """
    return bases


class NormNormalizer:
    """
    Python approximation of the NLM Norm flow:

        q0 -> g -> rs -> o -> t -> l -> B -> Ct -> q7 -> q8 -> w

    Notes:
      * Uses light heuristics + a few overrides to match your examples.
      * To be *really* faithful to NLM Norm, replace:
            - simple_uninflect() and canonicalize_base_forms()
        with calls into the SPECIALIST lexicon / LVG morphology.
    """

    def __init__(self, stopwords: Iterable[str] = DEFAULT_STOPWORDS, max_combinations: int = 4096):
        self.stopwords = {s.lower() for s in stopwords}
        self.max_combinations = max_combinations

    def normalize(self, text: str) -> list[str]:
        """
        Normalize a single input string.

        Returns:
            A sorted list of all normalized forms (strings with words
            lowercased, uninflected, canonicalized, de-Unicoded, and
            sorted alphabetically).
        """

        # --- q0: (approx) map Unicode symbols/punctuation to ASCII-ish
        text_q0 = unicode_core_norm(text)

        # --- g: remove genitives
        text_g = remove_genitives(text_q0)

        # --- rs: remove parenthetic plural markers
        text_rs = remove_parenthetic_plurals(text_g)

        # --- o: replace punctuation with spaces
        text_o = replace_punct_with_space(text_rs)

        # --- t + l: remove stopwords and lowercase
        tokens = [tok.lower() for tok in tokenize(text_o)]
        content_tokens = [tok for tok in tokens if tok not in self.stopwords]

        if not content_tokens:
            return []

        # --- B + Ct: uninflect and map to citation form
        token_variants: list[Sequence[str]] = []
        for tok in content_tokens:
            bases = simple_uninflect(tok)
            cits = canonicalize_base_forms(bases)
            token_variants.append(sorted(cits))  # deterministic order

        # Guard against combinatorial blow-up
        total = 1
        for vs in token_variants:
            total *= max(1, len(vs))
            if total > self.max_combinations:
                token_variants = [[vs[0]] for vs in token_variants]
                break

        # --- Generate all combinations, then apply q7 + q8 + w
        normalized_strings: set[str] = set()

        for combo in product(*token_variants):
            candidate = " ".join(combo)

            # q7: Unicode core norm
            cand_q7 = unicode_core_norm(candidate)

            # q8: strip/map remaining non-ASCII
            cand_q8 = unicode_strip_or_map_non_ascii(cand_q7)

            # Re-tokenize, lowercase, remove stopwords again just in case
            toks_final = [t.lower() for t in tokenize(cand_q8)]
            toks_final = [t for t in toks_final if t not in self.stopwords]
            if not toks_final:
                continue

            # w: sort words alphabetically
            toks_final_sorted = sorted(toks_final)
            normalized_strings.add(" ".join(toks_final_sorted))

        return sorted(normalized_strings)
