import re
import unicodedata
from collections.abc import Iterable, Sequence
from itertools import product

from lvg_norm.lexicon import citation_form, lexicon_uninflect, rule_uninflect
from lvg_norm.unicode_data import (
    load_unicode_tables,
    unicode_core_norm,
    unicode_strip_or_map_non_ascii,
    unicode_symbol_norm,
)
from lvg_norm.wordlists import load_lvg_stopword_set, load_remove_s_rules

WHITESPACE_RE = re.compile(r"[ \t]+")
REMOVE_S_PUNCT = {"-", "(", ","}
GENITIVE_DELIMS = {" ", "\t", ","}


def _remove_genitive_suffix(word: str, suffix: str, remove_chars: int) -> str:
    """Mirror Java's RemoveLastChars: drop suffix chars only when at the end."""

    lowered = word.lower()
    size = len(word)
    suffix_len = len(suffix)
    idx = lowered.rfind(suffix)

    while idx == size - suffix_len and idx > 0:
        word = word[: idx + suffix_len - remove_chars]
        lowered = word.lower()
        size = len(word)
        idx = lowered.rfind(suffix)

    return word


def _remove_genitive_from_word(word: str) -> str:
    """Remove trailing possessives only (s'/x'/z'/ 's), matching Java."""

    if len(word) < 3:
        return word

    word = _remove_genitive_suffix(word, "s'", 1)
    word = _remove_genitive_suffix(word, "x'", 1)
    word = _remove_genitive_suffix(word, "z'", 1)
    word = _remove_genitive_suffix(word, "'s", 2)
    return word


def remove_genitives(text: str) -> str:
    """
    g: Remove possessive endings without stripping internal apostrophes.

    Java tokenizes using space/tab/comma delimiters and only trims possessive
    suffixes on each token. We mirror that here to avoid collapsing tokens like
    "Ala'4" or "3'UTR".
    """

    tokens: list[str] = []
    current: list[str] = []

    for ch in text:
        if ch in GENITIVE_DELIMS:
            if current:
                tokens.append("".join(current))
                current.clear()
            tokens.append(ch)
        else:
            current.append(ch)

    if current:
        tokens.append("".join(current))

    processed: list[str] = []
    for tok in tokens:
        if tok in GENITIVE_DELIMS:
            processed.append(tok)
        else:
            processed.append(_remove_genitive_from_word(tok))

    return "".join(processed).strip()


def _trim_trailing_spaces(text: str) -> str:
    """Remove trailing spaces only (mirrors Java TrimEnd)."""

    while text.endswith(" "):
        text = text[:-1]
    return text


def _remove_pattern_casefold(text: str, pattern: str) -> str:
    """Remove all case-insensitive occurrences of *pattern*."""

    out = text
    lower = out.lower()
    pat = pattern.lower()
    size = len(pattern)

    idx = lower.find(pat)
    while idx != -1:
        left = _trim_trailing_spaces(out[:idx])
        right = out[idx + size :]
        out = left + right
        lower = out.lower()
        idx = lower.find(pat)
    return out


def _match_remove_s_key(key: str, term: str, index: int) -> bool:
    """
    Mirror RWildCard.IsMatchKey used by Java's removeS reverse trie.

    Wildcards:
      ^ = beginning of string
      C = any character
      D = digit
      L = letter
      S = space
      P = punctuation (-,(,)
    """

    if index < 0:
        return key == "^"

    ch = term[index]
    lower = ch.lower()

    if key == lower:
        return True
    if key == "C":
        return True
    if key == "D":
        return ch.isdigit()
    if key == "L":
        return ch.isalpha()
    if key == "S":
        return ch == " "
    if key == "P":
        return ch in REMOVE_S_PUNCT
    return False


def _matches_remove_s_pattern(term: str, pattern: str) -> bool:
    """Check if *term* matches a removeS pattern suffix."""

    if not pattern.endswith("$"):
        return False

    pat = pattern[:-1]  # strip trailing $
    pos = len(term) - 1

    for key in reversed(pat):
        if not _match_remove_s_key(key, term, pos):
            return False
        pos -= 1

    return True


def _matches_any_remove_s_rule(term: str, rules: Sequence[str]) -> bool:
    return any(_matches_remove_s_pattern(term, rule.strip()) for rule in rules if rule.strip())


def remove_parenthetic_plurals(text: str, rules: Sequence[str]) -> str:
    """
    rs: Remove parenthetic plural markers, honoring removeS exception rules.

    Mirrors Java ToRemoveS: strip (es)/(ies) globally, then remove (s) unless
    the preceding context matches an exception pattern from removeS.data.
    """

    out = _remove_pattern_casefold(text, "(es)")
    out = _remove_pattern_casefold(out, "(ies)")

    lower = out.lower()
    pat = "(s)"
    pat_len = len(pat)
    idx = lower.find(pat)

    while idx != -1:
        left = out[:idx]
        right = out[idx + pat_len :]

        if not _matches_any_remove_s_rule(left, rules):
            out = left + right
            if right and right[0].isalpha():
                out = left + " " + right
            lower = out.lower()
            idx = lower.find(pat)
        else:
            idx = lower.find(pat, idx + 1)

    return out


def replace_punct_with_space(text: str) -> str:
    """
    o: Replace punctuation and symbols with spaces (based on Unicode category).
    """
    out_chars: list[str] = []
    punct_categories = {"Pd", "Ps", "Pe", "Pc", "Po"}
    symbol_categories = {"Sm", "Sc", "Sk"}  # match Java Char.IsPunctuation
    for ch in text:
        category = unicodedata.category(ch)
        if category in punct_categories or category in symbol_categories:
            out_chars.append(" ")
        else:
            out_chars.append(ch)
    return "".join(out_chars)


def tokenize(text: str) -> list[str]:
    return [t for t in WHITESPACE_RE.split(text) if t]


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

    def __init__(
        self,
        stopwords: Iterable[str] | None = None,
        *,
        use_lvg_stopwords: bool = True,
        use_lexicon: bool = True,
        use_citation: bool = True,
        remove_s_rules: Sequence[str] | None = None,
        max_combinations: int = 10,
        min_term_length: int = 3,
    ):
        base_stopwords = load_lvg_stopword_set() if use_lvg_stopwords else set()
        extra_stopwords = {s.lower() for s in stopwords} if stopwords else set()
        self.stopwords = base_stopwords | extra_stopwords
        self.remove_s_rules = list(remove_s_rules) if remove_s_rules is not None else load_remove_s_rules()
        self.max_combinations = max_combinations
        self.use_lexicon = use_lexicon
        self.use_citation = use_citation
        self.min_term_length = min_term_length
        self.unicode_tables = load_unicode_tables()

    def normalize(self, text: str) -> list[str]:
        """
        Normalize a single input string.

        Returns:
            A sorted list of all normalized forms (strings with words
            lowercased before morphology/citation, de-Unicoded, and sorted
            alphabetically).
        """

        # --- q0: map Unicode symbols/punctuation to ASCII-ish (symbolMap only)
        text_q0 = unicode_symbol_norm(text, self.unicode_tables)

        # --- g: remove genitives
        text_g = remove_genitives(text_q0)

        # --- rs: remove parenthetic plural markers
        text_rs = remove_parenthetic_plurals(text_g, self.remove_s_rules)

        # --- o: replace punctuation with spaces
        text_o = replace_punct_with_space(text_rs)

        # --- t + l: remove stopwords and lowercase
        tokens = [tok.lower() for tok in tokenize(text_o)]
        content_tokens = [tok for tok in tokens if tok not in self.stopwords]

        if not content_tokens:
            return []

        # --- B: uninflect (lexicon with rule backoff)
        base_variants: list[list[str]] = []
        for tok in content_tokens:
            bases: set[str] = set()
            if self.use_lexicon:
                lex_bases = lexicon_uninflect(tok)
                if lex_bases:
                    bases |= lex_bases
                else:
                    rule_bases = {
                        base for base in rule_uninflect(tok) if len(base) >= self.min_term_length or base == tok
                    }
                    bases |= rule_bases

            if not bases:
                bases.add(tok)

            base_variants.append(sorted(bases))  # deterministic order

        # Guard against combinatorial blow-up (Java's MAX_UNINFLECTIONS)
        total = 1
        for vs in base_variants:
            total *= max(1, len(vs))
            if total > self.max_combinations:
                base_variants = [[tok] for tok in content_tokens]
                break

        # --- Ct: map bases to citation form (applied even when guard fires)
        token_variants: list[list[str]] = []
        for _bases in base_variants:
            if self.use_citation:
                # Java lowercases after citation lookup (GetCitation)
                cited = {(citation_form(base) or base).lower() for base in _bases}
                token_variants.append(sorted(cited))
            else:
                token_variants.append(_bases)

        # --- Generate all combinations, then apply q7 + q8 + w
        normalized_strings: set[str] = set()

        for combo in product(*token_variants):
            candidate = " ".join(combo)

            # q7: Unicode core norm (full tables, includes diacritics)
            cand_q7 = unicode_core_norm(candidate, self.unicode_tables)

            # q8: strip/map remaining non-ASCII
            cand_q8 = unicode_strip_or_map_non_ascii(cand_q7, self.unicode_tables)

            # w: strip punctuation introduced by q7/q8 (mirrors final SortWords)
            cand_w = replace_punct_with_space(cand_q8)

            # Re-tokenize after q7/q8/w mappings
            toks_final = tokenize(cand_w)
            if not toks_final:
                continue

            # w: sort words alphabetically
            toks_final_sorted = sorted(toks_final)
            normalized_strings.add(" ".join(toks_final_sorted))

        return sorted(normalized_strings)


_DEFAULT_NORMER: NormNormalizer | None = None


def _get_default_normer() -> NormNormalizer:
    global _DEFAULT_NORMER
    if _DEFAULT_NORMER is None:
        _DEFAULT_NORMER = NormNormalizer()
    return _DEFAULT_NORMER


def lvg_normalize(
    text: str,
    *,
    stopwords: Iterable[str] | None = None,
    use_lvg_stopwords: bool = True,
    use_lexicon: bool = True,
    use_citation: bool = True,
    remove_s_rules: Sequence[str] | None = None,
    max_combinations: int = 10,
    min_term_length: int = 3,
) -> list[str]:
    """
    Normalize text using the LVG-inspired pipeline without managing an instance.

    When called with default options, reuse a shared NormNormalizer to avoid
    reloading resources on every invocation.
    """

    if (
        stopwords is None
        and use_lvg_stopwords
        and use_lexicon
        and use_citation
        and remove_s_rules is None
        and max_combinations == 10
        and min_term_length == 3
    ):
        normer = _get_default_normer()
    else:
        normer = NormNormalizer(
            stopwords=stopwords,
            use_lvg_stopwords=use_lvg_stopwords,
            use_lexicon=use_lexicon,
            use_citation=use_citation,
            remove_s_rules=remove_s_rules,
            max_combinations=max_combinations,
            min_term_length=min_term_length,
        )

    return normer.normalize(text)
