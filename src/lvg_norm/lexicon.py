from __future__ import annotations

import gzip
import importlib.resources.abc as resources_abc
import pickle
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache, lru_cache
from importlib.resources import files
from os import environ
from pathlib import Path

TABLES_PACKAGE = "lvg_norm.resources.tables"
RULES_PACKAGE = "lvg_norm.resources.rules"


# fmt: off
VOWEL_SET = {
    "a", "e", "i", "o", "u", 
    "à", "á", "â", "ã", "ä", "å",
    "è", "é", "ê", "ë",
    "ì", "í", "î", "ï",
    "ð", "ò", "ó", "ô", "õ", "ö", "ø",
    "ù", "ú", "û", "ü",
}
CONSONANT_SET = {
    "b", "c", "d", "f", "g", "h", "j", "k", "l",
    "m", "n", "p", "q", "r", "s", "t", "v", "w",
    "x", "y", "z", "ç", "ñ", "ý", "þ",
}
LETTER_SET = VOWEL_SET | CONSONANT_SET
LEGAL_WILDCARDS = {"V", "C", "S", "D", "L", "^", "$"}
# fmt: on


@dataclass(frozen=True)
class InflectionRule:
    in_suffix: str
    out_suffix: str
    exceptions: frozenset[str]


@dataclass(frozen=True)
class _InflTables:
    base_index: dict[str, set[str]]
    citation_index: dict[str, set[str]]


_CACHE_VERSION = 1


def _infl_cache_path() -> Path:
    base_dir = environ.get("LVG_NORM_CACHE_DIR") or environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
    return Path(base_dir) / "lvg_norm" / "infl_tables.pkl"


def _load_cached_tables(cache_path: Path, source_mtime: float) -> _InflTables | None:
    try:
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
    except OSError:
        return None
    except pickle.UnpicklingError:
        return None

    if not isinstance(payload, dict):
        return None
    if payload.get("version") != _CACHE_VERSION:
        return None
    if payload.get("source_mtime", 0) < source_mtime:
        return None

    base_index = payload.get("base_index")
    citation_index = payload.get("citation_index")
    if not isinstance(base_index, dict) or not isinstance(citation_index, dict):
        return None

    return _InflTables(base_index, citation_index)


def _write_cached_tables(cache_path: Path, tables: _InflTables, source_mtime: float) -> None:
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(".tmp")
        payload = {
            "version": _CACHE_VERSION,
            "source_mtime": source_mtime,
            "base_index": tables.base_index,
            "citation_index": tables.citation_index,
        }
        with tmp_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(cache_path)
    except OSError:
        # Best-effort cache; ignore write failures.
        return


def _traversable_mtime(node: resources_abc.Traversable) -> float:
    stat_fn = getattr(node, "stat", None)
    if callable(stat_fn):
        try:
            return stat_fn().st_mtime  # type: ignore[call-arg]
        except OSError:
            return 0.0
    return 0.0


@lru_cache(maxsize=1)
def _load_rule_system() -> list[InflectionRule]:
    rule_files = ["im.rul", "plural.rul", "verbinfl.rul"]
    rules: list[InflectionRule] = []
    current_index = -1

    for name in rule_files:
        path = files(RULES_PACKAGE).joinpath(name)
        with path.open(encoding="utf-8") as handle:
            for raw in handle:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("RULE:"):
                    body = line[len("RULE:") :].strip()
                    parts = body.split("|")
                    if len(parts) < 4:
                        continue
                    # Load in reverse (inflected -> base) like the Java trie
                    in_suffix = parts[3].strip()
                    out_suffix = parts[0].strip()
                    rules.append(InflectionRule(in_suffix, out_suffix, frozenset()))
                    current_index = len(rules) - 1
                elif line.startswith("EXCEPTION:"):
                    if current_index == -1:
                        continue
                    body = line[len("EXCEPTION:") :].strip()
                    exc_terms: set[str] = set(rules[current_index].exceptions)
                    for pair in body.split(";"):
                        pair = pair.strip()
                        if not pair:
                            continue
                        parts = pair.split("|")
                        if len(parts) != 2:
                            continue
                        base, inflected = parts[0].strip().lower(), parts[1].strip().lower()
                        # Reverse the exception key to match reversed rules
                        key = inflected or base
                        if key:
                            exc_terms.add(key)
                    rules[current_index] = InflectionRule(
                        rules[current_index].in_suffix,
                        rules[current_index].out_suffix,
                        frozenset(exc_terms),
                    )

    return rules


@lru_cache(maxsize=1)
def _infl_source_mtime() -> float:
    path_gz = files(TABLES_PACKAGE).joinpath("infl.data.gz")
    path_plain = files(TABLES_PACKAGE).joinpath("infl.data")
    candidates = [_traversable_mtime(p) for p in (path_gz, path_plain) if p.is_file()]
    return max(candidates) if candidates else 0.0


@lru_cache(maxsize=1)
def _load_infl_tables() -> _InflTables:
    """
    Build both base and citation lookup tables in a single pass over infl.data.
    """

    path_gz = files(TABLES_PACKAGE).joinpath("infl.data.gz")
    path_plain = files(TABLES_PACKAGE).joinpath("infl.data")
    cache_path = _infl_cache_path()
    source_mtime = _infl_source_mtime()

    cached = _load_cached_tables(cache_path, source_mtime)
    if cached is not None:
        return cached

    base_index: dict[str, set[str]] = defaultdict(set)
    citation_index: dict[str, set[str]] = defaultdict(set)

    with (
        gzip.open(path_gz, "rt", encoding="utf-8")  # type: ignore
        if path_gz.is_file()
        else path_plain.open(encoding="utf-8")
    ) as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if not line or line.startswith("#"):
                continue

            parts = line.split("|")
            if len(parts) >= 5:
                surface = parts[0].lower()
                base = parts[4].strip().lower()
                if base:
                    base_index[surface].add(base)

            if len(parts) >= 6:
                infl = parts[2].strip()
                citation = parts[5].strip()
                # Preserve citation case to mirror Java's comparator ordering.
                if infl == "1" and citation:
                    surface = parts[0].strip().lower()
                    citation_index[surface].add(citation)

    tables = _InflTables(dict(base_index), dict(citation_index))
    _write_cached_tables(cache_path, tables, source_mtime)
    return tables


@lru_cache(maxsize=1)
def _load_infl_index() -> dict[str, set[str]]:
    """
    Surface -> uninflected base(s) pulled from the infl table.

    Mirrors DbUninflection.GetUninflections: only the uninflected term
    (column 5) is returned; citation forms (column 6) are *not* mixed in.
    """

    return _load_infl_tables().base_index


@lru_cache(maxsize=1)
def _inflected_surface_set() -> set[str]:
    """Lower‑cased surfaces that occur in the inflection table."""

    idx = _load_infl_index()
    return set(idx.keys())


@lru_cache(maxsize=1)
def _load_citation_index() -> dict[str, set[str]]:
    """
    Base form -> citation targets, restricted to base inflection rows.

    Matches ToNormalize.GetCitation, which calls DbCitation.GetCitationsFromBase
    (infl = 1) and then picks the first citation after sorting.
    """

    return _load_infl_tables().citation_index


def lexicon_uninflect(word: str) -> set[str]:
    """Return base forms for *word* using the SPECIALIST infl table."""

    idx = _load_infl_index()
    return set(idx.get(word.lower(), ()))


def citation_form(word: str) -> str | None:
    """
    Return the citation (canonical) form for a base token if available.

    Mirrors ToNormalize.GetCitation:
      - look up citations for the *base* term (infl = BASE_BIT / value 1)
      - pick the shortest citation; tie-break lexicographically
      - do not branch: at most one citation per input token
    """

    citations = _load_citation_index()
    options = citations.get(word.lower())
    if not options:
        return None

    return sorted(options, key=lambda c: (len(c), c))[0]


def _wildcard_matches_key(key: str, index: int, chars: Sequence[str]) -> bool:
    if index < 0:
        return key == "^"
    if index >= len(chars):
        return False

    ch = chars[index]
    if key == ch:
        return True
    if key == "V":
        return ch in VOWEL_SET
    if key == "C":
        return ch in CONSONANT_SET
    if key == "D":
        return ch.isdigit()
    if key == "L":
        return ch in LETTER_SET or ch.isalpha()
    if key == "S":
        return index < len(chars) - 1 and ch == chars[index + 1]
    if key == "$":
        return index == len(chars) - 1
    return False


def _matches_suffix(term: str, pattern: str) -> bool:
    """Mirror WildCard.IsMatchKey over a reversed trie suffix."""

    chars = list(term) + ["$"]
    for offset, key in enumerate(reversed(pattern)):
        index = len(chars) - 1 - offset
        if not _wildcard_matches_key(key, index, chars):
            return False
    return True


def _apply_suffix(term: str, in_suffix: str, out_suffix: str) -> str:
    """
    Reproduce WildCard.GetSuffix: copy wildcard positions from the input
    suffix, otherwise emit literal characters.
    """

    temp = term + "$"
    in_len = len(in_suffix)
    end_str = temp[-in_len:]
    pieces: list[str] = []

    for i, char in enumerate(out_suffix):
        if char in LEGAL_WILDCARDS and char != "$":
            if i >= in_len - 1:
                # outsuffix longer than insuffix: reuse the penultimate char
                source = end_str[-2] if len(end_str) >= 2 else end_str[0]
            else:
                source = end_str[i]
            pieces.append(source)
        else:
            pieces.append(char)

    unchanged = temp[: len(temp) - in_len]
    out = unchanged + "".join(pieces)
    return out.removesuffix("$")


@cache
def rule_uninflect(word: str) -> set[str]:
    """Return base forms using SPECIALIST rule files (plural/verb/IM rules)."""

    rules = _load_rule_system()
    inflected_surfaces = _inflected_surface_set()
    w = word.lower()
    bases: set[str] = set()

    for rule in rules:
        if w in rule.exceptions:
            continue
        if not _matches_suffix(w, rule.in_suffix):
            continue
        candidate = _apply_suffix(w, rule.in_suffix, rule.out_suffix)
        if candidate not in inflected_surfaces:
            bases.add(candidate)

    return bases
