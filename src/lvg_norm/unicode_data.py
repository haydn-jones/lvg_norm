"""Data-driven Unicode normalization helpers (q0, q7, q8).

Parses the LVG Unicode tables that live under ``lvg_norm/resources/Unicode`` and
exposes small helpers that mirror the LVG flow:

    1) map symbols/punctuation (symbolMap)
    2) map Unicode characters to ASCII lookalikes (unicodeMap)
    3) split ligatures (ligatureMap)
    4) strip diacritics (diacriticMap)
    5) strip or map remaining non-ASCII (nonStripMap)

Synonym mappings are parsed for completeness but are not applied in Norm's q7
flow (matching Java's ToUnicodeCoreNorm). These helpers are intentionally
lightweight so they can be reused for both the initial q0 pass and the later
q7/q8 passes inside the normalizer.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files

RESOURCE_PACKAGE = "lvg_norm.resources.Unicode"


@dataclass(frozen=True)
class UnicodeTables:
    symbol_map: dict[int, str]
    unicode_map: dict[int, str]
    ligature_map: dict[int, str]
    diacritic_map: dict[int, str]
    synonym_map: dict[int, str]
    non_strip_map: dict[int, str]


def _parse_codepoint(raw: str) -> int:
    if not raw.startswith("U+"):
        msg = f"Expected U+XXXX codepoint, got: {raw!r}"
        raise ValueError(msg)
    return int(raw.replace("U+", "0x"), base=0)


def _iter_table_lines(resource_name: str) -> Iterable[list[str]]:
    path = files(RESOURCE_PACKAGE).joinpath(resource_name)
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            yield line.split("|")


def _parse_simple_mapping(resource_name: str) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for parts in _iter_table_lines(resource_name):
        if len(parts) < 2:
            continue
        cp = _parse_codepoint(parts[0])
        mapped = parts[1]
        mapping[cp] = mapped
    return mapping


def _parse_synonym_mapping(resource_name: str) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for parts in _iter_table_lines(resource_name):
        if len(parts) < 2:
            continue
        cp_src = _parse_codepoint(parts[0])
        cp_dst = _parse_codepoint(parts[1])
        mapping[cp_src] = chr(cp_dst)
    return mapping


@lru_cache(maxsize=1)
def load_unicode_tables() -> UnicodeTables:
    return UnicodeTables(
        symbol_map=_parse_simple_mapping("symbolMap.data"),
        unicode_map=_parse_simple_mapping("unicodeMap.data"),
        ligature_map=_parse_simple_mapping("ligatureMap.data"),
        diacritic_map=_parse_simple_mapping("diacriticMap.data"),
        synonym_map=_parse_synonym_mapping("synonymMap.data"),
        non_strip_map=_parse_simple_mapping("nonStripMap.data"),
    )


def _map_chars(text: str, mapping: dict[int, str]) -> str:
    if not mapping:
        return text
    out: list[str] = []
    for ch in text:
        mapped = mapping.get(ord(ch))
        if mapped is None:
            out.append(ch)
        else:
            out.append(mapped)
    return "".join(out)


def _strip_diacritics_generic(text: str) -> str:
    """
    Best-effort stripping using canonical decomposition for unmapped chars.

    Java's ToStripDiacritics falls back to NFD (not NFKD) so compatibility
    characters like MICRO SIGN are left intact for the nonStripMap stage.
    """

    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def _core_norm_char(ch: str, tables: UnicodeTables) -> str:
    """
    Mirror Java's CoreNorm recursion for a single character.

    Ordering:
      1) symbolMap
      2) unicodeMap
      3) ligatureMap, else NFKC fallback when unmapped
      4) diacriticMap
      5) strip canonical diacritics (NFD) when present
    """

    cur = ch
    # Match Java's recursive limit (25 * len) with a simple loop guard.
    for _ in range(50):  # plenty for single characters
        if not cur:
            return cur

        # Java exits early for ASCII without attempting further mappings.
        if len(cur) == 1 and ord(cur) < 128:
            return cur

        # If the current expansion is multi-character, normalize each piece.
        if len(cur) > 1:
            return "".join(_core_norm_char(piece, tables) for piece in cur)

        code = ord(cur)

        mapped = tables.symbol_map.get(code)
        if mapped is not None:
            return mapped

        mapped = tables.unicode_map.get(code)
        if mapped is not None:
            return mapped

        if code in tables.ligature_map:
            lig = tables.ligature_map[code]
            if lig != cur:
                cur = lig
                continue
        else:
            # Java's ToSplitLigatures falls back to NFKC when no table entry
            # exists; trim only multi-char expansions to preserve mapped spaces
            # from NBSP/NNBSP and similar separators.
            nfkc_raw = unicodedata.normalize("NFKC", cur)
            nfkc = nfkc_raw.strip() if len(nfkc_raw) > 1 else nfkc_raw
            if nfkc != cur:
                cur = nfkc
                continue

        diacritic = tables.diacritic_map.get(code)
        if diacritic is not None and diacritic != cur:
            cur = diacritic
            continue

        stripped = _strip_diacritics_generic(cur)
        if stripped != cur:
            cur = stripped
            continue

        return cur

    return cur


def unicode_core_norm(text: str, tables: UnicodeTables | None = None) -> str:
    """Apply the LVG Unicode core sequence (q7-ish) using data tables."""

    if tables is None:
        tables = load_unicode_tables()

    return "".join(_core_norm_char(ch, tables) for ch in text)


def unicode_symbol_norm(text: str, tables: UnicodeTables | None = None) -> str:
    """
    Lightweight q0 pass: map symbols/punctuation to ASCII using symbolMap only.

    Mirrors Java's ToMapSymbolToAscii (applied before uninflection) so that
    diacritics and other Unicode mappings are postponed until the later q7
    core normalization step.
    """
    if tables is None:
        tables = load_unicode_tables()
    text = _map_chars(text, tables.symbol_map)
    text = _map_chars(text, tables.unicode_map)
    return text


def unicode_strip_or_map_non_ascii(text: str, tables: UnicodeTables | None = None) -> str:
    """Final q8 step: map via nonStripMap or drop non-ASCII characters."""

    if tables is None:
        tables = load_unicode_tables()

    out: list[str] = []
    for ch in text:
        code = ord(ch)
        if code < 128:
            out.append(ch)
            continue

        mapped = tables.non_strip_map.get(code)
        if mapped:
            out.append(mapped)
        # If mapped is empty string or key missing, the character is dropped.

    return "".join(out)
