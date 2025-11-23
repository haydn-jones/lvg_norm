from __future__ import annotations

from collections.abc import Iterable
from importlib.resources import files

MISC_PACKAGE = "lvg_norm.resources.misc"


def _read_wordlist(name: str) -> list[str]:
    path = files(MISC_PACKAGE).joinpath(name)
    words: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            words.append(line)
    return words


def load_stopwords() -> set[str]:
    return {w.lower() for w in _read_wordlist("stopWords.data")}


def load_remove_s_rules() -> list[str]:
    return _read_wordlist("removeS.data")


def load_lvg_stopword_set(extra: Iterable[str] | None = None) -> set[str]:
    # Norm uses only the core stopWords list (see ToStripStopWords in Java).
    base = load_stopwords()
    if extra:
        base |= {w.lower() for w in extra if w}
    return base
