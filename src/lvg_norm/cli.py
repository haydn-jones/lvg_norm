from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from pathlib import Path

from .norm import NormNormalizer


def load_stopwords(path: Path | None) -> Iterable[str]:
    if not path:
        return []
    with path.open("r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


def collect_inputs(args: argparse.Namespace) -> list[str]:
    inputs: list[str] = []

    if args.file:
        inputs.extend(line.rstrip("\n") for line in args.file.read_text(encoding="utf-8").splitlines() if line.strip())

    if args.text:
        inputs.extend([t for t in args.text if t.strip()])

    # If nothing was provided, read non-empty lines from stdin
    if not inputs and not sys.stdin.isatty():
        inputs.extend([line.strip() for line in sys.stdin if line.strip()])

    return inputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize text using the LVG-inspired pipeline.")
    parser.add_argument(
        "text",
        nargs="*",
        help="Input text to normalize. If omitted, reads from --file or stdin.",
    )
    parser.add_argument(
        "-f",
        "--file",
        type=Path,
        help="Path to a file containing one input string per line.",
    )
    parser.add_argument(
        "--stopwords",
        type=Path,
        help="Optional stopword list (one token per line). Default stopwords are used otherwise.",
    )
    parser.add_argument(
        "--no-lvg-stopwords",
        action="store_true",
        help="Disable built-in LVG stop words/non-info/conjunction lists (use only custom stopwords).",
    )
    parser.add_argument(
        "--max-combinations",
        type=int,
        default=10,
        help="Limit on variant combinations to guard against explosion (default: 10, matches LVG MAX_RULE_UNINFLECTED_TERMS).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    stopwords = load_stopwords(args.stopwords)
    inputs = collect_inputs(args)

    if not inputs:
        parser.error("No input provided. Supply text, --file, or stdin.")

    normer = NormNormalizer(
        stopwords=stopwords,
        use_lvg_stopwords=not args.no_lvg_stopwords,
        max_combinations=args.max_combinations,
    )

    for original in inputs:
        normalized = normer.normalize(original)
        joined = "; ".join(normalized) if normalized else "(no tokens)"
        print(f"{original} -> {joined}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
