# lvg_norm
Python implementation of LVG Norm (https://lhncbc.nlm.nih.gov/LSG/Projects/lvg/current/docs/userDoc/tools/norm.html).

This package focuses on the `norm` flow from the NLM Lexical Tools. It bundles
the LVG-derived resources needed by the normalizer and exposes both a Python API
and a small CLI.

## What It Does

Given an input string, `lvg_norm` produces one or more normalized forms by
applying an LVG-inspired pipeline. Two presets are available:

### `pipeline="medical"` (default)

The full LVG-inspired flow, best for free-text English (MeSH/UMLS-style
content):

`q0 -> g -> rs -> o -> t -> l -> B -> Ct -> q7 -> q8 -> w`

In practice, that means it handles things like:

- Unicode folding
- Possessive stripping
- Parenthetic plural cleanup
- Stopword removal
- Lexicon/rule-based uninflection
- Citation-form mapping
- Final token sorting

The implementation is aimed at the `norm` tool behavior, not the full LVG
suite.

### `pipeline="chemical"`

For IUPAC names and small-molecule nomenclature, where punctuation is
structural and word order is meaningful. The flow is reduced to:

`q0 -> q7 -> q8 -> casefold + whitespace collapse`

Genitives, parenthetic-plural removal, punctuation-to-space, stopword
stripping, English uninflection/citation lookup, and the final token sort
are all skipped. This preserves locants, hyphens, parens, brackets, primes,
stereo descriptors, and substitution-position prefixes (`N-`, `1H-`, `D-`,
…). Greek letters are still expanded via the LVG `nonStripMap` (`α` →
`alpha`), but they remain glued to their parent name because the
punctuation step is skipped. `±` is mapped to `+/-` for the same reason.

Use `pipeline="medical"` for prose, `pipeline="chemical"` for chemistry
names — the two are not designed to be mixed within a single call. If you
can run names through OPSIN first, do that; reach for `pipeline="chemical"`
when you need fuzzy matching on messy strings that OPSIN can't parse.

## Install

From PyPI:

```bash
pip install lvg-norm
```

From the repository:

```bash
pip install .
```

For local development with uv:

```bash
uv sync --group dev
```

## Python API

The distribution name is `lvg-norm`, while the Python import package is
`lvg_norm`.

```python
from lvg_norm import NormNormalizer, lvg_normalize

lvg_normalize("β-lactam antibiotics")
# ['antibiotic beta lactam']

normer = NormNormalizer(max_combinations=5)
normer.normalize("HNF1A p.Q125*")
# ['hnf1a p q125', 'hnf1on p q125', 'hnf1um p q125']

# Chemistry preset for IUPAC / small-molecule names
lvg_normalize("(2S,3R)-2,3-dihydro-1H-indole", pipeline="chemical")
# ['(2s,3r)-2,3-dihydro-1h-indole']
lvg_normalize("β-lactam antibiotics", pipeline="chemical")
# ['beta-lactam antibiotics']
lvg_normalize("(±)-tartaric acid", pipeline="chemical")
# ['(+/-)-tartaric acid']
```

## CLI

The package installs a `lvg-norm` command:

```bash
lvg-norm "β-lactam antibiotics"
lvg-norm --file inputs.txt
echo "HNF1A p.Q125*" | lvg-norm
```

Useful flags:

- `--pipeline {medical,chemical}` to pick the preset (default: `medical`)
- `--stopwords PATH` to provide an extra stopword list
- `--no-lvg-stopwords` to disable the bundled LVG stopword list
- `--max-combinations N` to cap variant expansion

## Development

```bash
uv sync --group dev
pytest
ruff check .
ruff format --check .
```
