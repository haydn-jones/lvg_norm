# lvg_norm
Python implementation of LVG Norm (https://lhncbc.nlm.nih.gov/LSG/Projects/lvg/current/docs/userDoc/tools/norm.html).

This package focuses on the `norm` flow from the NLM Lexical Tools. It bundles
the LVG-derived resources needed by the normalizer and exposes both a Python API
and a small CLI.

## What It Does

Given an input string, `lvg_norm` produces one or more normalized forms by
applying an LVG-inspired pipeline:

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
```

## CLI

The package installs a `lvg-norm` command:

```bash
lvg-norm "β-lactam antibiotics"
lvg-norm --file inputs.txt
echo "HNF1A p.Q125*" | lvg-norm
```

Useful flags:

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
