# Autocompleter-Assistant

A self-learning autocompleter that uses a Trie, BK-tree and contextual bigram learning to improve its text suggestions over time.

---

## Features

- Prefix-based searching using Trie
- Fuzzy spelling allowance using BK-tree
- Learning in context using bigrams and reinforcement
- Adaptive fuzzy logic
- CLI interface, including output and statistics
- Evaluation harness used for performance comparisons

---

## Setup

```bash
git clone https://github.com/BonnieProsper/Autocompleter-Assistant.git
cd Autocompleter-Assistant
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
---

## Quickstart (developer)

Requirements: Poetry (v1.8+), Python 3.10

```bash
# clone
git clone git@github.com:<BonnieProsper>/<Autocompleter-Assistant-incomplete->.git 
cd <Autocompleter-Assistant-incomplete->

# install
poetry install

# run tests
poetry run pytest -q

# lint
poetry run ruff check .
poetry run black .

# run the small profiling harness
poetry run python tools/profile_suggest.py
```

## Running Tests

Install dependencies:

    pip install -r requirements.txt

Run the test suite:

    pytest -q

Run with coverage:

    pytest --cov=.

Run linting:

    ruff check .

Run type checks:

    mypy .

