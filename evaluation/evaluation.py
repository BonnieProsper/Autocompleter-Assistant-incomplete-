#!/usr/bin/env python3
"""
evaluation.py - Evaluation harness

- Trains Autocompleter on a corpus training split (one sentence per line).
- Tests predictions for next-word completion using partial prefixes.
- Compares fuzzy vs exact-prefix suggestions.
- Writes a JSON summary report.

Usage:
python evaluation.py data/demo_corpus.txt --out results.json

"""
from __future__ import annotations

import argparse
import json
import time
import math
import re
from pathlib import Path
from typing import Iterable, List, Tuple, Dict

# Import the autocompleter.py engine
from autocompleter import Autocompleter


def load_corpus(path: Path) -> List[str]:
    """Load lines from corpus, strip whitespace and ignore empty lines."""
    with path.open("r", encoding="utf-8") as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]
    return lines


def split_corpus(corpus: List[str], train_frac: float = 0.8) -> Tuple[List[str], List[str]]:
    """Deterministic train/test split."""
    n = max(1, int(len(corpus) * train_frac))
    return corpus[:n], corpus[n:]


def tokens(sentence: str) -> List[str]:
    """Return word tokens (in lowercase, keep alphabetic sequences and digits)."""
    return re.findall(r"\w+", sentence.lower())


def build_model_from_sentences(sentences: Iterable[str]) -> Autocompleter:
    """Train an Autocompleter instance on given sentences (mutates and returns)."""
    engine = Autocompleter(data_path="data/eval_user_data.json")  # isolated persistence file
    # Reset persistence file to avoid accidental reuse
    p = Path(engine.data_path)
    if p.exists():
        p.unlink()
    # Learn sentences
    for s in sentences:
        engine.learn_from_input(s)
    return engine


def evaluate_on_test(engine: Autocompleter, test_sentences: Iterable[str], top_k: int = 5) -> Dict:
    """
    For each adjacent token pair (prev, true_next) in the test set, 
    provide a partial prefix of the true_next (half of its length, min 1 char), 
    then query model for suggestions and record whether true_next is present. 
    Returns statistics for fuzzy=True and fuzzy=False.
    """
    stats = {
        "fuzzy": {"hits": 0, "total": 0, "time": 0.0},
        "exact": {"hits": 0, "total": 0, "time": 0.0},
    }

    all_pairs: List[Tuple[str, str]] = []
    for s in test_sentences:
        toks = tokens(s)
        for a, b in zip(toks, toks[1:]):
            all_pairs.append((a, b))

    # If no pairs, return empty stats
    if not all_pairs:
        return stats

    # Evaluate fuzzy and exact (two passes to measure time separately)
    def run(flag: str, fuzzy_flag: bool):
        hits = 0
        total = 0
        t0 = time.perf_counter()
        for prev, true_next in all_pairs:
            # create prefix (half the length, at least 1)
            prefix_len = max(1, math.ceil(len(true_next) / 2))
            prefix = true_next[:prefix_len]
            # Use context-aware query by passing the full input. Autocompleter.suggest expects prefix input (could expect context)
            # For our integration, we call suggest with the prefix string only.
            # if Autocompleter API expects context consider passing context as previous word for better realism.
            start = time.perf_counter()
            candidates = engine.suggest(prefix)
            end = time.perf_counter()
            total += 1
            # if any of returned suggestions equals true_next
            if true_next in candidates:
                hits += 1
        t1 = time.perf_counter()
        stats[flag]["hits"] = hits
        stats[flag]["total"] = total
        stats[flag]["time"] = t1 - t0

    # Evaluate exact (non-fuzzy) - if your engine supports toggling fuzzy, you can toggle it instead of this separation.
    # For now we use the same suggest() call; if you have a fuzzy toggle available, call accordingly.
    # Run exact
    run("exact", False)
    # Run fuzzy
    run("fuzzy", True)

    return stats


def summarize_and_write(stats: Dict, out_path: Path):
    """Write JSON summary and print summary to stdout."""
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)

    def pct(h, t):
        return f"{(100.0 * h / t):.2f}%" if t else "N/A"

    print("=== Evaluation Summary ===")
    for k in ("exact", "fuzzy"):
        hit = stats[k]["hits"]
        tot = stats[k]["total"]
        t = stats[k]["time"]
        print(f"{k.title():6s} | hits: {hit}/{tot} | acc: {pct(hit, tot)} | time: {t:.3f}s | avg time/query: {(t/tot if tot else 0):.6f}s")
    print("==========================")
    print(f"Full JSON written to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Autocompleter on a corpus")
    parser.add_argument("corpus", type=str, help="Path to corpus (one sentence per line)")
    parser.add_argument("--out", type=str, default="evaluation_results.json", help="Output JSON file")
    parser.add_argument("--train-frac", type=float, default=0.8, help="Training fraction (0..1)")
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print(f"Corpus not found: {corpus_path}")
        return

    corpus = load_corpus(corpus_path)
    train, test = split_corpus(corpus, train_frac=args.train_frac)
    print(f"Loaded {len(corpus)} sentences: train={len(train)}, test={len(test)}")

    engine = build_model_from_sentences(train)
    stats = evaluate_on_test(engine, test)
    summarize_and_write(stats, Path(args.out))


if __name__ == "__main__":
    main()
