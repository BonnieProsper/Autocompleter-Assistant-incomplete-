#!/usr/bin/env python3
"""
analytics.py - Small analytics helpers for the Autocompleter Assistant.

Functions:
 - top_words(data_file, n): returns top-n learned words by frequency.
 - read_eval_results(json_path): print evaluation results.

"""

from __future__ import annotations
import json
from pathlib import Path
from typing import List, Tuple, Dict


def top_words(data_file: Path, n: int = 20) -> List[Tuple[str, int]]:
    """
    Return a list of (word, frequency) top-n pairs from the persisted user_data.json.
    Expects a JSON with {"words": ["word1", "word2", ...]}.
    """
    if not data_file.exists():
        return []
    raw = json.loads(data_file.read_text(encoding="utf-8"))
    words = raw.get("words", [])
    # If words are unique list, counts are computed by frequency in list
    counts: Dict[str, int] = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n]


def read_eval_results(path: Path) -> Dict:
    """Load evaluation JSON and return it."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def print_eval_summary(eval_json: Dict):
    """Prints a compact evaluation summary to stdout."""
    if not eval_json:
        print("No evaluation results found.")
        return
    for mode in ("exact", "fuzzy"):
        mode_stats = eval_json.get(mode, {})
        hits = mode_stats.get("hits", 0)
        tot = mode_stats.get("total", 0)
        time_s = mode_stats.get("time", 0.0)
        acc = f"{(100.0 * hits / tot):.2f}%" if tot else "N/A"
        avg = (time_s / tot) if tot else 0.0
        print(f"{mode.title():6s} -- hits: {hits}/{tot} acc: {acc} time: {time_s:.3f}s avg/query: {avg:.6f}s")
