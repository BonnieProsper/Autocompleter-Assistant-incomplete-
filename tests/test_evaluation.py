# to test test_evaluation.py
import json
import tempfile
from pathlib import Path
from autocompleter import Autocompleter
from evaluation import build_model_from_sentences, evaluate_on_test

def test_build_and_eval_small_corpus(tmp_path):
    corpus = [
        "thank you very much",
        "please call me back",
        "the quick brown fox jumps over the lazy dog",
        "hello world this is a test",
        "i love programming in python"
    ]
    train = corpus[:3]
    test = corpus[3:]
    engine = build_model_from_sentences(train)
    stats = evaluate_on_test(engine, test)
    # stats keys must be present and totals non-negative integers
    assert "exact" in stats and "fuzzy" in stats
    assert isinstance(stats["exact"]["total"], int)
    assert isinstance(stats["fuzzy"]["total"], int)
