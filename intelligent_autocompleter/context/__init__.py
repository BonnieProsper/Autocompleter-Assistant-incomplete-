# intelligent_autocompleter/context/__init__.py
# imports components for managing and processing context

from .pipeline import ContextPipeline  # context processing pipeline
from .normalizer import normalize_text  # function that normalizes text
from .tokenizer import simple_tokenize  # tokenizer to split text into tokens
from .scorers import (
    score_uncertainty,
    score_repetition,
)  # score uncertainty/repetition in text to improve prediction confidence and redundancy

__all__ = [
    "ContextPipeline",
    "normalize_text",
    "simple_tokenize",
    "score_uncertainty",
    "score_repetition",
]
