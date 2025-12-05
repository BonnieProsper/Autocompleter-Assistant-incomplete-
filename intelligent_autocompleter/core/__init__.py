"""
intelligent_autocompleter.core

The core engine powering the Autocompleter Assistant (remove/redundant?).
Contains:
 - statistical next-token models (MarkovPredictor)
 - semantic/embedding engines (SemanticEngine)
 - ranking and signal fusion logic (FusionRanker)
 - command reasoning and contextual intelligence (CommandReasoner)
 - central plugin/command registry (CommandRegistry)
 - learning, feedback, and personalization components
"""

from .markov_predictor import MarkovPredictor
from .semantic_engine import SemanticEngine
from .fusion_ranker import FusionRanker
from .reasoner import CommandReasoner, ReasoningResult
from .registry import CommandRegistry

__all__ = [
    "MarkovPredictor",
    "SemanticEngine",
    "FusionRanker",
    "CommandReasoner",
    "ReasoningResult",
    "CommandRegistry",
]

# Semantic version of the core module (updated automatically in release tooling)
__version__ = "0.1.0"
