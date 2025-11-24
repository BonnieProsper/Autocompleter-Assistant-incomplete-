"""
reasoner.py
Core reasoning engine for autocompleter.
Performs:
 - rule-based safety checks
 - typo detection and corrections
 - next-argument prediction heuristics
 - plugin-driven reasoning extensions
 - semantic (AI) command prediction

It returns structured suggestions that the UI layer formats.
"""

from __future__ import annotations
import re
from typing import List, Dict, Any

from intelligent_autocompleter.core.semantic_engine import SemanticEngine

class ReasoningResult:
    """
    Represents a single reasoning insight.
    The UI layer chooses how to display these.
    """
    __slots__ = ("type", "message", "metadata")

    def __init__(self, type: str, message: str, metadata: Dict[str, Any] | None = None):
        self.type = type                # "warning", "suggestion", "prediction"
        self.message = message          # human-readable string
        self.metadata = metadata or {}  # optional extra info

    def __repr__(self):
        return f"ReasoningResult(type={self.type}, message={self.message})"

class CommandReasoner:
    """
    Reasoning layer that provides CLI command input
    with structured suggestions.
    It combines static rules, pattern-based heuristics, 
    plugin-provided reasoning, and semantic AI prediction
    """

    # Static rule definitions ---------------------------------------------------
    DANGEROUS_PATTERNS = [
        r"\brm\s+-rf\b",
        r"\bsudo\s+rm\b",
        r"\bmkfs\b",
        r"\bshutdown\b"
    ]

    COMMON_CORRECTIONS = {
        r"\bgti\b": "git",
        r"\bpip\s+instal\b": "pip install",
    }

    GIT_FOLLOWUPS = ["commit", "push", "pull", "status", "branch"]

    def __init__(self, plugin_registry):
        """
        plugin_registry must expose a `run_reasoning(input: str) -> List[ReasoningResult]` method.
        """
        self.registry = plugin_registry
        self.semantic = SemanticEngine()

    # ENTRY POINT ----------------------------------------------------------------
    def analyze_input(self, user_input: str) -> List[ReasoningResult]:
        insights: List[ReasoningResult] = []
        insights.extend(self._check_dangerous_patterns(user_input))
        insights.extend(self._suggest_corrections(user_input))
        insights.extend(self._predict_next_steps(user_input))
        insights.extend(self.registry.run_reasoning(user_input))  # plugin hooks
        insights.extend(self._semantic_prediction(user_input))
        return insights

    # RULE-BASED REASONING ---------------------------------------------------------
    def _check_dangerous_patterns(self, user_input: str) -> List[ReasoningResult]:
        results = []
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, user_input):
                results.append(
                    ReasoningResult(
                        type="warning",
                        message=f"Command looks dangerous: '{user_input}'",
                        metadata={"pattern": pattern},
                    )
                )
        return results

    def _suggest_corrections(self, user_input: str) -> List[ReasoningResult]:
        results = []
        for pattern, correction in self.COMMON_CORRECTIONS.items():
            if re.search(pattern, user_input):
                results.append(
                    ReasoningResult(
                        type="suggestion",
                        message=f"Did you mean '{correction}'?",
                        metadata={"correction": correction},
                    )
                )
        return results

    def _predict_next_steps(self, user_input: str) -> List[ReasoningResult]:
        if user_input.startswith("git "):
            return [
                ReasoningResult(
                    type="prediction",
                    message="Likely next git commands: " + ", ".join(self.GIT_FOLLOWUPS),
                    metadata={"options": self.GIT_FOLLOWUPS},
                )
            ]
        return []

    # SEMANTIC ASSISTED REASONING --------------------------------------------------
    def _semantic_prediction(self, user_input: str) -> List[ReasoningResult]:
        prediction = self.semantic.predict_command(user_input)
        if not prediction:
            return []
        return [
            ReasoningResult(
                type="semantic",
                message=f"AI Suggestion → {prediction}",
                metadata={"model": "SemanticEngine"},
            )
        ]

