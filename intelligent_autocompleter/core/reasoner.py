# intelligent_autocompleter/core/reasoner??/reasoner.py
"""
Reasoner pipeline for Intelligent Autocompleter.
Responsibilities:
 - rule-based safety checks for dangerous commands
 - typo detection/common corrections
 - next-argument/next-step heuristics (e.g git, docker, cli flags)
 - semantic suggestions via SemanticEngine with safety sanitisation
 - plugin hooks (pre and post reasoning)
 - returns structured ReasoningResult objects grouped by category

Kept as a single entrypoint that composes small helper components.
Plugins may provide 'run_reasoning(input: str) -> List[ReasoningResult]'
or be registered through the plugin_registry.py/registry.py (decide later) that exposes the same.
The pipeline favours deterministic, explainable outputs.
"""

from __future__ import annotations
import re
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Iterable, Callable

# try to import project logger and fallback to print
try:
    from intelligent_autocompleter.utils.logger_utils import Log
except Exception:

    class Log:  # minimal shim
        @staticmethod
        def write(msg: str):
            print(msg)


# try to import SemanticEngine and optionally other helpers
try:
    from intelligent_autocompleter.core.semantic_engine import SemanticEngine
except Exception:
    SemanticEngine = None  # pipeline will degrade gracefully

# optional fuzzy/typo helpers (BKTree). Not required but used if present.
try:
    from intelligent_autocompleter.core.bktree import BKTree, levenshtein  # type:ignore
except Exception:
    BKTree = None
    levenshtein = None

# plugin_registry is optional, pipeline will call its run_reasoning if available
try:
    from intelligent_autocompleter.plugins.registry import PluginRegistry  # type: ignore
except Exception:
    PluginRegistry = None


# -------------------------
# Structured return object
# -------------------------
@dataclass
class ReasoningResult:
    """
    A single reasoning insight produced by the pipeline.
    type: one of "warning", "suggestion", "prediction", "correction", "info", "semantic", "plugin"
    message: human-friendly message
    metadata: optional dict for structured data (e.g. {"pattern": r"...", "suggestion": "git"})
    """

    type: str
    message: str
    metadata: Dict[str, Any] | None = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "message": self.message,
            "metadata": self.metadata or {},
        }


# -------------------------
# Small helper components
# -------------------------
class RuleChecker:
    """Includes all static regex-based safety checks and rules."""

    DANGEROUS_PATTERNS = [
        r"\brm\s+-rf\b",
        r"\bsudo\s+rm\b",
        r"\bmkfs\b",
        r"\bshutdown\b",
        r"\bdd\s+if=",
    ]

    def __init__(self, patterns: Optional[Iterable[str]] = None):
        self.patterns = (
            list(patterns) if patterns is not None else list(self.DANGEROUS_PATTERNS)
        )
        # compile for speed
        self._compiled = [re.compile(p, flags=re.IGNORECASE) for p in self.patterns]

    def check(self, text: str) -> List[ReasoningResult]:
        out = []
        for pat, cre in zip(self.patterns, self._compiled):
            if cre.search(text):
                out.append(
                    ReasoningResult(
                        type="warning",
                        message=f"Command looks dangerous: '{text.strip()}'",
                        metadata={"pattern": pat},
                    )
                )
        return out


class TypoSuggester:
    """Suggest corrections for common typos and optionally use BK-tree for fuzzy matches."""

    COMMON_SUBS = {
        r"\bgti\b": "git",
        r"\bpip\s+instal\b": "pip install",
        r"\bconosle\b": "console",
        r"\bbranh\b": "branch",
    }

    def __init__(self, bk_tree: Optional[object] = None, max_edit: int = 1):
        self.bk = bk_tree
        self.max_edit = max_edit
        # precompile patterns
        self._patterns = [
            (re.compile(p, flags=re.IGNORECASE), v) for p, v in self.COMMON_SUBS.items()
        ]

    def suggest(self, text: str) -> List[ReasoningResult]:
        out = []
        for cre, rep in self._patterns:
            if cre.search(text):
                out.append(
                    ReasoningResult(
                        type="correction",
                        message=f"Possible typo: did you mean '{rep}'?",
                        metadata={"correction": rep},
                    )
                )
        # optional fuzzy single-word suggestions when BKTree available
        if self.bk:
            # inspect words and query bk for near matches
            words = [w for w in re.findall(r"\w+", text) if w.isalpha()]
            for w in words:
                try:
                    res = self.bk.query(w, max_dist=self.max_edit)
                except Exception:
                    res = []
                # pick best not identical candidate
                for cand, dist in res:
                    if cand.lower() != w.lower():
                        out.append(
                            ReasoningResult(
                                type="correction",
                                message=f"Fuzzy suggestion: '{w}' → '{cand}' (edits={dist})",
                                metadata={"from": w, "to": cand, "edits": dist},
                            )
                        )
                        break
        return out


class HeuristicPredictor:
    """
    Heuristics for next-argument/next-step suggestions.
    Deterministic rules for common CLIs (git, docker, python).
    """

    GIT_FOLLOWUPS = ["commit", "push", "pull", "status", "branch"]
    DOCKER_FOLLOWUPS = ["build", "run", "ps", "images", "logs"]

    def predict(self, text: str) -> List[ReasoningResult]:
        out = []
        s = text.strip()
        if s.startswith("git "):
            out.append(
                ReasoningResult(
                    type="prediction",
                    message="Likely next git commands: "
                    + ", ".join(self.GIT_FOLLOWUPS),
                    metadata={"options": self.GIT_FOLLOWUPS},
                )
            )
        elif s.startswith("docker "):
            out.append(
                ReasoningResult(
                    type="prediction",
                    message="Likely next docker subcommands: "
                    + ", ".join(self.DOCKER_FOLLOWUPS),
                    metadata={"options": self.DOCKER_FOLLOWUPS},
                )
            )
        elif s.startswith("python ") or s.endswith(".py"):
            out.append(
                ReasoningResult(
                    type="suggestion",
                    message="Consider running with `-m` or `-u` if intending packages or unbuffered I/O",
                    metadata={},
                )
            )
        return out


class SemanticBridge:
    """
    Wraps SemanticEngine to provide predictions, with sanitization against RuleChecker.
    Returns ReasoningResult(type='semantic') if successful.
    """

    def __init__(
        self,
        engine: Optional[SemanticEngine],
        rules: RuleChecker,
        min_confidence: float = 0.8,
    ):
        self.engine = engine
        self.rules = rules
        self.min_confidence = float(min_confidence)

    def predict(self, text: str) -> List[ReasoningResult]:
        if not self.engine:
            return []
        try:
            # infer intent with score if engine supports it
            intent_out = None
            score = None
            try:
                out = self.engine.infer_intent(text)
                # engine returns tuple (intent,score) or string
                if isinstance(out, tuple) and len(out) == 2:
                    intent_out, score = out
                elif isinstance(out, str):
                    intent_out, score = out, 1.0
            except Exception:
                # fallback to semantic predict_command
                try:
                    cmd = self.engine.predict_command(text)
                    if cmd:
                        return [
                            ReasoningResult(
                                type="semantic",
                                message=f"AI Suggestion → {cmd}",
                                metadata={"cmd": cmd},
                            )
                        ]
                except Exception:
                    return []

            # if intent is present and it maps to a command fetch it
            if intent_out:
                cmd_map = {}
                try:
                    cmd_map = getattr(self.engine, "intent_commands", {}) or {}
                except Exception:
                    cmd_map = {}
                if intent_out in cmd_map and (
                    score is None or score >= self.min_confidence
                ):
                    cmd = cmd_map[intent_out]
                    # run safety checks on command string before returning
                    for r in self.rules.check(cmd):
                        # if any dangerous pattern applies, refuse to suggest command
                        Log.write(
                            f"[Reasoner] semantic suggestion suppressed: matched dangerous pattern {r.metadata}"
                        )
                        return []
                    return [
                        ReasoningResult(
                            type="semantic",
                            message=f"AI Suggestion → {cmd}",
                            metadata={"intent": intent_out, "score": score},
                        )
                    ]

            # fallback: ask engine for best matching stored entry
            try:
                hits = self.engine.search(text, k=1)
                if hits:
                    entry, sim = hits[0]
                    # sanitize entry
                    for r in self.rules.check(entry):
                        Log.write(
                            f"[Reasoner] semantic fallback suppressed: matched dangerous pattern {r.metadata}"
                        )
                        return []
                    return [
                        ReasoningResult(
                            type="semantic",
                            message=f"AI Suggestion → {entry}",
                            metadata={"score": sim},
                        )
                    ]
            except Exception:
                pass

        except Exception as e:
            Log.write(f"[Reasoner] semantic bridge error: {e}")
        return []


# -------------------------
# Pipeline orchestrator
# -------------------------
class ReasonerPipeline:
    """
    Orchestrator. Compose the helper components above and expose a single
    API 'analyze_input(input_str)' that returns a dict of lists grouped by category.
    """

    def __init__(
        self,
        plugin_registry: Optional[object] = None,
        semantic_engine: Optional[SemanticEngine] = None,
        bk_tree: Optional[object] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        plugin_registry: optional plugin registry; must implement run_reasoning(input)->List[ReasoningResult]
        semantic_engine: optional SemanticEngine instance
        bk_tree: optional BKTree instance for fuzzy suggestions
        config: small tuning dict
        """
        self.plugins = plugin_registry
        self.rules = RuleChecker()
        self.typo = TypoSuggester(bk_tree=bk_tree)
        self.heuristics = HeuristicPredictor()
        self.semantic = SemanticBridge(
            semantic_engine,
            self.rules,
            min_confidence=(config or {}).get("semantic_min_confidence", 0.8),
        )
        # plugin hooks: simple callables that accept input and return List[ReasoningResult]
        self.pre_hooks: List[Callable[[str], List[ReasoningResult]]] = []
        self.post_hooks: List[
            Callable[[str, List[ReasoningResult]], List[ReasoningResult]]
        ] = []

    # optional plugin hook registration helpers
    def register_pre_hook(self, fn: Callable[[str], List[ReasoningResult]]):
        self.pre_hooks.append(fn)

    def register_post_hook(
        self, fn: Callable[[str, List[ReasoningResult]], List[ReasoningResult]]
    ):
        self.post_hooks.append(fn)

    def analyze_input(self, input_text: str) -> Dict[str, List[ReasoningResult]]:
        """
        Main pipeline. Returns dict with keys:
         warnings, corrections, predictions, suggestions, semantic, plugin
        Each value is a list[ReasoningResult].
        """
        now = time.time()
        input_text = (input_text or "").strip()
        if not input_text:
            return {}

        results: Dict[str, List[ReasoningResult]] = {
            "warnings": [],
            "corrections": [],
            "predictions": [],
            "suggestions": [],
            "semantic": [],
            "plugin": [],
            "info": [],
        }

        # early plugin hooks (allow plugin to short-circuit or add annotations)
        if self.plugins:
            try:
                # plugin_registry.run_reasoning should return list[ReasoningResult]
                plugin_res = self.plugins.run_reasoning(input_text)
                for r in plugin_res:
                    results.setdefault("plugin", []).append(r)
            except Exception as e:
                Log.write(f"[Reasoner] plugin pre-run error: {e}")

        for hook in self.pre_hooks:
            try:
                ph = hook(input_text) or []
                for r in ph:
                    results.setdefault(r.type, []).append(r)
            except Exception as e:
                Log.write(f"[Reasoner] pre-hook error: {e}")

        # rule-based safety checks
        try:
            warn = self.rules.check(input_text)
            for r in warn:
                results["warnings"].append(r)
        except Exception as e:
            Log.write(f"[Reasoner] rule check error: {e}")

        # typo/correction suggestions
        try:
            corr = self.typo.suggest(input_text)
            for r in corr:
                results["corrections"].append(r)
        except Exception as e:
            Log.write(f"[Reasoner] typo suggest error: {e}")

        # heuristics (git/docker etc.)
        try:
            heur = self.heuristics.predict(input_text)
            for r in heur:
                results["predictions"].append(r)
        except Exception as e:
            Log.write(f"[Reasoner] heuristics error: {e}")

        # semantic bridge (intent + fallback), only if not obviously dangerous
        # If rule checker already flagged warnings skip semantic suggestions to be safe
        if not results["warnings"]:
            try:
                sem = self.semantic.predict(input_text)
                for r in sem:
                    results["semantic"].append(r)
            except Exception as e:
                Log.write(f"[Reasoner] semantic step error: {e}")
        else:
            Log.write("[Reasoner] semantic suggestion skipped due to safety warnings")

        # late plugin hooks (post processing)
        if self.plugins:
            try:
                # allow plugins to modify final set as they like
                late = self.plugins.run_reasoning(input_text)
                for r in late:
                    results.setdefault("plugin", []).append(r)
            except Exception as e:
                Log.write(f"[Reasoner] plugin post-run error: {e}")

        for hook in self.post_hooks:
            try:
                out_add = hook(input_text, results) or []
                for r in out_add:
                    results.setdefault(r.type, []).append(r)
            except Exception as e:
                Log.write(f"[Reasoner] post-hook error: {e}")

        # compact/dedupe by message for presentation
        for k, lst in list(results.items()):
            seen = set()
            compacted = []
            for r in lst:
                key = (r.type, r.message)
                if key in seen:
                    continue
                seen.add(key)
                compacted.append(r)
            results[k] = compacted

        results["meta"] = {"took_s": round(time.time() - now, 4)}
        return results
