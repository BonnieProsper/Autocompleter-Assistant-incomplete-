# autocompleter.py
"""
AutoCompleter - application facade.
 - check: (redundant due to facade in HybridPredictor/fusion ranker, cli and tui?).
 - remove local fallbacks in all files?

Purpose:
 - Own HybridPredictor instance
 - Load/save persisted pieces (Markov + user context)
 - Provide a stable, simple public API for UI/CLI/tests:
     suggest(prefix, topn), train_lines(lines), retrain(sentence), accept(word), stats()
 - Provide deterministic robust shutdown persistence and explicit save() API
"""

from __future__ import annotations
from typing import Iterable, List, Tuple, Dict, Any, Optional
import atexit
import time
from pathlib import Path

# resilient imports (support package vs local dev)
try:
    from intelligent_autocompleter.core.hybrid_predictor import HybridPredictor
    from intelligent_autocompleter.core.model_store import load_markov, save_markov, load_user_cache, save_user_cache
    from intelligent_autocompleter.utils.logger_utils import Log
except Exception:
    try:
        # local fallbacks
        from hybrid_predictor import HybridPredictor  # type: ignore
        from model_store import load_markov, save_markov, load_user_cache, save_user_cache  # type: ignore
        from logger_utils import Log  # type: ignore
    except Exception:
        # minimal fallback logger (check: extend/take out?)
        class _SimpleLog:
            @staticmethod
            def write(msg: str):
                print(msg)
            @staticmethod
            def metric(*args, **kwargs):
                pass
            @staticmethod
            def time_block(label):
                class _T:
                    def __enter__(self): return self
                    def __exit__(self, exc_type, exc, tb): pass
                return _T()
        Log = _SimpleLog  # type: ignore

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def _log(msg: str) -> None:
    """Compatibility wrapper to call a project's Log facility safely."""
    try:
        # many Log implementations expose static write()
        if hasattr(Log, "write") and callable(getattr(Log, "write")):
            Log.write(msg)
        else:
            inst = Log()
            if hasattr(inst, "info"):
                inst.info(msg)
            elif hasattr(inst, "write"):
                inst.write(msg)
            else:
                print(msg)
    except Exception:
        try:
            print(msg)
        except Exception:
            pass


class AutoCompleter:
    """
    Facade around HybridPredictor.

    Public API:
      - suggest(prefix: str, topn: int = 5) -> List[(word, score)]
      - train_lines(lines: Iterable[str]) -> None
      - retrain(sentence: str) -> None
      - accept(word: str, context: Optional[str]=None, source: Optional[str]=None) -> None
      - save() -> None
      - stats() -> Dict[str, Any]
    """

    def __init__(self, user: str = "default", ranker_preset: str = "balanced", autosave: bool = True):
        self.user = user
        _log(f"[AutoCompleter] initializing user={user!r} preset={ranker_preset!r}")
        # Build predictor with dependency injection options
        self.hp = HybridPredictor(user=user, ranker_preset=ranker_preset)
        # convenience pointer
        self._markov = getattr(self.hp, "markov", None) or getattr(self.hp, "_markov", None)
        # restore persisted state
        self._restore_state()
        self._started_at = time.time()
        self._closed = False
        if autosave:
            atexit.register(self._shutdown)

    # Persistence --------------------------------------------
    def _restore_state(self) -> None:
        """Restore markov and user context using model_store helpers (best-effort)."""
        try:
            raw_markov = load_markov()
            if raw_markov:
                # Prefer an explicit loader on markov if available
                mk = getattr(self.hp, "markov", None) or getattr(self.hp, "_markov", None)
                if mk and hasattr(mk, "load_state"):
                    try:
                        mk.load_state(raw_markov)
                        _log("[AutoCompleter] markov loaded via mk.load_state")
                        return
                    except Exception:
                        _log("[AutoCompleter] mk.load_state failed, falling back")
                # Fallback: if HybridPredictor exposes load_state
                if hasattr(self.hp, "load_state"):
                    try:
                        self.hp.load_state(raw_markov)
                        _log("[AutoCompleter] hybrid.load_state used")
                        return
                    except Exception:
                        _log("[AutoCompleter] hybrid.load_state failed, fallback to internal assignment")
                # final attempt: attempt to coerce into mk._table if present
                if mk and hasattr(mk, "_table"):
                    try:
                        from collections import Counter
                        table = {}
                        for prev, vals in raw_markov.items():
                            if isinstance(vals, dict):
                                table[prev] = Counter({k: int(v) for k, v in vals.items()})
                            elif isinstance(vals, list):
                                c = Counter(vals)
                                table[prev] = c
                        mk._table = table
                        _log("[AutoCompleter] markov table restored.")
                    except Exception as e:
                        _log(f"[AutoCompleter] failed to set markov internals: {e}")
            else:
                _log("[AutoCompleter] no markov snapshot found")
        except Exception as e:
            _log(f"[AutoCompleter] markov restore error: {e}")

        # user context
        try:
            raw_user = load_user_cache()
            if raw_user:
                ctx = getattr(self.hp, "ctx", None)
                if ctx:
                    # prefer ctx.load or ctx._load-like API
                    if hasattr(ctx, "load"):
                        try:
                            ctx.load(raw_user)
                            _log("[AutoCompleter] user context loaded via ctx.load")
                            return
                        except Exception:
                            _log("[AutoCompleter] ctx.load failed. Falling back")
                    if hasattr(ctx, "learn"):
                        # fallback: apply history list items
                        hist = raw_user.get("history", [])
                        for item in hist:
                            try:
                                w = item.get("word")
                                cnt = int(item.get("count", 1))
                                for _ in range(cnt):
                                    ctx.learn(w)
                            except Exception:
                                pass
                        _log(f"[AutoCompleter] restored user history ({len(hist)} items)")
                    else:
                        _log("[AutoCompleter] ctx present but has no load/learn, skipping user restore")
        except Exception as e:
            _log(f"[AutoCompleter] user restore error: {e}")

    def _serialise_markov(self) -> Dict[str, Dict[str, int]]:
        """Serialize the markov predictor into a JSON-compatible dict."""
        mk = getattr(self.hp, "markov", None) or getattr(self.hp, "_markov", None)
        out: Dict[str, Dict[str, int]] = {}
        if not mk:
            return out
        # prefer export_state/save_state/save_state style APIs
        try:
            if hasattr(mk, "export_state"):
                return mk.export_state()
            if hasattr(mk, "save_state"):
                return mk.save_state()
        except Exception:
            # continue to internal probing
            pass
        # probe internals
        table = getattr(mk, "_table", None) or getattr(mk, "table", None)
        if isinstance(table, dict):
            for prev, counts in table.items():
                try:
                    out[prev] = dict(counts)
                except Exception:
                    # try iterate pairs
                    try:
                        out[prev] = {k: int(v) for k, v in (counts.items() if hasattr(counts, "items") else [])}
                    except Exception:
                        out[prev] = {}
        return out

    def _shutdown(self) -> None:
        """Called at interpreter exit (registered via atexit)."""
        # allow idempotent close
        if getattr(self, "_closed", False):
            return
        _log("[AutoCompleter] shutdown: persisting state")
        try:
            save_markov(self._serialise_markov())
        except Exception as e:
            _log(f"[AutoCompleter] save_markov failed: {e}")
        try:
            ctx = getattr(self.hp, "ctx", None)
            if ctx:
                # prefer export_state/save
                try:
                    if hasattr(ctx, "export_state"):
                        save_user_cache(ctx.export_state())
                    elif hasattr(ctx, "save"):
                        # some ctx.save writes directly, still provide consistent cache shape
                        ctx.save()
                        # try to call export_state afterwards if present
                        if hasattr(ctx, "export_state"):
                            save_user_cache(ctx.export_state())
                        else:
                            save_user_cache({"history": [], "saved_at": int(time.time())})
                    else:
                        # fallback to hist-like structure
                        if hasattr(ctx, "hist") and hasattr(ctx.hist, "most_common"):
                            hist_items = [{"word": w, "count": int(c)} for w, c in ctx.hist.most_common(200)]
                            save_user_cache({"history": hist_items, "saved_at": int(time.time())})
                        else:
                            save_user_cache({"history": [], "saved_at": int(time.time())})
                    _log("[AutoCompleter] user context saved")
                except Exception as e:
                    _log(f"[AutoCompleter] save_user_cache failed: {e}")
        except Exception as e:
            _log(f"[AutoCompleter] shutdown error: {e}")
        finally:
            self._closed = True

    # Public API --------------------------------------------
    def suggest(self, prefix: str, topn: int = 5) -> List[Tuple[str, float]]:
        """Return top predictions for a prefix via HybridPredictor.suggest."""
        try:
            return self.hp.suggest(prefix, topn=topn)
        except Exception as e:
            _log(f"[AutoCompleter] suggest error: {e}")
            return []

    def train_lines(self, lines: Iterable[str]) -> None:
        """Train internal predictors with corpus lines."""
        try:
            self.hp.train(list(lines))
        except Exception as e:
            _log(f"[AutoCompleter] train_lines error: {e}")

    def retrain(self, sentence: str) -> None:
        """Incremental learning from a new sentence."""
        try:
            self.hp.retrain(sentence)
        except Exception as e:
            _log(f"[AutoCompleter] retrain error: {e}")

    def accept(self, word: str, context: Optional[str] = None, source: Optional[str] = None) -> None:
        """Record acceptance of a suggestion. Delegates to HybridPredictor.accept or Reinforcement API."""
        try:
            if hasattr(self.hp, "accept"):
                self.hp.accept(word, context=context, source=source)
            elif hasattr(self.hp, "record_accept"):
                self.hp.record_accept(context or "", word, source)
            else:
                _log("[AutoCompleter] accept: HybridPredictor has no accept API")
        except Exception as e:
            _log(f"[AutoCompleter] accept error: {e}")

    def save(self) -> None:
        """Explicitly persist state (same logic as atexit)."""
        try:
            self._shutdown()
        except Exception as e:
            _log(f"[AutoCompleter] save failed: {e}")

    def stats(self) -> Dict[str, Any]:
        """Return diagnostic info (uptime, vocab size if available)."""
        uptime = round(time.time() - getattr(self, "_started_at", time.time()), 1)
        vocab_size = 0
        try:
            trie = getattr(self.hp, "trie", None)
            if trie and hasattr(trie, "size"):
                vocab_size = trie.size()
            else:
                # fallback: if markov provides a vocabulary_size or uni counter
                mk = getattr(self.hp, "markov", None) or getattr(self.hp, "_markov", None)
                if mk and hasattr(mk, "vocabulary_size"):
                    vocab_size = mk.vocabulary_size()
                elif mk and hasattr(mk, "_uni"):
                    vocab_size = len(getattr(mk, "_uni", []))
        except Exception:
            vocab_size = 0
        return {"uptime_s": uptime, "vocab_size": int(vocab_size)}

    # context manager support
    def close(self) -> None:
        """Explicit close equivalent to shutdown(); idempotent."""
        self._shutdown()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


# quick manual test
if __name__ == "__main__":
    ac = AutoCompleter()
    print("Try: ac.suggest('the')")

