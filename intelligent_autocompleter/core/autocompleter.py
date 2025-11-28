# autocompleter.py
"""
AutoCompleter - application facade for the Intelligent Autocompleter (redundant due to facade in HybridPredictor/fusion ranker, cli and tui?).

Responsibilities:
 - Owns a HybridPredictor instance
 - Handles persistence hooks (Markov + user context)
 - API for CLI/TUI/tests:
     suggest(prefix), train_lines(lines), retrain(sentence), accept(word), stats()
 - Safe shutdown autosave (atexit), robust to missing persistence/backend implementations
 - Small logging wrapper to tolerate different Log implementations in repo
"""

from __future__ import annotations
import atexit
import time
from pathlib import Path
from typing import Iterable, List, Dict, Any

# Import project modules (support local dev layout fallback)
try:
    from intelligent_autocompleter.core.hybrid_predictor import HybridPredictor
    from intelligent_autocompleter.core.model_store import load_markov, save_markov, load_user_cache, save_user_cache
    from intelligent_autocompleter.utils.logger_utils import Log
except Exception:
    # local fallback
    from hybrid_predictor import HybridPredictor
    from model_store import load_markov, save_markov, load_user_cache, save_user_cache
    try:
        from logger_utils import Log  # your repo's logger_utils
    except Exception:
        Log = None  # fallback to print if missing

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def _log_write(msg: str) -> None:
    """Small compatibility wrapper.
    Many modules call Log.write(msg) but Log object may be instance-based."""
    try:
        if Log is None:
            print(msg)
        else:
            # prefer static method if available
            if hasattr(Log, "write") and callable(getattr(Log, "write")):
                # Some logger implementations expect (msg) or (level,msg). Try safe calls.
                try:
                    Log.write(msg)
                except TypeError:
                    # maybe Log.write expects (level,msg) or is an instance method
                    try:
                        Log().write("INFO", msg)
                    except Exception:
                        print(msg)
            else:
                # instantiate and call info/debug
                logger = Log()
                if hasattr(logger, "info"):
                    logger.info(msg)
                elif hasattr(logger, "write"):
                    logger.write(msg)
                else:
                    print(msg)
    except Exception:
        print(msg)


class AutoCompleter:
    """
    Facade for autocompleter.

    Public API:
      - suggest(prefix: str, topn: int=5) -> list[(word,score)]
      - train_lines(lines: Iterable[str]) -> None
      - retrain(sentence: str) -> None
      - accept(word: str, context: str|None = None) -> None
      - stats() -> dict for UI/tests
    """

    def __init__(self, user: str = "default", ranker_preset: str = "balanced"):
        self.user = user
        _log_write(f"[AutoCompleter] booting user='{user}'")

        # create predictor. HybridPredictor should gracefully handle missing optional dependencies.
        self.hp = HybridPredictor(user=user, ranker_preset=ranker_preset)

        # convenience pointer to Markov predictor (if exposed)
        self._mk = getattr(self.hp, "markov", getattr(self.hp, "_markov", None))

        # restore persisted state
        self._restore_state()

        # register autosave on normal exit
        atexit.register(self._shutdown)

        self._started_at = time.time()

    # Persistence helpers ------------------------------------
    def _restore_state(self) -> None:
        """Attempt to restore persisted model pieces."""
        try:
            raw_markov = load_markov()
            if raw_markov:
                # HybridPredictor may implement a load_state/mk.load_from_serialized - check 
                mk = getattr(self.hp, "markov", None) or getattr(self.hp, "_markov", None)
                if mk and hasattr(mk, "load_from_serialized"):
                    mk.load_from_serialized(raw_markov)
                    _log_write("[AutoCompleter] restored markov via mk.load_from_serialized")
                else:
                    # fallback: try set internals (older API)
                    if hasattr(self.hp, "_markov") and hasattr(self.hp._markov, "_table"):
                        # expect format {prev: {next: count}}
                        try:
                            # Accept both dict-of-dict and legacy list shapes
                            tbl = {}
                            for prev, vals in raw_markov.items():
                                if isinstance(vals, dict):
                                    tbl[prev] = dict(vals)
                                elif isinstance(vals, list):
                                    # convert list to counts
                                    c = {}
                                    for nxt in vals:
                                        c[nxt] = c.get(nxt, 0) + 1
                                    tbl[prev] = c
                            # assign into internal markov table if safe
                            try:
                                self.hp._markov._table = {k: __import__("collections").Counter(v) for k, v in tbl.items()}
                                _log_write(f"[AutoCompleter] restored markov ({len(tbl)} keys)")
                            except Exception:
                                _log_write("[AutoCompleter] markov restore encountered unexpected internal shape")
                        except Exception as e:
                            _log_write(f"[AutoCompleter] markov restore parse failed: {e}")
            else:
                _log_write("[AutoCompleter] no saved markov found; starting cold")
        except Exception as e:
            _log_write(f"[AutoCompleter] markov restore failed: {e}")

        try:
            raw_user = load_user_cache()
            if raw_user:
                ctx = getattr(self.hp, "ctx", None)
                if ctx and hasattr(ctx, "load") and callable(ctx.load):
                    try:
                        ctx.load(raw_user)
                        _log_write("[AutoCompleter] restored user context via ctx.load")
                    except Exception:
                        # fallback to applying history items
                        hist = raw_user.get("history", [])
                        for item in hist:
                            w = item.get("word")
                            cnt = int(item.get("count", 1))
                            for _ in range(cnt):
                                try:
                                    ctx.learn(w)
                                except Exception:
                                    pass
                        _log_write(f"[AutoCompleter] restored user history ({len(hist)} items)")
                else:
                    # fallback: attempt direct structure
                    hist = raw_user.get("history", [])
                    ctx = getattr(self.hp, "ctx", None)
                    if ctx and hasattr(ctx, "learn"):
                        for item in hist:
                            w = item.get("word")
                            cnt = int(item.get("count", 1))
                            for _ in range(cnt):
                                try:
                                    ctx.learn(w)
                                except Exception:
                                    pass
                    _log_write("[AutoCompleter] user context restored (best-effort)")
        except Exception as e:
            _log_write(f"[AutoCompleter] user restore failed: {e}")

    def _serialise_markov(self) -> Dict[str, Dict[str, int]]:
        """Serialize markov internals into a JSON-safe dict. Best-effort depending on markov API."""
        mk = getattr(self.hp, "markov", None) or getattr(self.hp, "_markov", None)
        out = {}
        try:
            if mk and hasattr(mk, "export_state"):
                return mk.export_state()
            # try common internal shapes
            table = getattr(mk, "_table", None) or getattr(mk, "table", None)
            if isinstance(table, dict):
                for prev, cnts in table.items():
                    try:
                        # Counter-like -> dict
                        out[prev] = dict(cnts)
                    except Exception:
                        # fallback: attempt iterate
                        out[prev] = {k: int(v) for k, v in (cnts.items() if hasattr(cnts, "items") else [])}
        except Exception as e:
            _log_write(f"[AutoCompleter] serialise_markov failed: {e}")
        return out

    def _shutdown(self) -> None:
        """Persist state on interpreter exit."""
        _log_write("[AutoCompleter] shutdown: saving state")
        try:
            save_markov(self._serialise_markov())
        except Exception as e:
            _log_write(f"[AutoCompleter] save_markov failed: {e}")

        try:
            ctx = getattr(self.hp, "ctx", None)
            if ctx:
                # prefer ctx.export_state/ctx.save, otherwise best-effort history shape
                try:
                    if hasattr(ctx, "export_state"):
                        save_user_cache(ctx.export_state())
                    elif hasattr(ctx, "export"):
                        save_user_cache(ctx.export())
                    elif hasattr(ctx, "hist") and hasattr(ctx.hist, "most_common"):
                        hist_items = [{"word": w, "count": int(c)} for w, c in ctx.hist.most_common(200)]
                        save_user_cache({"history": hist_items, "saved_at": int(time.time())})
                    else:
                        save_user_cache({"history": [], "saved_at": int(time.time())})
                    _log_write("[AutoCompleter] user context saved")
                except Exception as e:
                    _log_write(f"[AutoCompleter] save_user_cache failed: {e}")
        except Exception as e:
            _log_write(f"[AutoCompleter] shutdown user save failed: {e}")

    # Public API -------------------------------------------------------
    def suggest(self, prefix: str, topn: int = 5) -> List[Tuple[str, float]]:
        """Return top predictions for a prefix. Small wrapper to HybridPredictor.suggest."""
        try:
            return self.hp.suggest(prefix, topn=topn)
        except Exception as e:
            _log_write(f"[AutoCompleter] suggest failed: {e}")
            return []

    def train_lines(self, lines: Iterable[str]) -> None:
        """Bulk-train the internal predictors from a corpus."""
        try:
            lines = list(lines)
            self.hp.train(lines)
        except Exception as e:
            _log_write(f"[AutoCompleter] train_lines failed: {e}")

    def retrain(self, sentence: str) -> None:
        """Incremental learning from a newly entered sentence."""
        try:
            self.hp.retrain(sentence)
        except Exception as e:
            _log_write(f"[AutoCompleter] retrain failed: {e}")

    def accept(self, word: str, context: str | None = None, source: str | None = None) -> None:
        """Record that a suggestion was accepted."""
        try:
            # hybrid predictor exposes accept/reject
            if hasattr(self.hp, "accept"):
                self.hp.accept(word, context=context, source=source)
            elif hasattr(self.hp, "record_accept"):
                self.hp.record_accept(context or "", word, source)
            else:
                _log_write("[AutoCompleter] accept: no feedback API available on HybridPredictor")
        except Exception as e:
            _log_write(f"[AutoCompleter] accept failed: {e}")

    def stats(self) -> Dict[str, Any]:
        """Return simple diagnostics for UI/tests."""
        try:
            vocab_size = 0
            trie = getattr(self.hp, "trie", None)
            if trie and hasattr(trie, "size"):
                try:
                    vocab_size = trie.size()
                except Exception:
                    # fallback: iterate root-level
                    vocab_size = 0
            uptime = round(time.time() - getattr(self, "_started_at", time.time()), 1)
            return {"uptime_s": uptime, "vocab_size": vocab_size}
        except Exception:
            return {"uptime_s": 0.0, "vocab_size": 0}

# quick manual test
if __name__ == "__main__":
    ac = AutoCompleter()
    print("Try: ac.suggest('the')")
