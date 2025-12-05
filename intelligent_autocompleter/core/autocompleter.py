# autocompleter.py
"""
AutoCompleter - application facade.
 - check: (redundant due to facade in HybridPredictor/fusion ranker, cli and tui?).
 - remove local fallbacks in all files?

Purpose:
 - Own HybridPredictor instance
 - Load/save persisted pieces (Markov + user context)
 - Simple public API for UI/CLI/tests:
     suggest(prefix, topn), train_lines(lines), retrain(sentence), accept(word), rejecr(word), stats()
 - Autosave on atexit
 - Provide deterministic robust shutdown persistence and explicit save() API
"""

from __future__ import annotations
import atexit
import time
from typing import Iterable, List, Tuple, Dict, Any
from pathlib import Path

try:
    from intelligent_autocompleter.core.hybrid_predictor import HybridPredictor
    from intelligent_autocompleter.core.model_store import (
        load_markov,
        save_markov,
        load_user_cache,
        save_user_cache,
    )
    from intelligent_autocompleter.utils.logger_utils import Log
except Exception:
    # Local fallback for tests etc, type: ignore
    from core.hybrid_predictor import HybridPredictor
    from model_store import load_markov, save_markov, load_user_cache, save_user_cache
    from logger_utils import Log

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def _log(msg: str) -> None:
    # compatibility wrapper: many diff Log variants exist in repo
    try:
        if hasattr(Log, "write"):
            Log.write(msg)
        else:
            logger = Log()
            if hasattr(logger, "info"):
                logger.info(msg)
            else:
                print(msg)
    except Exception:
        print(msg)


class AutoCompleter:
    """Application facade exposing small API
    Public API:
      - suggest(prefix: str, topn: int = 5) -> List[(word, score)]
      - train_lines(lines: Iterable[str]) -> None
      - retrain(sentence: str) -> None
      - accept(word: str, context: Optional[str]=None, source: Optional[str]=None) -> None
      - reject(word: str, etc)
      - save() -> None
      - stats() -> Dict[str, Any]
    """

    def __init__(self, user: str = "default", ranker_preset: str = "balanced"):
        self.user = user
        _log(f"[AutoCompleter] booting user='{user}'")
        # build predictor with dependency injection options
        self.hp = HybridPredictor(user=user, ranker_preset=ranker_preset)
        self._started_at = time.time()
        self._restore_state()  # restore persisted state
        atexit.register(self._shutdown)

    # Persistence ---------------------------------------------------------
    def _restore_state(self) -> None:
        try:
            raw_markov = load_markov()
            if raw_markov:
                try:
                    # try hybrid predictor adapter
                    if hasattr(self.hp, "load_state") and callable(
                        getattr(self.hp, "load_state")
                    ):
                        # Implementations expect path or payload
                        # attempt to load markov into predictor via load_state if available
                        # first try structured load_state: many versions accept a path for full state
                        # fall back to set internals if load_state not appropriate
                        if hasattr(self.hp, "markov") and hasattr(
                            self.hp.markov, "load_state"
                        ):
                            self.hp.markov.load_state({"chain": raw_markov})
                            _log(
                                "[AutoCompleter] markov restored via markov.load_state()"
                            )
                        else:
                            # attempt to set _markov internals
                            if hasattr(self.hp, "markov") and hasattr(
                                self.hp.markov, "train_sentence"
                            ):
                                # can retrain quickly from serialized table if needed, skip heavy restoration
                                _log(
                                    "[AutoCompleter] markov present (skipping full restore)."
                                )
                except Exception as e:
                    _log(f"[AutoCompleter] markov restore failed: {e}")
            else:
                _log("[AutoCompleter] no saved markov; cold start")
        except Exception as e:
            _log(f"[AutoCompleter] markov restore read failed: {e}")

        try:
            raw_user = load_user_cache()
            if raw_user:
                try:
                    ctx = getattr(self.hp, "ctx", None)
                    if ctx and hasattr(ctx, "load"):
                        try:
                            ctx.load(raw_user)  # if custom loader
                            _log("[AutoCompleter] user context restored via ctx.load")
                        except Exception:
                            # fall back to feeding tokens into ctx.learn
                            history = raw_user.get("history", [])
                            for item in history:
                                w = item.get("word")
                                cnt = int(item.get("count", 1))
                                for _ in range(cnt):
                                    try:
                                        ctx.learn(w)
                                    except Exception:
                                        pass
                            _log("[AutoCompleter] user context restored (fallback)")
                    else:
                        # fallback: apply history
                        history = raw_user.get("history", [])
                        ctx = getattr(self.hp, "ctx", None)
                        if ctx and hasattr(ctx, "learn"):
                            for item in history:
                                w = item.get("word")
                                cnt = int(item.get("count", 1))
                                for _ in range(cnt):
                                    try:
                                        ctx.learn(w)
                                    except Exception:
                                        pass
                            _log("[AutoCompleter] user history applied (best-effort)")
                except Exception as e:
                    _log(f"[AutoCompleter] user restore failed: {e}")
        except Exception as e:
            _log(f"[AutoCompleter] load_user_cache failed: {e}")

    def _serialise_markov(self) -> Dict[str, Dict[str, int]]:
        mk = getattr(self.hp, "markov", None)
        out: Dict[str, Dict[str, int]] = {}
        if not mk:
            return out
        # prefer explicit export_state/save_state
        try:
            if hasattr(mk, "save_state"):
                return mk.save_state()
            if hasattr(mk, "export_state"):
                return mk.export_state()
        except Exception:
            pass

        # inspect internals
        table = (
            getattr(mk, "_chain", None)
            or getattr(mk, "_table", None)
            or getattr(mk, "table", None)
        )
        if isinstance(table, dict):
            for prev, cnts in table.items():
                try:
                    out[prev] = dict(cnts)
                except Exception:
                    try:
                        out[prev] = {
                            k: int(v)
                            for k, v in (cnts.items() if hasattr(cnts, "items") else [])
                        }
                    except Exception:
                        out[prev] = {}
        return out

    def _shutdown(self) -> None:
        _log("[AutoCompleter] shutdown: persisting state")
        try:
            save_markov(self._serialise_markov())
        except Exception as e:
            _log(f"[AutoCompleter] save_markov failed: {e}")

        try:
            ctx = getattr(self.hp, "ctx", None)
            if ctx:
                try:
                    if hasattr(ctx, "export_state"):
                        save_user_cache(ctx.export_state())
                    elif hasattr(ctx, "save"):
                        ctx.save()
                        _log("[AutoCompleter] ctx.save() called")
                    else:
                        # fallback to history list
                        if hasattr(ctx, "hist") and hasattr(ctx.hist, "most_common"):
                            hist_items = [
                                {"word": w, "count": int(c)}
                                for w, c in ctx.hist.most_common(200)
                            ]
                            save_user_cache(
                                {"history": hist_items, "saved_at": int(time.time())}
                            )
                        else:
                            save_user_cache(
                                {"history": [], "saved_at": int(time.time())}
                            )
                    _log("[AutoCompleter] user context saved")
                except Exception as e:
                    _log(f"[AutoCompleter] save_user_cache failed: {e}")
        except Exception as e:
            _log(f"[AutoCompleter] shutdown user save failed: {e}")

    # Public API ---------------------------------------------------------
    def suggest(self, prefix: str, topn: int = 5) -> List[Tuple[str, float]]:
        try:
            return self.hp.suggest(prefix, topn=topn)
        except Exception as e:
            _log(f"[AutoCompleter] suggest failed: {e}")
            return []

    def train_lines(self, lines: Iterable[str]) -> None:
        try:
            self.hp.train(list(lines))
        except Exception as e:
            _log(f"[AutoCompleter] train_lines failed: {e}")

    def retrain(self, sentence: str) -> None:
        try:
            self.hp.retrain(sentence)
        except Exception as e:
            _log(f"[AutoCompleter] retrain failed: {e}")

    def accept(
        self, word: str, context: str | None = None, source: str | None = None
    ) -> None:
        try:
            if hasattr(self.hp, "accept"):
                self.hp.accept(word, context=context, source=source)
            else:
                _log("[AutoCompleter] accept: HybridPredictor missing accept API")
        except Exception as e:
            _log(f"[AutoCompleter] accept failed: {e}")

    def reject(
        self, word: str, context: str | None = None, source: str | None = None
    ) -> None:
        try:
            if hasattr(self.hp, "reject"):
                self.hp.reject(word, context=context, source=source)
        except Exception as e:
            _log(f"[AutoCompleter] reject failed: {e}")

    def stats(self) -> Dict[str, Any]:
        try:
            uptime = round(time.time() - self._started_at, 1)
            vocab_size = 0
            mk = getattr(self.hp, "markov", None)
            if mk and hasattr(mk, "vocabulary_size"):
                try:
                    vocab_size = mk.vocabulary_size()
                except Exception:
                    vocab_size = 0
            return {"uptime_s": uptime, "vocab_size": vocab_size}
        except Exception:
            return {"uptime_s": 0.0, "vocab_size": 0}


# quick manual test
if __name__ == "__main__":
    ac = AutoCompleter()
    print("Try: ac.suggest('the')")
