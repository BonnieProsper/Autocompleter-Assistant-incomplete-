# autocompleter.py
# Orchestration layer for the Autocompleter system.
# Purpose:
# - own HybridPredictor instance
# - load persisted Markov transitions + user history
# - friendly API for CLI/TUI/tests
# - manage autosave + lifecycle shutdown (is this needed e.g is already in cli?/tui?)
# intentionally avoids algorithmic details, those go inside predictors and context modules.

import atexit
import time
from pathlib import Path
from collections import defaultdict, Counter

from hybrid_predictor import HybridPredictor
from logger_utils import Log
from model_store import (load_markov, save_markov, load_user_cache, save_user_cache,)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

class AutoCompleter:
    """
    Main application façade.
    This wraps the full predictive stack and provides a clean API:
     - suggest(prefix)
     - train_lines(lines)
     - retrain(sentence)
     - accept(word)
     - stats()
    It also handles persistence and session lifecycle.
    """
    def __init__(self, user: str = "default"):
        self.user = user
        Log.write(f"[init] Autocompleter boot for user='{user}'")

        # HybridPredictor combines Markov, semantic, trie, etc
        self.hp = HybridPredictor(user=user)

        # Convenient pointer for persistence routines
        self._mk = self.hp.mk
        # Load data from previous sessions
        self._restore_persisted_state()
        # Register autosave hook
        atexit.register(self._shutdown)

        self._started_at = time.time()

    # Load persisted data (Markov transitions + personalization) -----------------------------
    def _restore_persisted_state(self):
        """Restore Markov transitions and personalization history."""
        raw_markov = load_markov()
        if not raw_markov:
            Log.write("[restore] No markov model found; cold start.")
        else:
            self._apply_markov(raw_markov)

        raw_user = load_user_cache()
        if raw_user:
            self._apply_user_history(raw_user)

    def _apply_markov(self, data):
        """
        Convert persisted Markov JSON structure into actual
        defaultdict(Counter) tables used internally.
        """
        tbl = defaultdict(Counter)
        unigrams = Counter()
        for prev, vals in data.items():
            if isinstance(vals, dict):
                # standard shape: {next: count}
                for nxt, cnt in vals.items():
                    cnt = int(cnt)
                    tbl[prev][nxt] += cnt
                    unigrams[prev] += cnt
            elif isinstance(vals, list):
                # legacy shape: list of next tokens
                for nxt in vals:
                    tbl[prev][nxt] += 1
                    unigrams[prev] += 1
            else:
                # unknown data format, skip
                continue

        # write into MarkovPredictor internals
        try:
            self._mk._table = tbl
            self._mk._unigrams = unigrams
            Log.write(f"[restore] Markov rebuild complete ({len(tbl)} states).")
        except Exception as e:
            Log.write(f"[restore] Failed to load Markov predictor: {e}")

    def _apply_user_history(self, cache):
        """Restore user personalization history into CtxPersonal."""
        hist = cache.get("history", [])
        if not hist:
            return

        # history format: [{"word": w, "count": n}, ...]
        ctx = self.hp.ctx

        try:
            for item in hist:
                w = item.get("word")
                cnt = int(item.get("count", 1))
                for _ in range(cnt):
                    ctx.learn(w)
            Log.write(f"[restore] User history restored ({len(hist)} words).")
        except Exception as e:
            Log.write(f"[restore] Failed to load user history: {e}")

    # Persistence (on shutdown) --------------------------------------------------------------
    def _serialise_markov(self):
        """Convert Markov tables into JSON-safe shape."""
        out = {}
        for prev, counter in self._mk._table.items():
            if isinstance(counter, Counter):
                out[prev] = dict(counter)
            else:
                # convert any odd shapes to a Counter
                out[prev] = dict(Counter(counter))
        return out

    def _shutdown(self):
        """Automatically invoked during normal interpreter exit."""
        Log.write("[shutdown] Saving state to disk...")

        # Save markov transitions
        try:
            save_markov(self._serialise_markov())
        except Exception as e:
            Log.write(f"[shutdown] save_markov failed: {e}")

        # Save personalization history
        try:
            ctx = self.hp.ctx
            hist_items = []
            if hasattr(ctx, "hist"):
                for word, cnt in ctx.hist.most_common(200):
                    hist_items.append({"word": word, "count": int(cnt)})
            save_user_cache({
                "history": hist_items,
                "saved_at": int(time.time()),
            })
            Log.write("[shutdown] User history saved.")
        except Exception as e:
            Log.write(f"[shutdown] save_user_cache failed: {e}")

    # Public API (used by CLI, TUI, tests etc) -----------------------------------------------
    def suggest(self, prefix: str, topn: int = 5):
        """Return top predictions for a given prefix."""
        return self.hp.suggest(prefix, topn=topn)

    def train_lines(self, lines):
        """Bulk-train using corpus lines."""
        return self.hp.train(lines)

    def retrain(self, sentence):
        """Incrementally learn from a newly accepted line."""
        return self.hp.retrain(sentence)

    def accept(self, word):
        """Reinforce a successful prediction."""
        return self.hp.accept_suggestion(word)

    def stats(self):
        """Basic diagnostics for UI or tests."""
        return {
            "uptime_s": round(time.time() - self._started_at, 1),
            "vocab_size": len(list(self.hp.trie.iter_words_from(""))),
        }

# small manual test
if __name__ == "__main__":
    ac = AutoCompleter()
    print("Try: ac.suggest('the')")

