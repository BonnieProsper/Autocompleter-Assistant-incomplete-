# autocompleter.py
# Orchestration core to load/save model state, tie HybridPredictor to persistence, small API for the CLI.
# combines other files

import atexit
import time
from collections import defaultdict, Counter
from pathlib import Path

from hybrid_predictor import HybridPredictor
from markov_predictor import MarkovPredictor
from context_personal import CtxPersonal
from logger_utils import Log
from model_store import (
    load_markov,
    save_markov,
    load_user_cache,
    save_user_cache,
)


DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

class AutoCompleter:
    """
    Wrapper:
     - initialise HybridPredictor
     - restore saved markov table + user cache at start
     - saves on exit
     - simple api for cli/tests
    """

    def __init__(self, user="default"):
        self.user = user
        Log.write("Autocompleter init")
        # main predictive engine (combines markov, embeddings and personalization)
        self.hp = HybridPredictor(user=user)

        # keep a handle to raw markov predictor for persistence tweaks
        self.mk = self.hp.mk  # convenience alias - HybridPredictor has .mk as MarkovPredictor instance

        # try to restore saved data from disk (markov table, user history)
        self._load_persisted()
        
        # autosave on normal exit
        atexit.register(self._shutdown)
        self._started = time.time() # start timestamp

    # Persistence helper logic  ---------------------------
    def _load_persisted(self):
        # load saved markov state from disk (should be mapping prev -> {next: count} or prev -> list (for legacy saves))
        mdata = load_markov() or {}
        if not mdata:
            Log.write("No markov data found, starting from scratch.")
            return

        # convert loaded structure into MarkovPredictor's internal table
        # accept: {prev: {next: count}} or {prev: [next, next, ...]}
        # counters:
        # tbl : bigram transition frequencies, unig: unigram totals (e.g frequency of each token)
        tbl = defaultdict(Counter)
        unig = Counter()
        for prev, vals in mdata.items():
            if isinstance(vals, dict):
                for nxt, cnt in vals.items():
                    tbl[prev][nxt] += int(cnt)
                    unig[prev] += int(cnt)
            elif isinstance(vals, list):
                for nxt in vals:
                    tbl[prev][nxt] += 1
                    unig[prev] += 1
            else:
                # unknown shape, skip and ignore
                continue

        # apply data to MarkovPredictor
        try:
            # markov predictor uses _table: defaultdict(Counter) and _unigrams Counter
            self.mk._table = tbl
            self.mk._unigrams = unig
            Log.write(f"Restored markov data ({len(tbl)} keys)")
        except Exception as e:
            Log.write(f"Failed to restore markov into predictor: {e}")

        # load user personalisation cache if available
        ucache = load_user_cache() or {}
        try:
            # CtxPersonal stores history via self.hist. if empty, keep the existing history
            if ucache.get("history"):
                # convert list of phrases to per-word counts
                for phrase in ucache.get("history", []):
                    self.hp.ctx.learn(phrase)
                Log.write(f"Restored user history ({len(ucache.get('history', []))} items)")
        except Exception as e:
            Log.write(f"Failed to restore user cache: {e}")

    def _serialize_markov(self):
        """
        Convert MarkovPredictor's internal table to JSON-serializable plain dictionary:
        prev -> {next: count}
        """
        out = {}
        try:
            for prev, counter in self.mk._table.items():
                if isinstance(counter, Counter):
                    out[prev] = dict(counter)
                else:
                    #  if it's a list, convert to counter first
                    out[prev] = dict(Counter(counter))
        except Exception as e:
            Log.write(f"Error serialising Markov model: {e}")
        return out

    def _shutdown(self):
        """save models and user cache at exit"""
        Log.write("Shutdown: saving state to disk")

        # save markov chain by converting to prev -> {next: count}
        try:
            mdata = self._serialize_markov()
            save_markov(mdata)
        except Exception as e:
            Log.write(f"save_markov failed: {e}")

        # save user cache (store history from CtxPersonal if it is available (above))
        try:
            hist = []
            try:
                # try to find prior history from ctx.hist or hp.ctx.hist (Counter)
                if hasattr(self.hp.ctx, "hist"):
                    # expand into a list
                    # store fixed recent slice, top 200 most common entries to avoid too big files
                    for w, cnt in self.hp.ctx.hist.most_common(200):
                        # store the word > count times? store as word:count mapping
                        hist.append({"word": w, "count": int(cnt)})
                # object and metadata
                uobj = {
                    "history": hist,
                    "saved_at": int(time.time()),
                }
                save_user_cache(uobj)
                Log.write("User cache saved")
            except Exception:
                # fallback
                save_user_cache({"history": hist})
        except Exception as e:
            Log.write(f"Failed to save user cache: {e}")

    # API -------------------------------------
    def train_lines(self, lines):
        """Train model on list of lines (corpus.txt)."""
        self.hp.train(lines)

    def retrain(self, sentence):
        """Update incrimentally with new sentence."""
        self.hp.retrain(sentence)

    def suggest(self, prefix, topn=5):
        """Return suggestions from hybrid predictor based on prefix given."""
        return self.hp.suggest(prefix, topn=topn)

    def accept(self, word):
        """Called when user accepts a suggestion, model reinforces succesfull predictions as training"""
        self.hp.accept_suggestion(word)

    def stats(self):
        """Return a summary of stats."""
        return {
            "uptime_s": round(time.time() - self._started, 1),
            "vocab_size": len(list(self.hp.trie.iter_words_from(""))),
        }


# small manual test when run directly
if __name__ == "__main__":
    ac = AutoCompleter()
    print("Autocompleter loaded. Try ac.suggest('the') or ac.train_lines([...])")



