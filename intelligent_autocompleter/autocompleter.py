# autocompleter.py
# Autocompleter Assistant - an adaptive autocomplete system that uses Trie, BKTree, and ContextPredictor with persistence and CLI UX.

import json
import os
import sys
from pathlib import Path
from collections import Counter
from colorama import Fore, Style, init
from textwrap import dedent

from trie import Trie
from bktree import BKTree
from context_predictor import ContextPredictor
from embeddings import Embeddings
from markov_predictor import MarkovPredictor
from cache_utils import simple_lru, timed
from threaded_runner import run_parallel


init(autoreset=True)

class Autocompleter:
    """Model using prefix, fuzzy, and contextual predictions."""
    def __init__(self, data_path: str = "data/user_data.json"):
        self.trie = Trie()
        self.bktree = BKTree()
        self.context = ContextPredictor()
        self.data_path = Path(data_path)
        self.stats = Counter()
        self._load_data()

        self.emb = Embeddings()    # could call emb.load(path) in setup (if there is a model)
        self.markov = MarkovPredictor()
        # keep in-memory corpus list to train the markov quickly
        self._local_sentences = []
    
    # Persistence ------------------------------------------------
    def _load_data(self):
        """Load user data (words and context) from disk."""
        if not self.data_path.exists():
            self.data_path.parent.mkdir(parents=True, exist_ok=True)
            self.data_path.write_text(json.dumps({"words": []}, indent=2))
            return

        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for word in data.get("words", []):
            self.trie.insert(word)
            self.bktree.insert(word)

    def _save_word(self, word: str):
        """Persist a newly learned word to disk."""
        if not word.isalpha():
            return
        data = json.load(open(self.data_path, "r", encoding="utf-8"))
        if word not in data["words"]:
            data["words"].append(word)
            with open(self.data_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            
    # Learning and Suggestion ----------------------------------------
    def learn_from_input(self, text: str):
        """Learn words and context from user input."""
        words = [w.lower() for w in text.split() if w.isalpha()]
        for w in words:
            self.trie.insert(w)
            self.bktree.insert(w)
            self._save_word(w)
        self.context.learn(text)
        self.markov.train_sentence(text)
        self._local_sentences.append(text)

    def suggest(self, prefix: str):
        """Return a ranked list of suggestions (context + fuzzy + prefix)."""
        prefix = prefix.lower().strip()
        prefix_suggestions = self.trie.search_prefix(prefix)
        fuzzy_suggestions = self.bktree.query(prefix, max_dist=1)
        context_suggestions = self.context.predict(prefix)
        ranked = Counter()
        for w, freq in prefix_suggestions:
            ranked[w] += 3 + freq
        for w, dist in fuzzy_suggestions:
            ranked[w] += max(1, 2 - dist)
        for w in context_suggestions:
            ranked[w] += 2
        return [w for w, _ in ranked.most_common(5)]
        # OR - EXTEND OR REPLACE??
    
        prefix = text.strip().lower().split()[-1] if text.strip() else ""
        # prefix matches
        pref = [w for w, _ in self.trie.search_prefix(prefix)]
        # fuzzy matches
        fuzzy = [w for w, _ in self.bktree.query(prefix, max_dist=1)]
        # context / markov
        prev = text.strip().split()[-2] if len(text.split()) > 1 else None
        markov_cands = [w for w, _ in (self.markov.top_next(prev, topn=5) if prev else [])] if prev else []
        # embeddings
        emb_cands = []
        if use_emb and self.emb._loaded:
            emb_cands = [w for w, _ in self.emb.similar(prefix, topn=5)]
        # scoring -simple weighted counts
        score = {}
        def add(cand, w):
            score[cand] = score.get(cand, 0.0) + w
        for w in pref:
            add(w, 3.0)
        for w in fuzzy:
            add(w, 1.5)
        for w in markov_cands:
            add(w, 2.0)
        for w in emb_cands:
             add(w, 1.2)
         # sort by score then freq (if in trie)
         ranked = sorted(score.items(), key=lambda kv: (-kv[1], - (self.trie._find(kv[0]).frequency if self.trie._find(kv[0]) else 0)))
         return [w for w, _ in ranked][:topn]


    # Reinforcement and Stats --------------------------------------------------
    def accept_suggestion(self, word: str):
        """Reinforcement learning to boost the frequency of accepted words."""
        self.trie.insert(word)
        self.stats["accepted"] += 1

    def get_stats(self):
        """Return the usage statistics."""
        total_words = len(json.load(open(self.data_path, "r", encoding="utf-8"))["words"])
        return {
            "total_words": total_words,
            "accepted_suggestions": self.stats["accepted"],
        }

# CLI Interface ----------------------------------------------------------------------
class CLI:
    """Interactive command-line interface."""
    def __init__(self):
        self.engine = Autocompleter()

    def _banner(self):
        print(Fore.CYAN + Style.BRIGHT + dedent("""
             -----------------------------------------
                   AUTOCOMPLETER ASSISTANT v1.0          
              Adaptive learning, context & fuzzy logic.   
             -----------------------------------------
        """))

    def _help(self):
        print(Fore.YELLOW + dedent("""
            Commands:
              /help     Show this help message
              /stats    Show model statistics
              /quit     Exit the program
            Simply type any word or sentence to get suggestions.
        """))

    def run(self):
        self._banner()
        while True:
            try:
                text = input(Fore.GREEN + "You: ").strip()
                if not text:
                    continue
                if text.startswith("/"):
                    if text == "/quit":
                        print(Fore.CYAN + "Goodbye!")
                        break
                    elif text == "/help":
                        self._help()
                    elif text == "/stats":
                        stats = self.engine.get_stats()
                        print(Fore.BLUE + f"Total learned words: {stats['total_words']}")
                        print(Fore.BLUE + f"Accepted suggestions: {stats['accepted_suggestions']}")
                    else:
                        print(Fore.RED + "Unknown command. Try /help.")
                    continue

                # Learn and suggest
                self.engine.learn_from_input(text)
                suggestions = self.engine.suggest(text.split()[-1])
                if not suggestions:
                    print(Fore.MAGENTA + "No suggestions yet.")
                    continue

                print(Fore.CYAN + "Suggestions:", ", ".join(suggestions))
                accept = input(Fore.GREEN + "Accept suggestion? (y/n) ").strip().lower()
                if accept.startswith("y"):
                    chosen = suggestions[0]
                    self.engine.accept_suggestion(chosen)
                    print(Fore.YELLOW + f"✅ Learned preference for '{chosen}'")
            except (EOFError, KeyboardInterrupt):
                print(Fore.CYAN + "\nExiting Autocompleter Assistant. Bye!")
                sys.exit(0)

    def bench_suggest(engine, prefixes, runs=5):
        import time
         times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            for p in prefixes:
                engine.suggest(p)
            t1 = time.perf_counter()
            times.append((t1 - t0) / len(prefixes))
        return sum(times) / len(times)


if __name__ == "__main__":
    CLI().run()


