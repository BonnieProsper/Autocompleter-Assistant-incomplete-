# context_personal.py - context + user pref tracking

import json, os
from collections import Counter
try:
    import spacy
    NLP = spacy.load("en_core_web_sm")
except Exception:
    NLP = None


class CtxPersonal:
    # tracks user prefs + word contexts
    def __init__(self, user="default", data_dir="userdata"):
        self.user = user
        self.data_dir = data_dir
        self.hist = Counter()
        self.pos_hist = Counter()
        os.makedirs(data_dir, exist_ok=True)
        self._load()

    def _path(self):
        return os.path.join(self.data_dir, f"{self.user}.json")

    def _load(self):
        p = self._path()
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf8") as f:
                    d = json.load(f)
                self.hist.update(d.get("words", {}))
                self.pos_hist.update(d.get("pos", {}))
            except Exception:
                pass

    def save(self):
        d = {"words": dict(self.hist), "pos": dict(self.pos_hist)}
        with open(self._path(), "w", encoding="utf8") as f:
            json.dump(d, f)

    def learn(self, text: str):
        toks = text.split()
        for w in toks: self.hist[w.lower()] += 1
        if NLP:
            doc = NLP(text)
            for t in doc: self.pos_hist[t.pos_] += 1

    def bias_words(self, suggs):
        # boosts freq for user's common words
        if not self.hist: return suggs
        out = []
        for w, sc in suggs:
            boost = 1.0 + (self.hist[w] / (max(self.hist.values()) or 1)) * 0.2
            out.append((w, sc * boost))
        return out
