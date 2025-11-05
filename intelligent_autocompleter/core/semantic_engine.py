# semantic_engine.py - semantic suggestion engine using pretrained embeddings
# tries gensim first then spaCy

import os, math, logging
from typing import List, Tuple

try:
    from gensim.models import KeyedVectors
    _GENSIM = True
except ImportError:
    _GENSIM = False

try:
    import spacy
    _SPACY = True
except ImportError:
    _SPACY = False


class SemanticEngine:
    # semantic word suggestion engine
    def __init__(self, model_path: str = None):
        self.model = None
        self.backend = None
        if model_path and _GENSIM:
            self._load_gensim(model_path)
        elif _SPACY:
            self._load_spacy()
        else:
            logging.warning("No semantic model available. Install gensim or spacy.")

    def _load_gensim(self, path: str):
        if not os.path.exists(path):
            logging.warning(f"Model not found at {path}")
            return
        try:
            self.model = KeyedVectors.load_word2vec_format(path, binary=True)
            self.backend = "gensim"
            logging.info(f"Loaded gensim model from {path}")
        except Exception as e:
            logging.warning(f"Failed to load gensim model: {e}")

    def _load_spacy(self):
        #load spacy model
        for name in ("en_core_web_md", "en_core_web_lg", "en_core_web_sm"):
            try:
                self.model = spacy.load(name)
                self.backend = "spacy"
                logging.info(f"Using spaCy model: {name}")
                return
            except Exception:
                continue
        logging.warning("No usable spaCy model found.")

    def similar(self, word: str, topn: int = 5) -> List[Tuple[str, float]]:
        # return list of semantically similar words
        if not self.model:
            return []

        elif self.backend == "gensim":
            if word not in self.model.key_to_index:
                return []
            sims = self.model.most_similar(word, topn=topn)
            return [(w, float(s)) for w, s in sims]

        elif self.backend == "spacy":
            tok = self.model.vocab.get(word)
            if not tok or not tok.has_vector:
                return []
            vec = tok.vector
            out = []
            for lex in self.model.vocab:
                if lex.is_alpha and lex.has_vector:
                    sim = self._cos(vec, lex.vector)
                    if sim > 0.4:  # cutoff
                        out.append((lex.text, sim))
            out.sort(key=lambda x: -x[1])
            return out[:topn]

        else:
            return []

    @staticmethod
    def _cos(a, b):
        # find cosine similarity between 2 vectors
        num = sum(x * y for x, y in zip(a, b))
        ma = math.sqrt(sum(x * x for x in a))
        mb = math.sqrt(sum(y * y for y in b))
        if not ma or not mb:
            return 0.0
        return num / (ma * mb)
