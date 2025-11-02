# embeddings.py - embeddings helper.
# Try gensim first, then spaCy, if neither then fall back.

from typing import List, Tuple, Optional
import math

GENSIM_OK = False
SPACY_OK = False
_model = None
_spacy_nlp = None

try:
    from gensim.models import KeyedVectors
    GENSIM_OK = True
except Exception:
    GENSIM_OK = False

if not GENSIM_OK:
    try:
        import spacy
        SPACY_OK = True
    except Exception:
        SPACY_OK = False


class Embeddings:
    """Wrapper for a word vector model. Methods return list of (word,score)."""

    def __init__(self, path: str = None):
        """
        path: optional path to a gensim KeyedVectors bin or text model.
        If no path and gensim available, user calls load().
        """
        self._model = None
        self._loaded = False
        if path:
            self.load(path)

    def load(self, path: str):
        if GENSIM_OK:
            # gensim fast load
            self._model = KeyedVectors.load_word2vec_format(path, binary=True)
            self._loaded = True
            return
        if SPACY_OK:
            # load medium english vectors if available
            try:
                self._model = spacy.load("en_core_web_md")
                self._loaded = True
                return
            except Exception:
                # try the small pipeline as last resort, this has no vectors
                try:
                    self._model = spacy.load("en_core_web_sm")
                    self._loaded = True
                except Exception:
                    self._loaded = False
            return
        # no heavy model available
        self._loaded = False

    def similar(self, word: str, topn: int = 5) -> List[Tuple[str, float]]:
        """
        Return topn semantically similar words with descending scores.
        If no model loaded then return empty list.
        """
        if not self._loaded:
            return []
        if GENSIM_OK and isinstance(self._model, object):
            try:
                sims = self._model.most_similar(positive=[word], topn=topn)
                return [(w, float(score)) for w, score in sims]
            except Exception:
                return []
        if SPACY_OK and self._model:
            try:
                v = self._model.vocab.get(word)
                # spaCy doesn't have a direct most_similar method in v2, use brute force
                if v is None or not v.has_vector:
                    return []
                vec = v.vector
                out = []
                for lex in self._model.vocab:
                    if lex.is_alpha and lex.has_vector:
                        sim = self._cosine(vec, lex.vector)
                        out.append((lex.text, sim))
                out.sort(key=lambda x: -x[1])
                return out[:topn]
            except Exception:
                return []
        return []

    @staticmethod
    def _cosine(a, b):
        # numeric-only helper
        num = sum(x * y for x, y in zip(a, b))
        ma = math.sqrt(sum(x * x for x in a))
        mb = math.sqrt(sum(y * y for y in b))
        if ma == 0 or mb == 0:
            return 0.0
        return num / (ma * mb)
