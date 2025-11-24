# embeddings.py
# Semantic vector helper - redundant because of semantic_engine.py?
# unified access to Gensim or spaCy embeddings with degradation+caching of similarity computations

from __future__ import annotations

from typing import List, Tuple, Optional, Dict
import math
import logging

logger = logging.getLogger(__name__)

# Flags to indicate optional dependencies
GENSIM_OK = False
SPACY_OK = False

try:
    from gensim.models import KeyedVectors  # type: ignore
    GENSIM_OK = True
except Exception:
    GENSIM_OK = False
    KeyedVectors = None  # type: ignore[misc]

try:
    import spacy
    SPACY_OK = True
except Exception:
    SPACY_OK = False
    spacy = None  # type: ignore[misc]


class Embeddings:
    """
    Unified wrapper around Gensim or spaCy vector models.
    Usage:
     - Load vector model from disk or spaCy pipeline
     - Provide 'similar()' semantic-neighbour lookups
     - Provide direct vector access for other modules
     - Degrade if no embeddings backend available
    """
    def __init__(self, path: Optional[str] = None):
        self._model = None
        self._loaded = False
        self._cache_similar: Dict[Tuple[str, int], List[Tuple[str, float]]] = {}
        if path:
            self.load(path)

    # Model Loading ---------------------------------------------------------------
    def load(self, path: Optional[str] = None) -> bool:
        """
        Load a Gensim KeyedVectors or spaCy model.
        If 'path' is None, attempts to load spaCy vectors instead.
        Returns True if load succeeded.
        """
        # Prefer Gensim if explicit path is provided
        if path and GENSIM_OK:
            try:
                self._model = KeyedVectors.load_word2vec_format(path, binary=True)
                self._loaded = True
                logger.info(f"Loaded Gensim embeddings from: {path}")
                return True
            except Exception as e:
                logger.warning(f"Gensim load failed: {e}")

        # Try spaCy medium vectors
        if SPACY_OK:
            for pkg in ("en_core_web_md", "en_core_web_lg", "en_core_web_sm"):
                try:
                    self._model = spacy.load(pkg)
                    self._loaded = True
                    logger.info(f"Loaded spaCy model: {pkg}")
                    return True
                except Exception:
                    continue

        # No available model
        self._loaded = False
        logger.warning("No embeddings backend available; semantic features disabled.")
        return False

    # Access utilities ----------------------------------------------------------------
    def is_loaded(self) -> bool:
        return self._loaded

    def get_vector(self, word: str) -> Optional[List[float]]:
        """Return raw vector for a word, if available."""
        if not self._loaded:
            return None
        # Gensim path
        if GENSIM_OK and isinstance(self._model, KeyedVectors):
            if word in self._model:
                return self._model[word].tolist()
            return None
        # spaCy path
        if SPACY_OK:
            lex = self._model.vocab.get(word)
            if lex and lex.has_vector:
                return lex.vector
        return None

    # Similarity search------------------------------------------------------------
    def similar(self, word: str, topn: int = 5) -> List[Tuple[str, float]]:
        """
        Return top-N semantically similar words.
        Results are cached per (word, topn) to improve performance.
        """
        if not self._loaded:
            return []
        cache_key = (word, topn)
        if cache_key in self._cache_similar:
            return self._cache_similar[cache_key]
            
        result: List[Tuple[str, float]] = []

        # Gensim backend
        if GENSIM_OK and isinstance(self._model, KeyedVectors):
            try:
                sims = self._model.most_similar(positive=[word], topn=topn)
                result = [(w, float(score)) for w, score in sims]
                self._cache_similar[cache_key] = result
                return result
            except Exception:
                return []

        # spaCy backend (manual similarity scoring)
        if SPACY_OK:
            lex = self._model.vocab.get(word)
            if not lex or not lex.has_vector:
                return []
            target = lex.vector
            scores = []
            for other in self._model.vocab:
                if other.is_alpha and other.has_vector:
                    sim = self._cosine(target, other.vector)
                    scores.append((other.text, sim))
            scores.sort(key=lambda x: -x[1])
            result = scores[:topn]
            self._cache_similar[cache_key] = result
            return result
        return []

    # Cosine similarity ------------------------------------------------
    @staticmethod
    def _cosine(a, b) -> float:
        """Fast cosine similarity between two numeric vectors."""
        num = sum(x * y for x, y in zip(a, b))
        da = math.sqrt(sum(x * x for x in a))
        db = math.sqrt(sum(y * y for y in b))
        if da == 0 or db == 0:
            return 0.0
        return num / (da * db)
