# context_manager.py
# Short-Term Session Context Engine
# CtxPersonal is long-term per-user learning, ContextManager is short-term sesssion learning
# Short term context includes:
#   - Recent token window
#   - Session-level confirmed words
#   - Lightweight topic signals (noun chunks, embeddings)
#   - Recency-weighted bias coefficients
#   - Semantic context vector for the current topic
# Slightly biases predictions in a contextually
# relevant direction without affecting long-term memory.
# ----------------------------------------------------------------------

from __future__ import annotations
import json
import math
import re
from pathlib import Path
from collections import deque, Counter
from typing import Dict, Optional, List, Tuple

# Lazy spaCy loader
NLP = None

def _ensure_spacy():
    """Load spaCy on first demand."""
    global NLP
    if NLP is not None:
        return NLP

    try:
        import spacy
        NLP = spacy.load("en_core_web_sm", disable=["ner"])
    except Exception:
        NLP = None
    return NLP


class ContextManager:
    """
    Session-level context engine.

    Responsibilities:
        - Maintain a sliding window of recent tokens
        - Track confirmed word selections (short-term frequency)
        - Extract lightweight "topics" via noun-chunk detection (if spaCy)
        - Maintain a semantic context vector averaged over recent words
        - Provide soft bias weights for the prediction engine
    """

    VERSION = 1
    TOKEN_RE = re.compile(r"[a-zA-Z']+")

    def __init__(self, userdata_path: Path, window: int = 12):
        self.userdata_path = userdata_path
        self.window = window

        self.recent_tokens: deque[str] = deque(maxlen=window)
        self.confirmed: Counter[str] = Counter()
        self.topic_counter: Counter[str] = Counter()
        self.semantic_vector: Optional[List[float]] = None

        self._load()

    def _load(self) -> None:
        if not self.userdata_path.exists():
            return
        try:
            data = json.loads(self.userdata_path.read_text())
            if data.get("version") != self.VERSION:
                return  # incompatible, start fresh
            self.recent_tokens.extend(data.get("recent_tokens", []))
            self.confirmed.update(data.get("confirmed", {}))
            self.topic_counter.update(data.get("topics", {}))
            self.semantic_vector = data.get("semantic_vector")
        except Exception:
            # Corrupted or unreadable — safe recover
            self.recent_tokens.clear()
            self.confirmed.clear()
            self.topic_counter.clear()
            self.semantic_vector = None

    def _save(self) -> None:
        data = {
            "version": self.VERSION,
            "recent_tokens": list(self.recent_tokens),
            "confirmed": dict(self.confirmed),
            "topics": dict(self.topic_counter),
            "semantic_vector": self.semantic_vector,
        }
        try:
            self.userdata_path.write_text(json.dumps(data, indent=2))
        except Exception:
            pass

    
    def add_token(self, text: str) -> None:
        """Add raw user input, extract tokens, update vectors/topics."""
        tokens = [t.lower() for t in self.TOKEN_RE.findall(text)]
        if not tokens:
            return

        nlp = _ensure_spacy()

        for t in tokens:
            self.recent_tokens.append(t)
            self._update_semantic(t, nlp)

        # topic extraction
        if nlp:
            try:
                doc = nlp(text)
                for chunk in doc.noun_chunks:
                    if len(chunk.text) > 2:
                        self.topic_counter[chunk.text.lower()] += 1
            except Exception:
                pass

        self._save()

    def confirm_word(self, word: str) -> None:
        """The user explicitly selected this word."""
        w = word.lower()
        self.confirmed[w] += 1
        self._save()

    
    def _update_semantic(self, token: str, nlp) -> None:
        """Update the running semantic context vector."""
        if not nlp:
            return
        try:
            doc = nlp(token)
            if not doc.has_vector:
                return

            vec = doc.vector
            if self.semantic_vector is None:
                self.semantic_vector = vec.tolist()
            else:
                # Exponential moving average
                alpha = 0.25
                self.semantic_vector = [
                    (1 - alpha) * v_old + alpha * v_new
                    for v_old, v_new in zip(self.semantic_vector, vec)
                ]
        except Exception:
            pass


    
    def get_context_bias(self) -> Dict[str, float]:
        """
        Produce a score boost for candidate words.

        Boosts come from:
            - confirmed words (session frequency)
            - similarity to semantic context
            - match with recent tokens

        Returned:
            { "word": bias_factor }
        """
        bias: Dict[str, float] = {}

        # Confirmed selections get a confidence boost
        for w, c in self.confirmed.items():
            bias[w] = 1.0 + min(0.75, c * 0.2)

        # Semantic affinity (spaCy vectors)
        nlp = NLP or _ensure_spacy()
        if nlp and self.semantic_vector is not None:
            try:
                for w in list(bias.keys()):
                    doc = nlp(w)
                    if not doc.has_vector:
                        continue
                    sim = doc.similarity(doc.vocab.vectors.find(self.semantic_vector))
                    bias[w] *= 1.0 + max(0.0, sim * 0.1)
            except Exception:
                pass

        # Recency: newer tokens receive slight preference
        recent_words = set(self.recent_tokens)
        for w in recent_words:
            bias[w] = max(bias.get(w, 1.0), 1.05)

        return bias

    
    def summarize(self) -> str:
        """Readable diagnostic string."""
        recent = ", ".join(self.recent_tokens[-5:]) or "None"
        top = ", ".join([w for w, _ in self.confirmed.most_common(3)]) or "None"
        topic = ", ".join([t for t, _ in self.topic_counter.most_common(2)]) or "None"
        return f"Recent: {recent} | Confirmed: {top} | Topics: {topic}"
