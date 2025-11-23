# semantic_engine.py
# A hybrid semantic + rule-based engine with:
# - embeddings
# - persistent vector store
# - intent matching with ranking
# - semantic fallback recommendations
# - plugin-extensible rule system

import os
import pickle
from typing import List, Tuple, Dict, Any, Callable

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticEngine:
    """
    Core semantic engine for the Intelligent Autocompleter.
    Responsibilities:
     - vectorization via SentenceTransformer
     - semantic search (top-k)
     - hybrid intent system (scored rule matching)
     - mapping intents to CLI actions
     - persistent vector store (auto-de-duplicated)
    Plugin friendly:
     - add intent rules dynamically
     - add new command mappings
     - replace the vector store or model easily
    """

    # Initialization -----------------------------------------
    def __init__(
        self,
        store_path: str = "cli/memory/vector_store.pkl",
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.store_path = store_path
        self.model_name = model_name

        # local semantic model
        self.model = SentenceTransformer(self.model_name)
        # load or initialize store
        self.store = self._load_store()

        # build rule system
        self.intent_rules: Dict[str, List[Callable[[str], float]]] = {}
        self._register_default_intent_rules()

        # map intents to commands
        self.intent_commands = self._default_intent_command_map()

    # Persistence Layer ------------------------------------------------------------
    def _load_store(self) -> Dict[str, Any]:
        if os.path.exists(self.store_path):
            try:
                with open(self.store_path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass  # fall through to rebuild if corrupted

        return {"entries": [], "vectors": [], "meta": {"count": 0}}

    def _save_store(self):
        tmp = self.store_path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(self.store, f)
        os.replace(tmp, self.store_path)

    # Adding Knowledge -----------------------------------------------------------
    def add_entry(self, text: str) -> None:
        """Add a text entry unless already present (deduped)."""
        if not text or not isinstance(text, str):
            return
        if text in self.store["entries"]:
            return  # prevents duplicate vectors

        vector = self.model.encode([text])[0]
        self.store["entries"].append(text)
        self.store["vectors"].append(vector)
        self.store["meta"]["count"] += 1
        self._save_store()

    # Semantic Search ---------------------------------------------------------
    def search(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        if not self.store["entries"]:
            return []

        qvec = self.model.encode([query])[0]
        sims = cosine_similarity([qvec], self.store["vectors"])[0]
        idxs = sims.argsort()[::-1][:k]
        return [(self.store["entries"][i], float(sims[i])) for i in idxs]

    # Intent System: Scored Rules -----------------------------------------------------
    def _register_default_intent_rules(self):
        """Rule score: 0–1. Higher = stronger match."""

        def contains(term: str) -> Callable[[str], float]:
            return lambda text: 1.0 if term in text else 0.0

        def pair(a: str, b: str) -> Callable[[str], float]:
            return lambda text: 0.8 if (a in text and b in text) else 0.0
        
        self.intent_rules = {
            "git_push": [
                contains("push"),
                contains("upload"),
                pair("commit", "push"),
            ],
            "kill_process": [
                contains("kill"),
                pair("end", "process"),
            ],
            "list_python_processes": [
                pair("list", "python"),
                contains("python processes"),
            ],
        }

    def infer_intent(self, text: str) -> Tuple[str, float]:
        """
        Return (intent, score). Score is the max rule weight.
        """
        text = text.lower()
        best_intent = "unknown"
        best_score = 0.0
        for intent, rules in self.intent_rules.items():
            for rule in rules:
                try:
                    score = float(rule(text))
                except Exception:
                    score = 0.0
                if score > best_score:
                    best_intent = intent
                    best_score = score
        return best_intent, best_score

    # Intent → Command Mapping ------------------------------------------
    def _default_intent_command_map(self):
        return {
            "git_push": "git push origin main",
            "kill_process": "kill $(pgrep chrome)",
            "list_python_processes": "ps aux | grep python",
        }

    # Next-Command Prediction ----------------------------------------
    def predict_command(self, text: str) -> str | None:
        intent, score = self.infer_intent(text)

        # if intent confident → use it
        if score >= 0.8 and intent in self.intent_commands:
            return self.intent_commands[intent]

        # fallback → semantic search
        hits = self.search(text, k=1)
        if hits:
            return hits[0][0]

        return None
