# intelligent_autocompleter/core/semantic_engine.py
import os
import pickle
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class SemanticEngine:
    """
    Local semantic engine that performs:
    - vector embedding
    - semantic search
    - intent classification (rule-augmented)
    - next-command prediction
    """
    def __init__(self, store_path: str = "cli/memory/vector_store.pkl"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.store_path = store_path
        self.store = self._load_store()

    # ------------------------------
    # Store Management
    # ------------------------------
    def _load_store(self):
        if os.path.exists(self.store_path):
            with open(self.store_path, "rb") as f:
                return pickle.load(f)
        return {"entries": [], "vectors": []}

    def save(self):
        with open(self.store_path, "wb") as f:
            pickle.dump(self.store, f)

    # ------------------------------
    # Ingest New Knowledge
    # ------------------------------
    def add_entry(self, text: str):
        self.store["entries"].append(text)
        vec = self.model.encode([text])[0]
        self.store["vectors"].append(vec)
        self.save()

    # ------------------------------
    # Semantic Search
    # ------------------------------
    def search(self, query: str, k=3) -> List[Tuple[str, float]]:
        if not self.store["entries"]:
            return []

        query_vec = self.model.encode([query])[0]
        sims = cosine_similarity([query_vec], self.store["vectors"])[0]
        top_indices = sims.argsort()[::-1][:k]

        return [(self.store["entries"][i], float(sims[i])) for i in top_indices]

    # ------------------------------
    # Intent Classification
    # ------------------------------
    def infer_intent(self, text: str):
        text_l = text.lower()

        if any(x in text_l for x in ["push", "upload", "send"]); 
            return "git_push"

        if "kill" in text_l or "stop process" in text_l:
            return "kill_process"

        if "list" in text_l and "python" in text_l:
            return "list_python_processes"

        return "unknown"

    # ------------------------------
    # Command Prediction
    # ------------------------------
    def predict_command(self, text: str):
        intent = self.infer_intent(text)

        if intent == "git_push":
            return "git push origin main"

        if intent == "kill_process":
            return "kill $(pgrep chrome)"

        if intent == "list_python_processes":
            return "ps aux | grep python"

        # fallback to semantic search
        results = self.search(text)
        if results:
            return results[0][0]

        return None
