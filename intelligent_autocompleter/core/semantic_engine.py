# intelligent_autocompleter/core/semantic_engine.py
"""
SemanticEngine - semantic module. Features:
 - sentence embeddings using configurable SentenceTransformer model
 - persistent vector store (pickle-based, atomic writes)
 - top-k semantic search (fast numpy-based cosine similarity)
 - rule-based intent system (scored rules returning 0-1)
 - intent to command mapping with confidence threshold
 - plugin friendly methods to add/remove rules & mappings
 - lazy model loading (useful for unit tests/CI)
 - in-memory caches for search and vector lookup
 - error handling and logging hooks
 - module has single responsibility: semantic search + intent mapping, called by other modules
"""

from __future__ import annotations

import os
import pickle
import tempfile
from typing import List, Tuple, Dict, Any, Callable, Optional
import threading
import time

import numpy as np

# third-party dependencies (may raise at import time if not installed)
try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    SentenceTransformer = None  # type: ignore

# logger hook - replace with util.logger??
try:
    from intelligent_autocompleter.utils.logger_utils import Log
except Exception:

    class _SimpleLog:
        @staticmethod
        def write(msg: str):
            print(msg)

        @staticmethod
        def metric(*args, **kwargs):
            pass

    Log = _SimpleLog()  # type: ignore

# types
Vec = np.ndarray
EntryScore = Tuple[str, float]
RuleFn = Callable[[str], float]  # returns a score between 0-1

DEFAULT_STORE_PATH = os.path.join("data", "semantic_vector_store.pkl")


class SemanticEngine:
    def __init__(
        self,
        store_path: str = DEFAULT_STORE_PATH,
        model_name: str = "all-MiniLM-L6-v2",
        load_model: bool = True,
    ):
        """
        Args:
            store_path: path to persistent vector store (pickle).
            model_name: name of SentenceTransformer model to use.
            load_model: if False, defer loading model until first embed call.
        """
        self.store_path = store_path
        self.model_name = model_name
        self._model = None  # lazy-loaded
        self._model_lock = threading.Lock()
        self._model_loaded = False

        # storage layout:
        # { "entries": List[str], "vectors": List[np.ndarray], "meta": { ... } }
        self._store: Dict[str, Any] = self._load_store()

        # intent rule system: intent -> list[RuleFn]
        self._intent_rules: Dict[str, List[RuleFn]] = {}
        self._register_default_intent_rules()

        # mapping: intent -> command string
        self._intent_to_cmd: Dict[str, str] = self._default_intent_command_map()

        # caches (in-memory)
        self._search_cache: Dict[Tuple[str, int], List[EntryScore]] = {}
        # small TTL for search cache
        self._search_cache_ttl = 10.0
        self._search_cache_time: Dict[Tuple[str, int], float] = {}

        # control whether embeddings/model are available
        self.model_enabled = bool(load_model) and (SentenceTransformer is not None)
        if load_model and self.model_enabled:
            try:
                self._ensure_model_loaded()
            except Exception as e:
                Log.write(f"[SemanticEngine] failed to load model '{model_name}': {e}")
                self.model_enabled = False

    # -------------------------
    # Model lifecycle
    # -------------------------
    def _ensure_model_loaded(self):
        """Thread-safe model initialization."""
        if self._model_loaded:
            return
        if not self.model_enabled:
            raise RuntimeError(
                "Model disabled (SentenceTransformer not available or model_enabled=False)"
            )
        with self._model_lock:
            if not self._model_loaded:
                # may raise if model not present, caller should handle error
                self._model = SentenceTransformer(self.model_name)
                self._model_loaded = True
                Log.write(f"[SemanticEngine] model '{self.model_name}' loaded.")

    def disable_model(self):
        """Disable model usage (useful for unit tests)."""
        self.model_enabled = False
        self._model = None
        self._model_loaded = False

    def enable_model(self):
        """Enable model (attempt to load)."""
        self.model_enabled = True
        self._ensure_model_loaded()

    # -------------------------
    # persistence layer
    # -------------------------
    def _load_store(self) -> Dict[str, Any]:
        """Load from disk or initialize an empty store."""
        if os.path.exists(self.store_path):
            try:
                with open(self.store_path, "rb") as fh:
                    st = pickle.load(fh)
                    # ensure minimal structure
                    if "entries" in st and "vectors" in st:
                        # convert vectors to np.array if stored as lists
                        st["vectors"] = [np.asarray(v) for v in st["vectors"]]
                        st.setdefault("meta", {}).setdefault(
                            "count", len(st["entries"])
                        )
                        return st
            except Exception as e:
                Log.write(f"[SemanticEngine] corrupted store, recreating: {e}")

        # default empty
        return {"entries": [], "vectors": [], "meta": {"count": 0}}

    def _save_store(self):
        """Atomic(?) save of the vector store to disk (writes to tmp file then replace)."""
        dirname = os.path.dirname(self.store_path) or "."
        os.makedirs(dirname, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(prefix="vecstore_", dir=dirname)
        try:
            # convert np arrays to lists for pickle compatibility
            serializable = {
                "entries": list(self._store["entries"]),
                "vectors": [v.tolist() for v in self._store["vectors"]],
                "meta": dict(self._store.get("meta", {})),
            }
            with os.fdopen(tmp_fd, "wb") as fh:
                pickle.dump(serializable, fh)
            os.replace(tmp_path, self.store_path)
            Log.write(
                f"[SemanticEngine] vector store saved ({len(self._store['entries'])} entries)."
            )
        except Exception as e:
            Log.write(f"[SemanticEngine] failed to save store: {e}")
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    # -------------------------
    # adding knowledge
    # -------------------------
    def add_entry(self, text: str, persist: bool = True):
        """Add a new text entry to the store unless it's a duplicate.
        Args:
            text: string entry to add (must be non-empty).
            persist: if True, write store to disk after addition.
        """
        if not text or not isinstance(text, str):
            return

        if text in self._store["entries"]:
            # don't duplicate
            return

        # compute vector if model is available
        vec = None
        if self.model_enabled:
            try:
                self._ensure_model_loaded()
                vec = self._model.encode([text])[0]
                vec = np.asarray(vec, dtype=float)
            except Exception as e:
                Log.write(f"[SemanticEngine] failed to encode entry: {e}")
                # fall through, do not add entry without vector
                return
        else:
            # model disabled, do not add entries without vectors
            Log.write("[SemanticEngine] model disabled; skipping add_entry.")
            return

        # store
        self._store["entries"].append(text)
        self._store["vectors"].append(vec)
        self._store["meta"]["count"] = len(self._store["entries"])

        # invalidate search cache
        self._search_cache.clear()
        self._search_cache_time.clear()

        if persist:
            self._save_store()

    def batch_add(self, texts: List[str], persist: bool = True, batch_size: int = 32):
        """Add multiple entries efficiently in batches."""
        if not texts:
            return
        if not self.model_enabled:
            Log.write("[SemanticEngine] model disabled; batch_add aborted.")
            return

        self._ensure_model_loaded()
        to_add = []
        for t in texts:
            if not t or not isinstance(t, str):
                continue
            if t in self._store["entries"]:
                continue
            to_add.append(t)

        # process in batches
        for i in range(0, len(to_add), batch_size):
            batch = to_add[i : i + batch_size]
            try:
                vecs = self._model.encode(batch)
                for txt, v in zip(batch, vecs):
                    self._store["entries"].append(txt)
                    self._store["vectors"].append(np.asarray(v, dtype=float))
            except Exception as e:
                Log.write(
                    f"[SemanticEngine] batch encode failed on slice {i}:{i+batch_size}: {e}"
                )

        self._store["meta"]["count"] = len(self._store["entries"])
        self._search_cache.clear()
        self._search_cache_time.clear()
        if persist:
            self._save_store()

    # -------------------------
    # search utilities
    # -------------------------
    @staticmethod
    def _cosine_similarity(qvec: Vec, mat: List[Vec]) -> np.ndarray:
        """Return cosine similarity scores between qvec and each vector in mat."""
        if len(mat) == 0:
            return np.array([], dtype=float)
        M = np.vstack(mat)  # shape (n, dim)
        # compute dot and norms
        q = np.asarray(qvec, dtype=float)
        dots = M.dot(q)
        m_norms = np.linalg.norm(M, axis=1)
        q_norm = np.linalg.norm(q) + 1e-12
        denom = m_norms * q_norm + 1e-12
        sims = dots / denom
        return sims

    def search(
        self, query: str, k: int = 3, use_cache: bool = True
    ) -> List[EntryScore]:
        """Return top-k entries with cosine similarity scores (descending).
        If model disabled, returns empty list.
        """
        if not query or not self._store["entries"]:
            return []

        key = (query, k)
        now = time.time()
        if use_cache and key in self._search_cache:
            t0 = self._search_cache_time.get(key, 0.0)
            if now - t0 < self._search_cache_ttl:
                return list(self._search_cache[key])

        if not self.model_enabled:
            return []

        # embed query
        try:
            self._ensure_model_loaded()
            qvec = self._model.encode([query])[0]
            qvec = np.asarray(qvec, dtype=float)
        except Exception as e:
            Log.write(f"[SemanticEngine] query embed failed: {e}")
            return []

        sims = self._cosine_similarity(qvec, self._store["vectors"])
        if sims.size == 0:
            return []

        idxs = np.argsort(sims)[::-1][:k]
        results = [(self._store["entries"][int(i)], float(sims[int(i)])) for i in idxs]
        # cache
        self._search_cache[key] = results
        self._search_cache_time[key] = now
        return results

    # -------------------------
    # intent rules
    # -------------------------
    def _register_default_intent_rules(self):
        # helpers that return 0.0-1.0 scores
        def contains(term: str) -> RuleFn:
            term_l = term.lower()
            return lambda text: 1.0 if term_l in text else 0.0

        def all_terms(*terms: str) -> RuleFn:
            terms_l = [t.lower() for t in terms]
            return lambda text: 1.0 if all(t in text for t in terms_l) else 0.0

        self._intent_rules = {
            "git_push": [contains("git push"), contains("push"), contains("upload")],
            "kill_process": [contains("kill"), all_terms("kill", "process")],
            "list_python_processes": [
                contains("python processes"),
                contains("ps aux"),
                contains("list python"),
            ],
        }

    def add_intent_rule(self, intent: str, rule: RuleFn):
        """Add a custom scoring rule for an intent."""
        if intent not in self._intent_rules:
            self._intent_rules[intent] = []
        self._intent_rules[intent].append(rule)

    def remove_intent_rules(self, intent: str):
        """Remove all rules for an intent."""
        self._intent_rules.pop(intent, None)

    def infer_intent(self, text: str) -> Tuple[str, float]:
        """
        Evaluate rules and return best (intent, score).
        Score is max rule score for that intent (0-1).
        """
        if not text:
            return "unknown", 0.0
        t = text.lower()
        best_intent = "unknown"
        best_score = 0.0
        for intent, rules in self._intent_rules.items():
            for rule in rules:
                try:
                    score = float(rule(t))
                except Exception:
                    score = 0.0
                if score > best_score:
                    best_score = score
                    best_intent = intent
        return best_intent, best_score

    # -------------------------
    # intent -> command mapping
    # -------------------------
    def _default_intent_command_map(self) -> Dict[str, str]:
        return {
            "git_push": "git push origin main",
            "kill_process": "kill $(pgrep chrome)",
            "list_python_processes": "ps aux | grep python",
        }

    def register_intent_command(self, intent: str, command: str):
        """Map an intent to a shell command or action string."""
        self._intent_to_cmd[intent] = command

    def predict_command(
        self, text: str, confidence_threshold: float = 0.8
    ) -> Optional[str]:
        """
        Return a command string if an intent is confidently matched, otherwise fall back
        to semantic search (top-1).
        """
        if not text:
            return None

        intent, score = self.infer_intent(text)
        if score >= confidence_threshold and intent in self._intent_to_cmd:
            return self._intent_to_cmd[intent]

        # fallback to semantic match
        hits = self.search(text, k=1)
        if hits:
            return hits[0][0]
        return None

    # -------------------------
    # utilities & maintenance
    # -------------------------
    def get_store_size(self) -> int:
        return len(self._store["entries"])

    def clear_store(self, persist: bool = True):
        """Remove all stored entries."""
        self._store = {"entries": [], "vectors": [], "meta": {"count": 0}}
        self._search_cache.clear()
        self._search_cache_time.clear()
        if persist:
            self._save_store()

    def export_store(self) -> Dict[str, Any]:
        """Return a serializable snapshot (vectors converted to lists)."""
        return {
            "entries": list(self._store["entries"]),
            "vectors": [v.tolist() for v in self._store["vectors"]],
            "meta": dict(self._store.get("meta", {})),
        }

    def load_from_snapshot(self, snapshot: Dict[str, Any], persist: bool = True):
        """Load store from a previously exported snapshot (expects vectors as lists)."""
        entries = snapshot.get("entries", [])
        vectors = snapshot.get("vectors", [])
        self._store["entries"] = list(entries)
        self._store["vectors"] = [np.asarray(v, dtype=float) for v in vectors]
        self._store["meta"] = snapshot.get("meta", {"count": len(entries)})
        self._search_cache.clear()
        self._search_cache_time.clear()
        if persist:
            self._save_store()
