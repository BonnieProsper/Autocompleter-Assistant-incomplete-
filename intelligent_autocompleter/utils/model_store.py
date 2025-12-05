# model_store.py — simple persistence layer for an autocompleter system

# handles saving and loading data models:
# - markov chain model (for text prediction)
# - word/sentence embeddings (for semantic suggestions)
# - user cache (to store user history, preferences and learning progress)
# - uses JSON for data (Markov + user cache) and Pickle for binary embedding data

import os
import json
import pickle
from datetime import datetime

# Configuration -------------------
# Directory where all data files are stored
DATA_DIRECTORY = "data"
os.makedirs(DATA_DIRECTORY, exist_ok=True)  # Create if it doesn’t exist

# Paths for each stored file
MARKOV_PATH = os.path.join(DATA_DIRECTORY, "markov_model.json")
EMBED_PATH = os.path.join(DATA_DIRECTORY, "embeddings.pkl")
USER_CACHE = os.path.join(DATA_DIRECTORY, "user_context.json")


# Helper Functions ---------------
def ts() -> str:
    """Return timestamp for logging."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# Markov Model Persistence -------------------------
def save_markov(model: dict):
    """
    Save the Markov chain model to disk, in JSON format.
    Args:
        model (dict): The Markov chain data structure (e.g word transitions)
    """
    try:
        with open(MARKOV_PATH, "w", encoding="utf-8") as f:
            json.dump(model, f, indent=2)
        print(f"[{ts()}] Saved Markov model ({len(model)} keys)")
    except Exception as e:
        print(f"[{ts()}] save_markov error: {e}")


def load_markov() -> dict:
    """
    Load the Markov chain model from where its saved on disk.
    Returns:
        dict: The Markov chain data, or an empty dict if missing/there is an error.
    """
    if not os.path.exists(MARKOV_PATH):
        return {}
    try:
        with open(MARKOV_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"[{ts()}] Loaded Markov model ({len(data)} keys)")
        return data
    except Exception as e:
        print(f"[{ts()}] load_markov error: {e}")
        return {}


# Embeddings Persistence --------------------------
def save_embeddings(vectors: dict):
    """
    Save embedding vectors using pickle (in binary format).
    Args:
        vectors (dict): Mapping of tokens/sentences to embedding arrays.
    """
    try:
        with open(EMBED_PATH, "wb") as f:
            pickle.dump(vectors, f)
        print(f"[{ts()}] Saved {len(vectors)} embeddings")
    except Exception as e:
        print(f"[{ts()}] save_embeddings error: {e}")


def load_embeddings() -> dict:
    """
    Load embedding vectors from the disk.
    Returns:
        dict: the loaded embeddings or an empty dict if missing/if there is an error.
    """
    if not os.path.exists(EMBED_PATH):
        return {}
    try:
        with open(EMBED_PATH, "rb") as f:
            vecs = pickle.load(f)
        print(f"[{ts()}] Loaded {len(vecs)} embeddings")
        return vecs
    except Exception as e:
        print(f"[{ts()}] load_embeddings error: {e}")
        return {}


# User Cache Persistence ------------------
def save_user_cache(user_data: dict):
    """
    Save user-specific context (e.g typing history or XP).
    Args:
        user_data (dict): User state {
            "history": [...],
            "suggestions": {...},
            "xp": int
        }
    """
    try:
        with open(USER_CACHE, "w", encoding="utf-8") as f:
            json.dump(user_data, f, indent=2)
        print(f"[{ts()}] Saved user cache")
    except Exception as e:
        print(f"[{ts()}] save_user_cache error: {e}")


def load_user_cache() -> dict:
    """
    Load user context from disk. Initialize with defaults if none exists.
    Returns:
        dict: User context data (with defaults).
    """
    if not os.path.exists(USER_CACHE):
        # Default if no previous data exists
        return {"history": [], "suggestions": {}, "xp": 0}
    try:
        with open(USER_CACHE, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(
            f"[{ts()}] Loaded user cache ({len(data.get('history', []))} history items)"
        )
        return data
    except Exception as e:
        print(f"[{ts()}] load_user_cache error: {e}")
        return {"history": [], "suggestions": {}, "xp": 0}
