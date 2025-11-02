# cache_utils.py - v small cache and time helpers

from functools import lru_cache, wraps
import time
from typing import Callable, Any

def timed(func: Callable) -> Callable:
    """Decorator returns tuple: (result, elapsed)"""
    @wraps(func)
    def _wrap(*a, **kw):
        t0 = time.perf_counter()
        res = func(*a, **kw)
        t1 = time.perf_counter()
        return res, (t1 - t0)
    return _wrap


def simple_lru(maxsize: int = 1024):
    """wrapper around functools.lru_cache."""
    def _decor(fn):
        return lru_cache(maxsize=maxsize)(fn)
    return _decor
