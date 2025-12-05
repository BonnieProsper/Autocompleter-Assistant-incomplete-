# threaded_runner.py - wrapper to run functions in threads and collect their results.

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, List, Tuple


def run_parallel(tasks: Iterable[Callable], max_workers: int = 4) -> List:
    """
    Run callables (no-arg functions) in small thread pool and return results in order of completion.
    Each task should be a zero-argument lambda or function.
    """
    out = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(t) for t in tasks]
        for f in as_completed(futs):
            out.append(f.result())
    return out
