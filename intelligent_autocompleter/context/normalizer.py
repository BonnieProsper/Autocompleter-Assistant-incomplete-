# intelligent_autocompleter/context/normalizer.py
import re

_strip_re = re.compile(r"[^\w\s'-]")  # keep simple punctuation useful for tokens


def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    # normalize weird whitespace
    s = " ".join(s.split())
    # remove control chars and odd symbols (keep apostrophes/hyphens)
    s = _strip_re.sub("", s)
    # lower-case but not for single-letter I
    return s
