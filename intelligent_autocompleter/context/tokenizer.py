# intelligent_autocompleter/context/tokenizer.py
# simple but extendable tokenizer


def simple_tokenize(s: str):
    """
    Return list of tokens (words). Only alpha tokens and short tokens kept.
    Simple, could swap in spacy/NLTK later.
    """
    if not s:
        return []
    toks = s.split()
    out = []
    for t in toks:
        t = t.strip()
        if not t:
            continue
        # keep alphanumeric + underscores, drop pure punctuation
        if any(ch.isalnum() for ch in t):
            out.append(t)
    return out
