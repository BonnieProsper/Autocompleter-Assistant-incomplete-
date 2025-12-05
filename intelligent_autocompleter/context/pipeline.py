# intelligent_autocompleter/context/pipeline.py
# for context processing

import time
from .normalizer import normalize_text
from .tokenizer import simple_tokenize
from .scorers import score_uncertainty, score_repetition

# try to reuse existing CtxPersonal if present
try:
    from ..context_personal import CtxPersonal  # existing file
except Exception:
    CtxPersonal = None  # if missing fallback so pipeline still works


class ContextPipeline:
    """
    Small pipeline object:
     - process(text) -> bundle
    bundle keys:
     - text: original (normalized)
     - tokens: token list
     - last_token: last token or ''
     - ctx_obj: CtxPersonal instance if available
     - uncertainty: 0..1
     - repetition: 0..1
     - timestamp: epoch
    """

    def __init__(self, user="default", ctx_obj=None):
        # prefers provided context object
        if ctx_obj is not None:
            self.ctx = ctx_obj
        else:
            self.ctx = CtxPersonal(user) if CtxPersonal is not None else None

    def process(self, raw_text: str) -> dict:
        t0 = time.time()
        text = normalize_text(raw_text or "")
        toks = simple_tokenize(text)
        last = toks[-1] if toks else ""
        bundle = {
            "text": text,
            "tokens": toks,
            "last_token": last,
            "timestamp": t0,
            "uncertainty": score_uncertainty(toks),
            "repetition": score_repetition(toks),
            "ctx_obj": self.ctx,
        }
        return bundle

    # convenience helper used by CLI, update context from final sentence
    def ingest(self, sentence: str):
        if self.ctx is None:
            return
        #  CtxPersonal does tokenization and learning
        try:
            self.ctx.learn(sentence)
            self.ctx.save()
        except Exception:
            # allow exception without error
            pass
