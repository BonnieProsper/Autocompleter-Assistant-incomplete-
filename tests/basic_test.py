# basic_test.py -  sanity tests for hybrid_predictor file, to test prediction code

import sys
import os
import time

# add root to import path so we can improt local modules directly
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from hybrid_predictor import HybridPredictor

# create instance to test the predictor
hp = HybridPredictor("tester")


def test_train_suggest():
    txt = [
        "the quick brown fox jumps over the lazy dog",
        "the timing of this is strange, don't you think",
        "the enemy of my enemy is my friend",
    ]
    hp.train(txt)  # train with sample data above
    out = hp.suggest("the")  # ask for suggestions given prefix 'the'
    # checking - following should return non-empty list
    assert isinstance(out, list)
    assert len(out) > 0
    print("suggest() works:", out[:3])


def test_balance_switch():
    # check that adjusting balance works
    hp.set_balance(0.2)
    assert 0 <= hp.alpha <= 1  # alpha should be between 0 and 1
    hp.set_balance(0.8)
    print("alpha works:", hp.alpha)


if __name__ == "__main__":
    t0 = time.time()
    test_train_suggest()
    test_balance_switch()
    print(f"Basic tests completed in {round(time.time() - t0, 3)} seconds.")
