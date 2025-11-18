def test_adaptive_basic(tmp_path):
    p = tmp_path / "profile.json"
    from intelligent_autocompleter.core.adaptive_learner import AdaptiveLearner
    a = AdaptiveLearner(str(p))
    w0 = a.get_weights()
    a.reward("semantic", 0.1)
    w1 = a.get_weights()
    assert w1["semantic"] >= w0["semantic"]
    a.penalize("markov", 0.05)
    assert a.get_weights()["markov"] <= w0["markov"] + 1e-6
