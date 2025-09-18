from trading_checklist import scoring


def make_synthetic_series():
    # create a synthetic rising series with a pullback
    opens = [i for i in range(1, 41)]
    closes = [i + 0.5 for i in range(1, 41)]
    highs = [c + 0.7 for c in closes]
    lows = [o - 0.7 for o in opens]
    volumes = [100 + (i % 5) * 10 for i in range(40)]
    # make a pullback two bars ago
    closes[-3] = closes[-3] - 2
    return highs, lows, opens, closes, volumes


def test_evaluate_and_score():
    highs, lows, opens, closes, volumes = make_synthetic_series()
    signals = scoring.evaluate_bar_series(highs, lows, opens, closes, volumes, ema_period=20)
    assert isinstance(signals, dict)
    scores = scoring.score_checklist(signals)
    assert scores["total"] <= 100
    assert all(isinstance(v, int) for k, v in scores.items())
