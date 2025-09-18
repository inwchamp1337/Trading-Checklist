import pandas as pd
from trading_checklist.engine import prepare_df, evaluate_df


def make_df():
    n = 200
    df = pd.DataFrame({
        "open": [i + 0.1 for i in range(n)],
        "high": [i + 0.5 for i in range(n)],
        "low": [i - 0.5 for i in range(n)],
        "close": [i + 0.2 for i in range(n)],
        "volume": [100 + (i % 10) * 5 for i in range(n)],
    })
    return df


def test_prepare_and_evaluate():
    df = make_df()
    pdf = prepare_df(df)
    assert "ema20" in pdf.columns
    assert "adx" in pdf.columns
    scores = evaluate_df(pdf)
    assert isinstance(scores, dict)
    assert "total" in scores
