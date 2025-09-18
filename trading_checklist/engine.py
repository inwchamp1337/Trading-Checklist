from typing import Dict
import pandas as pd
import numpy as np
import ta

from .scoring import evaluate_bar_series, score_checklist


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema20"] = ta.trend.EMAIndicator(df["close"], window=20).ema_indicator()
    df["adx"] = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14).adx()
    df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
    df["sma20"] = df["close"].rolling(window=20).mean()
    return df


def evaluate_df(df: pd.DataFrame) -> Dict[str, int]:
    """Evaluate the last bar of a prepared DataFrame and return scores."""
    if len(df) < 30:
        raise ValueError("Not enough bars to evaluate")
    highs = df["high"].tolist()
    lows = df["low"].tolist()
    opens = df["open"].tolist()
    closes = df["close"].tolist()
    volumes = df["volume"].tolist()
    signals = evaluate_bar_series(highs, lows, opens, closes, volumes, ema_period=20)
    return score_checklist(signals)
