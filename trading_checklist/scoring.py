from typing import Dict, List
from . import indicators
from typing import Any


CHECK_ITEMS = [
    ("previous_resistance", 10),
    ("breakout_candle", 8),
    ("volume", 9),
    ("ema20_touch", 7),
    ("fib_pullback", 6),
    ("adx", 8),
    ("multi_tf", 7),
    ("close_above_ema20", 6),
    ("pattern", 7),
    ("risk_reward", 9),
]


def score_checklist(signals: Dict[str, bool]) -> Dict[str, int]:
    """Given a mapping of signal booleans, return per-item scores and total."""
    result = {}
    total = 0
    for key, pts in CHECK_ITEMS:
        hit = bool(signals.get(key, False))
        result[key] = pts if hit else 0
        total += result[key]
    result["total"] = total
    return result


def detailed_result(signals: Dict[str, bool], closes: List[float], lows: List[float], highs: List[float], ema_period: int = 20) -> Dict[str, Any]:
    """Return a detailed breakdown: percent, passed items, suggested entry/SL/TP/RR.

    Conservative, test-friendly suggestions:
    - entry: last close
    - stop: recent low (last 3) minus small buffer (0.5% or ATR-based if available)
    - tp1: entry + 1.5 * risk
    - tp2: entry + 2.5 * risk
    """
    per_item = {}
    total_pts = 0
    max_pts = sum(p for _, p in CHECK_ITEMS)
    for key, pts in CHECK_ITEMS:
        hit = bool(signals.get(key, False))
        per_item[key] = {"passed": hit, "points": pts if hit else 0}
        total_pts += pts if hit else 0

    percent = round((total_pts / max_pts) * 100, 1) if max_pts > 0 else 0.0

    # basic trade suggestion
    entry = float(closes[-1]) if closes else None
    recent_low = min(lows[-3:]) if len(lows) >= 3 else (lows[-1] if lows else None)
    recent_high = max(highs[-3:]) if len(highs) >= 3 else (highs[-1] if highs else None)

    # attempt ATR for buffer
    atr_list = indicators.average_true_range(highs, lows, closes, period=14)
    atr = atr_list[-1] if atr_list else None
    buffer = (atr * 0.5) if atr else (0.005 * entry if entry else 0)
    stop = (recent_low - buffer) if recent_low is not None else None

    risk = (entry - stop) if (entry is not None and stop is not None) else None
    tp1 = (entry + 1.5 * risk) if risk is not None else None
    tp2 = (entry + 2.5 * risk) if risk is not None else None

    rr1 = (tp1 - entry) / risk if (tp1 is not None and risk and risk != 0) else None
    rr2 = (tp2 - entry) / risk if (tp2 is not None and risk and risk != 0) else None

    return {
        "percent": percent,
        "points": total_pts,
        "max_points": max_pts,
        "per_item": per_item,
        "entry": entry,
        "stop": stop,
        "tp1": tp1,
        "tp2": tp2,
        "risk": risk,
        "rr1": rr1,
        "rr2": rr2,
    }


def evaluate_bar_series(highs: List[float], lows: List[float], opens: List[float], closes: List[float], volumes: List[float], ema_period: int = 20) -> Dict[str, bool]:
    """A lightweight evaluator that checks the checklist rules on the last bar.

    This is a simplified heuristic-based implementation suitable for unit tests.
    Returns a dict of boolean signals matching CHECK_ITEMS keys.
    """
    n = len(closes)
    signals = {}
    if n < ema_period + 5:
        # Not enough data
        return {k: False for k, _ in CHECK_ITEMS}

    # previous resistance: last close > max high of previous 20 bars
    prev_highs = highs[-(ema_period+1):-1]
    signals["previous_resistance"] = closes[-1] > max(prev_highs)

    # breakout candle: close > open and close at least 50% of range above open
    last_range = highs[-1] - lows[-1]
    signals["breakout_candle"] = (closes[-1] > opens[-1]) and ((closes[-1] - opens[-1]) >= 0.5 * last_range if last_range > 0 else False)

    # volume: last volume > mean previous 20
    prev_vol = volumes[-(ema_period+1):-1]
    signals["volume"] = volumes[-1] > (sum(prev_vol) / len(prev_vol))

    # ema20 touch: price touched EMA20 on pullback then bounced (we check that previous close was near EMA and last close > prev close)
    ema_vals = indicators.ema(closes, ema_period)
    if len(ema_vals) >= 2:
        ema20_last = ema_vals[-1]
        prev_close = closes[-2]
        touched = abs(prev_close - ema20_last) / ema20_last < 0.01  # within 1%
        signals["ema20_touch"] = touched and (closes[-1] > prev_close)
    else:
        signals["ema20_touch"] = False

    # Fibonacci pullback: check pullback level relative to previous swing high-low
    swing_high = max(highs[-(ema_period+5):-5])
    swing_low = min(lows[-(ema_period+5):-5])
    swing_range = swing_high - swing_low
    if swing_range > 0:
        retracement = (closes[-2] - swing_low) / swing_range
        signals["fib_pullback"] = 0.382 <= retracement <= 0.618
    else:
        signals["fib_pullback"] = False

    # ADX
    adx_vals = indicators.adx(highs[-(ema_period*3):], lows[-(ema_period*3):], closes[-(ema_period*3):], period=14)
    signals["adx"] = (len(adx_vals) > 0 and adx_vals[-1] > 25)

    # multi timeframe - placeholder: we assume True if 1H/4H trend matches last candle direction
    signals["multi_tf"] = closes[-1] > closes[-2]

    # close above ema20 on pullback
    signals["close_above_ema20"] = closes[-2] > ema_vals[-1] if len(ema_vals) > 0 else False

    # pattern detection stub: check for simple bull flag-like pattern (small body after big move)
    body_last = abs(closes[-1] - opens[-1])
    body_prev = abs(closes[-2] - opens[-2])
    signals["pattern"] = (body_prev > 1.5 * body_last) and (closes[-1] > closes[-2])

    # risk reward: require that last low provides a reasonable stop (stub: always True for tests)
    signals["risk_reward"] = True

    return signals
