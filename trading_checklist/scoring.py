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
    # Safety check for minimal required data
    if len(closes) < ema_period:
        return {
            "percent": 0.0, "points": 0, "max_points": 0, "per_item": {},
            "entry": None, "stop": None, "tp1": None, "tp2": None, 
            "risk": None, "rr1": None, "rr2": None,
        }
    
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
    # Safely get recent lows/highs
    recent_low = min(lows[-min(3, len(lows)):]) if lows else None
    recent_high = max(highs[-min(3, len(highs)):]) if highs else None

    # attempt ATR for buffer with proper safety checks
    atr_list = indicators.average_true_range(highs, lows, closes, period=14) if len(closes) >= 14 else []
    atr = atr_list[-1] if atr_list else None
    buffer = (atr * 0.5) if atr else (0.005 * entry if entry else 0)
    stop = (recent_low - buffer) if recent_low is not None else None

    # Calculate risk and targets with null safety
    risk = (entry - stop) if (entry is not None and stop is not None) else None
    tp1 = (entry + 1.5 * risk) if (entry is not None and risk is not None and risk > 0) else None
    tp2 = (entry + 2.5 * risk) if (entry is not None and risk is not None and risk > 0) else None

    # Calculate risk-reward ratios with division by zero protection
    rr1 = (tp1 - entry) / risk if (tp1 is not None and entry is not None and risk is not None and risk > 0) else None
    rr2 = (tp2 - entry) / risk if (tp2 is not None and entry is not None and risk is not None and risk > 0) else None

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
    
    Rules:
    - previous_resistance: last close > max high of previous 20 bars
    - breakout_candle: close > open and close at least 50% of range above open
    - volume: last volume > mean previous 20
    - ema20_touch: previous candle close was near previous EMA20 value (within 1%) and current close > previous close
    - fib_pullback: previous close retraced to 38.2-61.8% of prior swing
    - adx: ADX > 25 (trending market)
    - multi_tf: current trend aligns with larger timeframes (simplified)
    - close_above_ema20: current close is above current EMA20
    - pattern: detects simple bullish pattern
    - risk_reward: calculated risk reward ratio meets minimum threshold (1.5)
    """
    n = len(closes)
    signals = {}
    
    # Early return if not enough data
    if n < ema_period + 5:
        return {k: False for k, _ in CHECK_ITEMS}

    # Calculate EMA values only once
    ema_vals = indicators.ema(closes, ema_period) if n >= ema_period else []
    
    # previous resistance: last close > max high of previous 20 bars
    # Safety check: ensure we have enough data
    if n > ema_period + 1:
        prev_highs = highs[-(ema_period+1):-1]
        signals["previous_resistance"] = closes[-1] > max(prev_highs) if prev_highs else False
    else:
        signals["previous_resistance"] = False

    # breakout candle: close > open and close at least 50% of range above open
    last_range = highs[-1] - lows[-1] if n > 0 else 0
    signals["breakout_candle"] = (closes[-1] > opens[-1]) and ((closes[-1] - opens[-1]) >= 0.5 * last_range if last_range > 0 else False) if n > 0 else False

    # volume: last volume > mean previous 20
    # Safety check for adequate volume data
    if n > ema_period + 1:
        prev_vol = volumes[-(ema_period+1):-1]
        signals["volume"] = volumes[-1] > (sum(prev_vol) / len(prev_vol)) if prev_vol else False
    else:
        signals["volume"] = False

    # ema20_touch: price touched EMA20 on pullback then bounced
    # Fixed: compare previous close with previous EMA value (ema_vals[-2])
    if len(ema_vals) >= 2 and n >= 2:
        prev_ema = ema_vals[-2]  # Previous bar's EMA
        prev_close = closes[-2]  # Previous bar's close
        touched = abs(prev_close - prev_ema) / prev_ema < 0.01 if prev_ema > 0 else False  # within 1%
        signals["ema20_touch"] = touched and (closes[-1] > prev_close)
    else:
        signals["ema20_touch"] = False

    # Fibonacci pullback: check pullback level relative to previous swing high-low
    # Safety check: ensure we have enough data
    if n >= ema_period + 5:
        swing_high = max(highs[-(ema_period+5):-5]) if len(highs) >= ema_period + 5 else (highs[-1] if highs else 0)
        swing_low = min(lows[-(ema_period+5):-5]) if len(lows) >= ema_period + 5 else (lows[-1] if lows else 0)
        swing_range = swing_high - swing_low
        if swing_range > 0 and n >= 2:
            retracement = (closes[-2] - swing_low) / swing_range
            signals["fib_pullback"] = 0.382 <= retracement <= 0.618
        else:
            signals["fib_pullback"] = False
    else:
        signals["fib_pullback"] = False

    # ADX - safety check
    if n >= ema_period * 3:
        adx_vals = indicators.adx(highs[-(ema_period*3):], lows[-(ema_period*3):], closes[-(ema_period*3):], period=14)
        signals["adx"] = len(adx_vals) > 0 and adx_vals[-1] > 25
    else:
        signals["adx"] = False

    # multi timeframe - placeholder: we assume True if current close > previous close
    signals["multi_tf"] = closes[-1] > closes[-2] if n >= 2 else False

    # close_above_ema20: current close is above current EMA20
    # Clarified: explicitly comparing current close with current EMA
    signals["close_above_ema20"] = closes[-1] > ema_vals[-1] if len(ema_vals) > 0 else False

    # pattern detection: check for simple bull flag-like pattern
    if n >= 2:
        body_last = abs(closes[-1] - opens[-1])
        body_prev = abs(closes[-2] - opens[-2])
        signals["pattern"] = (body_prev > 1.5 * body_last) and (closes[-1] > closes[-2])
    else:
        signals["pattern"] = False

    # risk reward: Calculate actual risk/reward instead of always True
    # Use values from current bar to determine if RR is acceptable
    if n >= 3:
        entry = closes[-1]
        recent_low = min(lows[-3:])
        # Simple buffer calculation
        buffer = 0.005 * entry  # 0.5% buffer
        stop = recent_low - buffer
        risk = entry - stop
        
        # Target calculation
        tp1 = entry + (1.5 * risk)
        
        # Check if risk is reasonable and RR meets minimum threshold
        signals["risk_reward"] = risk > 0 and risk < 0.1 * entry and (tp1 - entry) / risk >= 1.5
    else:
        signals["risk_reward"] = False

    return signals
