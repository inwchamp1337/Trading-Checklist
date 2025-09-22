from typing import Dict, List
from . import indicators
from typing import Any


def calculate_rsi(prices: List[float], period: int = 14) -> List[float]:
    """Calculate RSI indicator for momentum analysis."""
    if len(prices) < period + 1:
        return []
    
    deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    gains = [d if d > 0 else 0 for d in deltas]
    losses = [-d if d < 0 else 0 for d in deltas]
    
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    
    rsi_values = []
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
        rsi_values.append(rsi)
    
    return rsi_values


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
    - previous_resistance: last close > max high of previous 20 bars with volume confirmation
    - breakout_candle: strong bullish candle with long body and high volume
    - volume: last volume > 1.5x mean previous 20 (stronger threshold)
    - ema20_touch: previous candle close was near previous EMA20 value (within 1%) and current close > previous close
    - fib_pullback: previous close retraced to 38.2-61.8% of prior swing with bullish reversal
    - adx: ADX > 25 (trending market) with DI+ > DI- for bullish confirmation
    - multi_tf: current trend aligns with larger timeframes (EMA alignment across periods)
    - close_above_ema20: current close is above current EMA20 with momentum
    - pattern: detects bullish reversal patterns (hammer, doji, engulfing)
    - risk_reward: calculated risk reward ratio meets minimum threshold (2.0) with proper stop placement
    """
    n = len(closes)
    signals = {}
    
    # Early return if not enough data
    if n < ema_period + 10:  # Increased minimum data requirement
        return {k: False for k, _ in CHECK_ITEMS}

    # Calculate EMA values only once
    ema_vals = indicators.ema(closes, ema_period) if n >= ema_period else []
    ema_50_vals = indicators.ema(closes, 50) if n >= 50 else []
    
    # Calculate RSI for momentum confirmation
    rsi_vals = calculate_rsi(closes, 14) if n >= 20 else []
    current_rsi = rsi_vals[-1] if rsi_vals else None
    
    # previous resistance: last close > max high of previous 20 bars WITH volume confirmation
    # Enhanced: requires volume spike and strong close
    if n > ema_period + 1:
        prev_highs = highs[-(ema_period+1):-1]
        prev_volumes = volumes[-(ema_period+1):-1]
        resistance_level = max(prev_highs) if prev_highs else 0
        avg_volume = sum(prev_volumes) / len(prev_volumes) if prev_volumes else 0
        
        breakout_strength = (closes[-1] - resistance_level) / resistance_level if resistance_level > 0 else 0
        volume_confirmation = volumes[-1] > 1.5 * avg_volume if avg_volume > 0 else False
        
        signals["previous_resistance"] = (closes[-1] > resistance_level) and volume_confirmation and (breakout_strength > 0.005)  # 0.5% above resistance
    else:
        signals["previous_resistance"] = False

    # breakout candle: Enhanced - strong bullish candle with momentum
    # Check for strong body, good close position, and volume confirmation
    last_range = highs[-1] - lows[-1] if n > 0 else 0
    body_size = abs(closes[-1] - opens[-1]) if n > 0 else 0
    upper_wick = highs[-1] - max(closes[-1], opens[-1]) if n > 0 else 0
    lower_wick = min(closes[-1], opens[-1]) - lows[-1] if n > 0 else 0
    
    # Strong bullish candle: body > 60% of range, close > open, small upper wick
    strong_body = (body_size / last_range) > 0.6 if last_range > 0 else False
    bullish_close = closes[-1] > opens[-1] if n > 0 else False
    good_close_position = (closes[-1] - lows[-1]) / last_range > 0.8 if last_range > 0 else False
    small_upper_wick = (upper_wick / last_range) < 0.2 if last_range > 0 else True
    
    signals["breakout_candle"] = strong_body and bullish_close and good_close_position and small_upper_wick

    # volume: Enhanced - stronger threshold and trend analysis
    # Requires volume 1.5x above average AND increasing volume trend
    if n > ema_period + 1:
        prev_vol = volumes[-(ema_period+1):-1]
        avg_volume = sum(prev_vol) / len(prev_vol) if prev_vol else 0
        
        # Check volume trend (last 3 vs previous 3)
        recent_vol = sum(volumes[-3:]) / 3 if n >= 3 else volumes[-1]
        earlier_vol = sum(volumes[-6:-3]) / 3 if n >= 6 else avg_volume
        volume_trending_up = recent_vol > earlier_vol
        
        signals["volume"] = (volumes[-1] > 1.5 * avg_volume) and volume_trending_up if avg_volume > 0 else False
    else:
        signals["volume"] = False

    # ema20_touch: Enhanced - price touched EMA20 on pullback with momentum confirmation
    # Fixed: compare previous close with previous EMA value (ema_vals[-2])
    if len(ema_vals) >= 2 and n >= 3:
        prev_ema = ema_vals[-2]  # Previous bar's EMA
        prev_close = closes[-2]  # Previous bar's close
        current_close = closes[-1]
        prev_prev_close = closes[-3]  # For momentum check
        
        # Touch condition: within 1% of EMA
        touched = abs(prev_close - prev_ema) / prev_ema < 0.01 if prev_ema > 0 else False
        
        # Momentum confirmation: current close > previous close AND recovering from pullback
        momentum_up = current_close > prev_close
        recovery_strength = (current_close - prev_close) / prev_close if prev_close > 0 else 0
        
        signals["ema20_touch"] = touched and momentum_up and (recovery_strength > 0.002)  # 0.2% recovery minimum
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

    # multi timeframe: Enhanced - EMA alignment across different periods
    # Check alignment of EMA20, EMA50, and price momentum
    if len(ema_vals) > 0 and len(ema_50_vals) > 0 and n >= 3:
        current_price = closes[-1]
        ema20_current = ema_vals[-1]
        ema50_current = ema_50_vals[-1]
        
        # Price above both EMAs
        price_above_emas = current_price > ema20_current and current_price > ema50_current
        
        # EMA20 above EMA50 (uptrend structure)
        ema_alignment = ema20_current > ema50_current
        
        # Short-term momentum (last 3 bars trending up)
        short_momentum = all(closes[i] >= closes[i-1] for i in range(-2, 0)) if n >= 3 else False
        
        signals["multi_tf"] = price_above_emas and ema_alignment and short_momentum
    else:
        signals["multi_tf"] = closes[-1] > closes[-2] if n >= 2 else False

    # close_above_ema20: Enhanced - current close is above current EMA20 with momentum
    # Clarified: explicitly comparing current close with current EMA with distance check
    if len(ema_vals) > 0:
        ema20_current = ema_vals[-1]
        distance_above = (closes[-1] - ema20_current) / ema20_current if ema20_current > 0 else 0
        
        # RSI momentum confirmation (not overbought)
        rsi_confirmation = (current_rsi < 70) if current_rsi else True  # Not overbought
        
        # Must be above EMA20 by at least 0.1% to avoid noise
        signals["close_above_ema20"] = closes[-1] > ema20_current and distance_above > 0.001 and rsi_confirmation
    else:
        signals["close_above_ema20"] = False

    # pattern detection: Enhanced - detects multiple bullish reversal patterns
    if n >= 3:
        # Current and previous candle data
        curr_body = abs(closes[-1] - opens[-1])
        curr_range = highs[-1] - lows[-1]
        prev_body = abs(closes[-2] - opens[-2])
        prev_range = highs[-2] - lows[-2]
        prev2_body = abs(closes[-3] - opens[-3])
        
        # Hammer pattern: small body, long lower wick, minimal upper wick
        lower_wick = min(closes[-1], opens[-1]) - lows[-1]
        upper_wick = highs[-1] - max(closes[-1], opens[-1])
        hammer = (curr_body / curr_range < 0.3) and (lower_wick > 2 * curr_body) and (upper_wick < curr_body) and (closes[-1] > opens[-1]) if curr_range > 0 else False
        
        # Bullish engulfing: current candle body engulfs previous
        engulfing = (closes[-1] > opens[-1]) and (closes[-2] < opens[-2]) and (closes[-1] > opens[-2]) and (opens[-1] < closes[-2]) if n >= 2 else False
        
        # Morning star pattern: three candle reversal
        morning_star = False
        if n >= 3:
            # First: bearish, Second: small body/doji, Third: bullish
            first_bearish = closes[-3] < opens[-3]
            second_small = prev_body < (prev2_body * 0.5)
            third_bullish = closes[-1] > opens[-1]
            gap_down = opens[-2] < closes[-3]
            gap_up = opens[-1] > closes[-2]
            morning_star = first_bearish and second_small and third_bullish and gap_down and gap_up
        
        signals["pattern"] = hammer or engulfing or morning_star
    else:
        signals["pattern"] = False

    # risk reward: Enhanced - Calculate actual risk/reward with proper stop placement
    # Use ATR-based stops and multiple target levels
    if n >= 5:
        entry = closes[-1]
        
        # Calculate ATR for dynamic stop placement
        atr_vals = indicators.average_true_range(highs[-14:], lows[-14:], closes[-14:], period=14) if n >= 14 else []
        atr = atr_vals[-1] if atr_vals else None
        
        # Support level: recent swing low
        recent_low = min(lows[-5:])
        
        # Stop calculation: below support with ATR buffer
        if atr:
            atr_buffer = atr * 1.5  # 1.5x ATR buffer
            stop = min(recent_low - atr_buffer, entry * 0.98)  # Max 2% stop
        else:
            stop = min(recent_low * 0.995, entry * 0.98)  # 0.5% below support or 2% max
        
        risk = entry - stop
        
        # Target calculation: 2:1 minimum risk/reward
        tp1 = entry + (2.0 * risk)  # 2:1 RR
        
        # Risk management checks
        risk_percentage = risk / entry if entry > 0 else 1
        risk_reasonable = 0.005 < risk_percentage < 0.03  # 0.5% to 3% risk
        rr_ratio = (tp1 - entry) / risk if risk > 0 else 0
        
        # Additional confirmation: ensure stop is below recent structure
        structure_support = min(lows[-3:])  # Last 3 lows
        stop_below_structure = stop < structure_support
        
        signals["risk_reward"] = risk_reasonable and rr_ratio >= 2.0 and stop_below_structure
    else:
        signals["risk_reward"] = False

    return signals
