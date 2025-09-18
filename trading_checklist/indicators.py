from typing import List, Sequence
import math

def ema(values: Sequence[float], period: int) -> List[float]:
    if period <= 0:
        raise ValueError("period must be > 0")
    vals = list(values)
    if not vals:
        return []
    k = 2 / (period + 1)
    ema_vals = [vals[0]]
    for v in vals[1:]:
        ema_vals.append((v - ema_vals[-1]) * k + ema_vals[-1])
    return ema_vals

def sma(values: Sequence[float], period: int) -> List[float]:
    vals = list(values)
    if period <= 0:
        raise ValueError("period must be > 0")
    if len(vals) < period:
        return []
    out = []
    s = sum(vals[:period])
    out.append(s / period)
    for i in range(period, len(vals)):
        s += vals[i] - vals[i - period]
        out.append(s / period)
    return out

def average_true_range(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float], period: int) -> List[float]:
    if not (len(highs) == len(lows) == len(closes)):
        raise ValueError("highs, lows and closes must have same length")
    trs = []
    for i in range(len(highs)):
        if i == 0:
            tr = highs[i] - lows[i]
        else:
            tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        trs.append(tr)
    # use simple moving average for ATR
    return sma(trs, period)

def adx(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float], period: int = 14) -> List[float]:
    # Basic ADX implementation using Wilder's smoothing
    if not (len(highs) == len(lows) == len(closes)):
        raise ValueError("highs, lows and closes must have same length")
    length = len(highs)
    if length < period + 1:
        return []
    tr_list = []
    plus_dm = []
    minus_dm = []
    for i in range(1, length):
        up_move = highs[i] - highs[i-1]
        down_move = lows[i-1] - lows[i]
        pdm = up_move if (up_move > down_move and up_move > 0) else 0.0
        mdm = down_move if (down_move > up_move and down_move > 0) else 0.0
        plus_dm.append(pdm)
        minus_dm.append(mdm)
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        tr_list.append(tr)
    # Wilder's smoothing
    atr = []
    pdm_s = []
    mdm_s = []
    for i in range(len(tr_list)):
        if i == period - 1:
            atr.append(sum(tr_list[:period]) / period)
            pdm_s.append(sum(plus_dm[:period]) )
            mdm_s.append(sum(minus_dm[:period]) )
        elif i >= period:
            atr.append((atr[-1] * (period - 1) + tr_list[i]) / period)
            pdm_s.append((pdm_s[-1] - (pdm_s[-1] / period) + plus_dm[i]))
            mdm_s.append((mdm_s[-1] - (mdm_s[-1] / period) + minus_dm[i]))
    # compute di and dx
    di_plus = []
    di_minus = []
    dx = []
    for i in range(len(atr)):
        if atr[i] == 0:
            di_plus.append(0)
            di_minus.append(0)
            dx.append(0)
            continue
        p = 100 * (pdm_s[i] / atr[i])
        m = 100 * (mdm_s[i] / atr[i])
        di_plus.append(p)
        di_minus.append(m)
        dx.append(100 * abs(p - m) / (p + m) if (p + m) != 0 else 0)
    # smooth DX into ADX (Wilder)
    adx_vals = []
    for i in range(len(dx)):
        if i == period - 1:
            adx_vals.append(sum(dx[:period]) / period)
        elif i >= period:
            adx_vals.append((adx_vals[-1] * (period - 1) + dx[i]) / period)
    return adx_vals
