import math
from trading_checklist import indicators


def test_ema_simple():
    vals = [1, 2, 3, 4, 5]
    e = indicators.ema(vals, 3)
    assert len(e) == len(vals)
    # first value equals first input
    assert e[0] == 1


def test_sma_simple():
    vals = [1, 2, 3, 4, 5]
    s = indicators.sma(vals, 3)
    assert s == [sum(vals[:3]) / 3, sum(vals[1:4]) / 3, sum(vals[2:5]) / 3]


def test_adx_returns_list():
    highs = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    lows = [0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5,10.5,11.5,12.5,13.5,14.5,15.5]
    closes = [0.9,1.9,2.9,3.9,4.9,5.9,6.9,7.9,8.9,9.9,10.9,11.9,12.9,13.9,14.9,15.9]
    a = indicators.adx(highs, lows, closes, period=5)
    assert isinstance(a, list)
