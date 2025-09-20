import os
import pytest
from pathlib import Path

try:
    from scripts.ccxt_client import ExchangeClient
except Exception:
    ExchangeClient = None

from trading_checklist.utils import read_symbols


@pytest.mark.skipif(ExchangeClient is None, reason="ExchangeClient not available")
def test_symbols_exist_in_exchange():
    repo_root = Path(__file__).resolve().parents[1]
    symbols_path = repo_root / "symbols.txt"
    symbols = read_symbols(symbols_path)

    client = ExchangeClient(rate_limit=False)

    missing = []
    for s in symbols:
        try:
            # try futures first
            if client.futures:
                if s in client.futures.markets:
                    continue
            if s in client.spot.markets:
                continue
            # try some common variants
            variants = [s.replace('/USDT','/USDT'), s.replace('/USDT','/USDT:USDT')]
            found = False
            for v in variants:
                if (client.futures and v in client.futures.markets) or (v in client.spot.markets):
                    found = True
                    break
            if not found:
                missing.append(s)
        except Exception:
            missing.append(s)

    if missing:
        pytest.fail(f"Missing {len(missing)} symbols in exchange markets: {missing[:20]}")