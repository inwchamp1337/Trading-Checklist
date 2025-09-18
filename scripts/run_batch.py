"""Batch runner: reads `symbols.txt`, loads CSV data if available, else generates synthetic data, evaluates and writes results.csv"""
import os
import time
from pathlib import Path
import re
import sys
import pandas as pd
import numpy as np
import ccxt

# Ensure repository root is on sys.path so local package `trading_checklist` is importable
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from trading_checklist.utils import load_symbols, read_symbols
from trading_checklist.engine import prepare_df, evaluate_df
from trading_checklist.scoring import detailed_result, evaluate_bar_series
from scripts.ccxt_client import ExchangeClient


def synth_df(seed=0, n=200):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    close = np.cumsum(rng.normal(0.1, 1.0, size=n)) + 100
    open_ = close - rng.normal(0, 0.5, size=n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.5, size=n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.5, size=n))
    volume = (rng.integers(80, 200, size=n)).astype(float)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume})


def normalize_symbol(s: str) -> str:
    # symbols.txt entries may include leading quantities like "1000BONK/USDT:USDT".
    # Remove a leading numeric quantity, non-letters, and extract token like 'ASSET/QUOTE'.
    left = s.split(":", 1)[0]
    # strip leading digits and non-alpha
    left = re.sub(r"^[0-9_\-\s]+", "", left)
    m = re.search(r"([A-Za-z0-9]+/[A-Za-z0-9]+)", left)
    if m:
        return m.group(1).upper()
    return left.upper()


def fetch_ohlcv_for_symbol(exchange, symbol: str, timeframe="1h", limit=250):
    # ccxt expects symbols like "BTC/USDT"
    return exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)


def ohlcv_to_df(ohlcv):
    # ccxt OHLCV: [timestamp, open, high, low, close, volume]
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df[["open", "high", "low", "close", "volume"]]


def main():
    repo_root = Path(__file__).resolve().parents[1]
    symbols_path = repo_root / "symbols.txt"
    
    # Environment configuration
    run_limit = int(os.environ.get("RUN_LIMIT", "20"))
    timeframe = os.environ.get("TIMEFRAME", "1h")
    limit_bars = int(os.environ.get("LIMIT_BARS", "300"))
    rate_limit_sleep = float(os.environ.get("RATE_LIMIT_SLEEP", "0.25"))
    output_file = os.environ.get("OUTPUT_FILE", "results_ccxt.csv")
    start_index = int(os.environ.get("START_INDEX", "0"))
    use_futures = os.environ.get("USE_FUTURES", "true").lower() == "true"
    debug_mode = os.environ.get("DEBUG", "false").lower() == "true"
    
    print(f"Configuration:")
    print(f"  Run limit: {run_limit}")
    print(f"  Timeframe: {timeframe}")
    print(f"  Bars limit: {limit_bars}")
    print(f"  Rate limit sleep: {rate_limit_sleep}s")
    print(f"  Output file: {output_file}")
    print(f"  Start index: {start_index}")
    print(f"  Use futures: {use_futures}")
    print(f"  Debug mode: {debug_mode}")
    print()
    
    # use the new read_symbols which returns 'TOKEN/USDT' normalized
    symbols = read_symbols(symbols_path)
    
    # Apply start index and limit
    symbols_subset = symbols[start_index:start_index + run_limit] if run_limit > 0 else symbols[start_index:]
    
    # exchange client wrapper (tries futures then spot)
    ex_client = ExchangeClient(rate_limit=True)
    if not use_futures:
        ex_client.futures = None  # force spot only

    results = []
    total = len(symbols_subset)
    
    for idx, sym in enumerate(symbols_subset, start=1):
        try:
            fetch_sym = sym  # already in 'TOKEN/USDT' form
            ohlcv = ex_client.fetch_ohlcv(fetch_sym, timeframe=timeframe, limit=limit_bars)
            if not ohlcv:
                raise RuntimeError("no ohlcv returned")
            df = ohlcv_to_df(ohlcv)
            pdf = prepare_df(df)
            scores = evaluate_df(pdf)
            # compute detailed result
            signals = evaluate_bar_series(pdf["high"].tolist(), pdf["low"].tolist(), pdf["open"].tolist(), pdf["close"].tolist(), pdf["volume"].tolist(), ema_period=20)
            detail = detailed_result(signals, pdf["close"].tolist(), pdf["low"].tolist(), pdf["high"].tolist(), ema_period=20)
            row = {"symbol": sym, **scores}
            # flatten detail fields to top-level
            row.update({
                "percent": detail.get("percent"),
                "entry": detail.get("entry"),
                "stop": detail.get("stop"),
                "tp1": detail.get("tp1"),
                "tp2": detail.get("tp2"),
                "risk": detail.get("risk"),
                "rr1": detail.get("rr1"),
                "rr2": detail.get("rr2"),
            })
            results.append(row)
            if debug_mode:
                print(f"✅ {sym}: total={scores.get('total', 0)}, percent={detail.get('percent', 0)}%")
        except Exception as e:
            error_msg = str(e)
            # log more detail for missing symbols
            if "does not have market symbol" in error_msg or "could not fetch ohlcv" in error_msg:
                print(f"❌ {sym}: {error_msg}")
            elif debug_mode:
                print(f"❌ {sym}: {error_msg}")
            results.append({"symbol": sym, "error": error_msg})
        
        if rate_limit_sleep > 0:
            time.sleep(rate_limit_sleep)
        
        # print progress
        pct = round((idx / total) * 100, 1) if total else 0
        print(f"{idx}/{total} ({pct}%) - {sym}")

    out = pd.DataFrame(results)
    output_path = repo_root / output_file
    out.to_csv(output_path, index=False)
    print(f"\nWrote {len(results)} results to {output_file}")
    
    # Summary statistics
    success_count = len([r for r in results if "error" not in r])
    error_count = len(results) - success_count
    print(f"Success: {success_count}, Errors: {error_count}")
    
    if success_count > 0:
        successful_results = [r for r in results if "error" not in r and "total" in r]
        if successful_results:
            avg_score = sum(r["total"] for r in successful_results) / len(successful_results)
            max_score = max(r["total"] for r in successful_results)
            print(f"Average score: {avg_score:.1f}, Max score: {max_score}")


if __name__ == "__main__":
    main()
