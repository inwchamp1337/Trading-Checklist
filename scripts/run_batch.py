"""Batch runner: reads `symbols.txt`, loads CSV data if available, else generates synthetic data, evaluates and writes results.csv"""
import os
import time
import csv
from pathlib import Path

# Load .env from workspace root (using python-dotenv)
try:
    from dotenv import load_dotenv
    repo_root = Path(__file__).resolve().parents[1]
    env_path = repo_root / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        print(f"Loaded .env from {env_path}")
    else:
        print(f"No .env found at {env_path}")
except ImportError:
    print("Tip: pip install python-dotenv for .env support")

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
from scripts.discord_notify import send_discord_message


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
    min_score_percent = float(os.environ.get("MIN_SCORE_PERCENT", "75.0"))
    loop_mode = os.environ.get("LOOP_MODE", "false").lower() == "true"
    loop_interval = int(os.environ.get("LOOP_INTERVAL", "300"))  # seconds between loops
    realtime_csv = os.environ.get("REALTIME_CSV", "false").lower() == "true"
    # New env toggles
    write_csv = os.environ.get("WRITE_CSV", "true").lower() == "true"
    enable_top3 = os.environ.get("ENABLE_TOP3", "true").lower() == "true"
    # Discord config from .env
    discord_notify = os.environ.get("DISCORD_NOTIFY", "false").lower() == "true"
    
    print(f"Configuration:")
    print(f"  Run limit: {run_limit}")
    print(f"  Timeframe: {timeframe}")
    print(f"  Bars limit: {limit_bars}")
    print(f"  Rate limit sleep: {rate_limit_sleep}s")
    print(f"  Output file: {output_file}")
    print(f"  Start index: {start_index}")
    print(f"  Use futures: {use_futures}")
    print(f"  Debug mode: {debug_mode}")
    print(f"  Min score percent: {min_score_percent}%")
    print(f"  Loop mode: {loop_mode}")
    if loop_mode:
        print(f"  Loop interval: {loop_interval}s")
    print(f"  Real-time CSV: {realtime_csv}")
    print()
    
    # use the new read_symbols which returns 'TOKEN/USDT' normalized
    symbols = read_symbols(symbols_path)
    
    # Apply start index and limit
    symbols_subset = symbols[start_index:start_index + run_limit] if run_limit > 0 else symbols[start_index:]
    
    # exchange client wrapper (tries futures then spot)
    ex_client = ExchangeClient(rate_limit=True)
    if not use_futures:
        ex_client.futures = None  # force spot only

    # Initialize CSV file with headers if real-time mode
    output_path = repo_root / output_file
    csv_headers = ["timestamp", "symbol", "previous_resistance", "breakout_candle", "volume", "ema20_touch", 
                   "fib_pullback", "adx", "multi_tf", "close_above_ema20", "pattern", "risk_reward", 
                   "total", "percent", "entry", "stop", "tp1", "tp2", "risk", "rr1", "rr2", "error"]
    
    if realtime_csv and write_csv and not output_path.exists():
        # Create file with headers
        import csv
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(csv_headers)

    loop_count = 0
    try:
        while True:
            loop_count += 1
            if loop_mode:
                print(f"\n=== Loop {loop_count} started at {pd.Timestamp.now()} ===")
            
            process_symbols(symbols_subset, ex_client, timeframe, limit_bars, rate_limit_sleep, 
                          debug_mode, min_score_percent, realtime_csv, output_path, csv_headers, write_csv, enable_top3, discord_notify)
            
            if not loop_mode:
                break
                
            print(f"\n=== Loop {loop_count} completed. Sleeping {loop_interval}s ===")
            time.sleep(loop_interval)
            
    except KeyboardInterrupt:
        print(f"\n⏹ Stopped by user after {loop_count} loops")


def process_symbols(symbols_subset, ex_client, timeframe, limit_bars, rate_limit_sleep, 
                   debug_mode, min_score_percent, realtime_csv, output_path, csv_headers, write_csv=True, enable_top3=True, discord_notify=False):
    """Process symbols and optionally write to CSV in real-time.
    If no symbol meets min_score_percent, send top-3 results by percent to Discord.
    """
    results = []              # passed the threshold -> will be written/sent per current behavior
    all_evaluated = []        # all evaluated candidates with percent for fallback top3
    total = len(symbols_subset)
    filtered_count = 0

    for idx, sym in enumerate(symbols_subset, start=1):
        try:
            fetch_sym = sym
            ohlcv = ex_client.fetch_ohlcv(fetch_sym, timeframe=timeframe, limit=limit_bars)
            if not ohlcv:
                raise RuntimeError("no ohlcv returned")
            df = ohlcv_to_df(ohlcv)
            pdf = prepare_df(df)
            scores = evaluate_df(pdf)
            signals = evaluate_bar_series(pdf["high"].tolist(), pdf["low"].tolist(), pdf["open"].tolist(), pdf["close"].tolist(), pdf["volume"].tolist(), ema_period=20)
            detail = detailed_result(signals, pdf["close"].tolist(), pdf["low"].tolist(), pdf["high"].tolist(), ema_period=20)

            percent = float(detail.get("percent", 0) or 0)
            current_time = pd.Timestamp.now().isoformat()
            row = {"timestamp": current_time, "symbol": sym, **scores}
            row.update({
                "percent": percent,
                "entry": detail.get("entry"),
                "stop": detail.get("stop"),
                "tp1": detail.get("tp1"),
                "tp2": detail.get("tp2"),
                "risk": detail.get("risk"),
                "rr1": detail.get("rr1"),
                "rr2": detail.get("rr2"),
                "error": ""
            })

            # record for fallback/top3 decision
            all_evaluated.append(row)

            if percent < min_score_percent:
                if debug_mode:
                    print(f"⚠ {sym}: {percent}% (below {min_score_percent}% threshold)")
                # continue evaluating next symbol (but keep row in all_evaluated)
            else:
                # passed threshold: keep and optionally send/write immediately
                filtered_count += 1
                results.append(row)
                if debug_mode:
                    print(f"✅ {sym}: total={scores.get('total', 0)}, percent={percent}%")

                if realtime_csv and write_csv:
                    with open(output_path, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=csv_headers)
                        writer.writerow(row)
                        f.flush()

                if discord_notify:
                    try:
                        def fmt(x):
                            try:
                                return f"{float(x):.8f}"
                            except Exception:
                                return str(x)
                        msg = (
                            f"[{current_time}] {sym} — {scores.get('total', 0)} pts, {percent}%\n"
                            f"entry={fmt(row.get('entry'))} tp1={fmt(row.get('tp1'))} tp2={fmt(row.get('tp2'))} sl={fmt(row.get('stop'))}"
                        )
                        send_discord_message(msg)
                    except Exception as exc:
                        if debug_mode:
                            print(f"Discord notify failed for {sym}: {exc}")

        except Exception as e:
            error_msg = str(e)
            if debug_mode and ("does not have market symbol" in error_msg or "could not fetch ohlcv" in error_msg):
                print(f"❌ {sym}: {error_msg}")
            elif debug_mode:
                print(f"❌ {sym}: {error_msg}")
            # keep going

        if rate_limit_sleep > 0:
            time.sleep(rate_limit_sleep)

        pct = round((idx / total) * 100, 1) if total else 0
        status = "✅" if sym in [r["symbol"] for r in results] else "⚠"
        print(f"{status} {idx}/{total} ({pct}%) - {sym}")

    # If nothing passed threshold, send top-3 by percent to Discord (and optionally append to CSV)
    if filtered_count == 0 and all_evaluated and enable_top3:
        top3 = sorted(all_evaluated, key=lambda r: (r.get("percent") or 0), reverse=True)[:3]
        if discord_notify:
            try:
                lines = []
                for r in top3:
                    def fmt(x):
                        try:
                            return f"{float(x):.8f}"
                        except Exception:
                            return str(x)
                    lines.append(f"{r['symbol']} — {r.get('total', 0)} pts, {r.get('percent', 0)}% | entry={fmt(r.get('entry'))} tp1={fmt(r.get('tp1'))} sl={fmt(r.get('stop'))}")
                summary = "No symbol met threshold. Top 3 candidates:\n" + "\n".join(lines)
                send_discord_message(summary)
                if debug_mode:
                    print("Sent top-3 summary to Discord.")
            except Exception as exc:
                if debug_mode:
                    print(f"Discord notify failed for top-3: {exc}")

        # optionally write top3 to realtime CSV (only when CSV writes enabled)
        if realtime_csv and write_csv:
            with open(output_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=csv_headers)
                for r in top3:
                    writer.writerow(r)
                f.flush()

    # Write final results if not in real-time mode
    if write_csv and not realtime_csv and results:
        out = pd.DataFrame(results)
        out.to_csv(output_path, index=False)
        print(f"\nWrote {len(results)} results to {output_path}")

    # Summary
    print(f"Filtered results (≥{min_score_percent}%): {filtered_count}")
    print(f"Total processed: {total}")
    if results:
        avg_score = sum(r["total"] for r in results if isinstance(r.get("total"), (int, float))) / len(results)
        max_score = max(r["total"] for r in results if isinstance(r.get("total"), (int, float)))
        print(f"Average score: {avg_score:.1f}, Max score: {max_score}")

    return results


if __name__ == "__main__":
    main()
