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
import requests
from zoneinfo import ZoneInfo

# timezone for all outputs
TH_TZ = ZoneInfo("Asia/Bangkok")


def now_iso():
    """Return current time in Asia/Bangkok as ISO string (with offset)."""
    return pd.Timestamp.now(tz=TH_TZ).isoformat()


# Ensure repository root is on sys.path so local package `trading_checklist` is importable
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from trading_checklist.utils import load_symbols, read_symbols
from trading_checklist.engine import prepare_df, evaluate_df
from trading_checklist.scoring import detailed_result, evaluate_bar_series, CHECK_ITEMS
from scripts.ccxt_client import ExchangeClient
from scripts.discord_notify import send_discord_message, accumulate_message, send_bulk_discord_message
from trading_checklist.indicators import ema, adx


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


def analyze_btc_trends(ex_client, debug_mode=False):
    """Analyze BTC/USDT trends across multiple timeframes with advanced indicators for maximum accuracy."""
    symbol = "BTC/USDT"
    timeframes = ["15m", "1h", "4h", "1d"]
    trends = []
    
    for tf in timeframes:
        try:
            ohlcv = ex_client.fetch_ohlcv(symbol, timeframe=tf, limit=200)  # more data for accurate calculations
            if not ohlcv:
                trends.append({"timeframe": tf, "trend": "N/A", "strength": "N/A", "cross": "N/A", "rsi": "N/A", "macd": "N/A", "adx": "N/A", "ema20": None, "ema50": None, "confidence": "N/A"})
                continue
            df = ohlcv_to_df(ohlcv)
            closes = df["close"].tolist()
            highs = df["high"].tolist()
            lows = df["low"].tolist()
            volumes = df["volume"].tolist()
            if len(closes) < 100:
                trends.append({"timeframe": tf, "trend": "N/A", "strength": "N/A", "cross": "N/A", "rsi": "N/A", "macd": "N/A", "adx": "N/A", "ema20": None, "ema50": None, "confidence": "N/A"})
                continue
            
            # Calculate indicators using indicators.py
            ema20_vals = ema(closes, 20)
            ema50_vals = ema(closes, 50)
            ema20 = ema20_vals[-1] if ema20_vals else None
            ema50 = ema50_vals[-1] if ema50_vals else None
            
            # RSI calculation (14 period) - custom implementation
            rsi_vals = calculate_rsi(closes, 14)
            rsi_val = rsi_vals[-1] if rsi_vals else None
            
            # MACD calculation (12,26,9) - using ema from indicators.py
            macd_line, signal_line, hist = calculate_macd(closes, 12, 26, 9)
            macd_signal = "Bullish" if hist and hist[-1] > 0 else "Bearish" if hist and hist[-1] < 0 else "Neutral"
            
            # ADX calculation (14 period) - using adx from indicators.py
            adx_vals = adx(highs, lows, closes, 14)
            adx_val = adx_vals[-1] if adx_vals else None
            
            # Trend determination with advanced logic
            if ema20 and ema50:
                base_trend = "📈 Uptrend" if ema20 > ema50 else "📉 Downtrend"
                
                # Strength based on EMA slope, ADX, and volume trend
                ema20_slope = (ema20_vals[-1] - ema20_vals[-5]) / 5 if len(ema20_vals) >= 5 else 0
                ema50_slope = (ema50_vals[-1] - ema50_vals[-5]) / 5 if len(ema50_vals) >= 5 else 0
                avg_slope = (ema20_slope + ema50_slope) / 2
                
                # Volume trend (last 5 candles)
                volume_trend = sum(volumes[-5:]) / sum(volumes[-10:-5]) if len(volumes) >= 10 else 1
                
                adx_strength = "Strong" if adx_val and adx_val > 25 else "Weak" if adx_val and adx_val < 20 else "Neutral"
                
                slope_score = abs(avg_slope) * 1000  # Scale for comparison
                volume_score = volume_trend
                
                if slope_score > 3 and adx_strength == "Strong" and volume_score > 1.1:
                    strength = "Very Strong"
                elif (slope_score > 2 or adx_strength == "Strong") and volume_score > 1.0:
                    strength = "Strong"
                elif slope_score > 1 or adx_strength == "Neutral":
                    strength = "Moderate"
                else:
                    strength = "Weak"
                
                # EMA Cross detection with confirmation
                cross = "No Cross"
                if len(ema20_vals) >= 2 and len(ema50_vals) >= 2:
                    prev_ema20 = ema20_vals[-2]
                    prev_ema50 = ema50_vals[-2]
                    current_close = closes[-1]
                    
                    if prev_ema20 <= prev_ema50 and ema20 > ema50:
                        cross = "Bullish Cross"
                        # Confirm with close above EMA20
                        if current_close > ema20:
                            cross += " (Confirmed)"
                    elif prev_ema20 >= prev_ema50 and ema20 < ema50:
                        cross = "Bearish Cross"
                        # Confirm with close below EMA20
                        if current_close < ema20:
                            cross += " (Confirmed)"
                
                # Confidence based on multiple factors
                confidence_score = 0
                
                # EMA alignment (5 points)
                if base_trend == "📈 Uptrend":
                    if ema20 > ema50 and closes[-1] > ema20:
                        confidence_score += 5
                    elif ema20 > ema50:
                        confidence_score += 3
                else:
                    if ema20 < ema50 and closes[-1] < ema20:
                        confidence_score += 5
                    elif ema20 < ema50:
                        confidence_score += 3
                
                # Cross signal (3 points)
                if "Cross" in cross and "Confirmed" in cross:
                    confidence_score += 3
                elif "Cross" in cross:
                    confidence_score += 2
                
                # Strength (3 points)
                strength_points = {"Very Strong": 3, "Strong": 2, "Moderate": 1, "Weak": 0}
                confidence_score += strength_points.get(strength, 0)
                
                # RSI confirmation (2 points)
                if rsi_val:
                    if (base_trend == "📈 Uptrend" and 30 < rsi_val < 70) or (base_trend == "📉 Downtrend" and 30 < rsi_val < 70):
                        confidence_score += 2
                    elif (base_trend == "📈 Uptrend" and rsi_val < 30) or (base_trend == "📉 Downtrend" and rsi_val > 70):
                        confidence_score -= 1  # Overbought/oversold reduces confidence
                
                # MACD confirmation (2 points)
                if macd_signal == ("Bullish" if base_trend == "📈 Uptrend" else "Bearish"):
                    confidence_score += 2
                
                # ADX confirmation (2 points)
                if adx_strength == "Strong":
                    confidence_score += 2
                
                # Volume confirmation (1 point)
                if volume_score > 1.05:
                    confidence_score += 1
                
                # Determine confidence level
                if confidence_score >= 12:
                    confidence = "High"
                elif confidence_score >= 8:
                    confidence = "Medium"
                else:
                    confidence = "Low"
                
                trend = base_trend
            else:
                trend = "N/A"
                strength = "N/A"
                cross = "N/A"
                confidence = "N/A"
            
            trends.append({"timeframe": tf, "trend": trend, "strength": strength, "cross": cross, "rsi": f"{rsi_val:.1f}" if rsi_val else "N/A", "macd": macd_signal, "adx": f"{adx_val:.1f}" if adx_val else "N/A", "ema20": ema20, "ema50": ema50, "confidence": confidence})
        except Exception as e:
            if debug_mode:
                print(f"Error analyzing BTC {tf}: {e}")
            trends.append({"timeframe": tf, "trend": "Error", "strength": "N/A", "cross": "N/A", "rsi": "N/A", "macd": "N/A", "adx": "N/A", "ema20": None, "ema50": None, "confidence": "N/A"})
    
    # Overall trend with advanced multi-timeframe analysis
    overall_trend = "Neutral"
    overall_strength = "N/A"
    overall_confidence = "N/A"
    if trends:
        weights = {"15m": 1, "1h": 2, "4h": 3, "1d": 4}  # Higher TF has more weight
        up_score = 0
        down_score = 0
        strength_scores = []
        confidence_scores = []
        
        for t in trends:
            weight = weights.get(t["timeframe"], 1)
            if "Uptrend" in t["trend"]:
                up_score += weight
            elif "Downtrend" in t["trend"]:
                down_score += weight
            
            # Aggregate strength and confidence with weights
            strength_map = {"Very Strong": 4, "Strong": 3, "Moderate": 2, "Weak": 1, "N/A": 0}
            confidence_map = {"High": 3, "Medium": 2, "Low": 1, "N/A": 0}
            strength_scores.append(strength_map.get(t["strength"], 0) * weight)
            confidence_scores.append(confidence_map.get(t["confidence"], 0) * weight)
        
        total_weight = sum(weights.values())
        avg_strength = sum(strength_scores) / total_weight if total_weight else 0
        avg_confidence = sum(confidence_scores) / total_weight if total_weight else 0
        
        # Determine overall trend with hysteresis (avoid whipsaws)
        trend_diff = up_score - down_score
        if trend_diff > 2:  # Clear uptrend
            overall_trend = "📈 Overall Uptrend"
        elif trend_diff < -2:  # Clear downtrend
            overall_trend = "📉 Overall Downtrend"
        elif up_score > down_score:
            overall_trend = "📈 Weak Uptrend"
        elif down_score > up_score:
            overall_trend = "📉 Weak Downtrend"
        
        # Overall strength
        if avg_strength >= 3.5:
            overall_strength = "Very Strong"
        elif avg_strength >= 2.5:
            overall_strength = "Strong"
        elif avg_strength >= 1.5:
            overall_strength = "Moderate"
        else:
            overall_strength = "Weak"
        
        # Overall confidence
        if avg_confidence >= 2.5:
            overall_confidence = "High"
        elif avg_confidence >= 1.5:
            overall_confidence = "Medium"
        else:
            overall_confidence = "Low"
    
    # Format as comprehensive Markdown table
    summary = f"**BTC/USDT Advanced Trend Analysis ({now_iso()})**\n"
    summary += f"**Overall: {overall_trend} | Strength: {overall_strength} | Confidence: {overall_confidence}**\n\n"
    summary += "| TF | Trend | Strength | Cross | RSI | MACD | ADX | EMA20 | EMA50 | Conf |\n"
    summary += "|----|-------|----------|-------|-----|------|-----|-------|-------|------|\n"
    for t in trends:
        ema20_str = f"{t['ema20']:.2f}" if t['ema20'] else "N/A"
        ema50_str = f"{t['ema50']:.2f}" if t['ema50'] else "N/A"
        summary += f"| {t['timeframe']} | {t['trend']} | {t['strength']} | {t['cross']} | {t['rsi']} | {t['macd']} | {t['adx']} | {ema20_str} | {ema50_str} | {t['confidence']} |\n"
    
    # Always send to Discord at end of cycle
    try:
        send_discord_message(summary)
        if debug_mode:
            print("Sent advanced BTC trends to Discord.")
    except Exception as exc:
        if debug_mode:
            print(f"Discord notify failed for BTC trends: {exc}")
    
    return summary

# Helper functions for additional indicators (added to run_batch.py)
def calculate_rsi(prices, period=14):
    """Calculate RSI indicator."""
    if len(prices) < period + 1:
        return None
    
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

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD indicator using ema from indicators.py."""
    fast_ema = ema(prices, fast)
    slow_ema = ema(prices, slow)
    macd_line = [f - s for f, s in zip(fast_ema, slow_ema)]
    signal_line = ema(macd_line, signal)
    hist = [m - s for m, s in zip(macd_line, signal_line)]
    return macd_line, signal_line, hist


def fetch_binance_symbols_once() -> list:
    """Fetch futures exchange symbols from Binance fapi exchangeInfo once.
    Returns symbols in form 'TOKEN/USDT:USDT'. On error returns empty list.
    """
    url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        out = set()
        for s in data.get("symbols", []):
            sym = s.get("symbol", "")
            # only consider USDT-quoted perpetual/symbols
            if sym.endswith("USDT"):
                base = sym[:-4]
                if base:
                    out.add(f"{base}/USDT:USDT")
        symbols = sorted(out)
        print(f"Fetched {len(symbols)} symbols from Binance fapi")
        return symbols
    except Exception as exc:
        print(f"Failed to fetch remote symbols from Binance: {exc}")
        return []


def main():
    repo_root = Path(__file__).resolve().parents[1]
    symbols_path = repo_root / "symbols.txt"

    # Environment configuration (ensure these are available before using symbols)
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
    loop_interval = int(os.environ.get("LOOP_INTERVAL", "300"))
    realtime_csv = os.environ.get("REALTIME_CSV", "false").lower() == "true"
    write_csv = os.environ.get("WRITE_CSV", "true").lower() == "true"
    enable_top3 = os.environ.get("ENABLE_TOP3", "true").lower() == "true"
    discord_notify = os.environ.get("DISCORD_NOTIFY", "false").lower() == "true"

    if debug_mode:
        print("Environment:")
        print(f"  RUN_LIMIT={run_limit}, TIMEFRAME={timeframe}, LIMIT_BARS={limit_bars}")
        print(f"  START_INDEX={start_index}, USE_FUTURES={use_futures}, DEBUG={debug_mode}")
        print(f"  WRITE_CSV={write_csv}, REALTIME_CSV={realtime_csv}, DISCORD_NOTIFY={discord_notify}")

    # try remote fetch first (one-time)
    symbols = fetch_binance_symbols_once()
    if not symbols:
        # fallback to local file (read_symbols returns 'TOKEN/USDT')
        if symbols_path.exists():
            symbols = read_symbols(symbols_path)
            print(f"Loaded {len(symbols)} symbols from {symbols_path}")
        else:
            print("No symbols available (remote fetch failed and symbols.txt missing). Exiting.")
            return

    # Apply start index and limit to the fetched symbols
    symbols_subset = symbols[start_index:start_index + run_limit] if run_limit > 0 else symbols[start_index:]
    # Debug: print summary of subset
    if debug_mode:
        end_idx = start_index + (run_limit if run_limit > 0 else len(symbols))
        print(f"Processing symbols {start_index}..{end_idx} (total fetched: {len(symbols)})")

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
                print(f"\n=== Loop {loop_count} started at {now_iso()} ===")
            
            process_symbols(symbols_subset, ex_client, timeframe, limit_bars, rate_limit_sleep, 
                          debug_mode, min_score_percent, realtime_csv, output_path, csv_headers, write_csv, enable_top3, discord_notify)
            
            # Analyze BTC trends at end of each cycle
            analyze_btc_trends(ex_client, debug_mode)
            
            if not loop_mode:
                break
                
            print(f"\n=== Loop {loop_count} completed at {now_iso()}. Sleeping {loop_interval}s ===")
            time.sleep(loop_interval)
            
    except KeyboardInterrupt:
        print(f"\n⏹ Stopped by user after {loop_count} loops")


def process_symbols(symbols_subset, ex_client, timeframe, limit_bars, rate_limit_sleep, 
                   debug_mode, min_score_percent, realtime_csv, output_path, csv_headers, write_csv=True, enable_top3=True, discord_notify=False):
    """Process symbols and optionally write to CSV in real-time.
    If no symbol meets min_score_percent, send top-3 results by percent to Discord.
    """
    results = []
    all_evaluated = []
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
            current_time = now_iso()
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

            all_evaluated.append(row)

            if percent < min_score_percent:
                if debug_mode:
                    print(f"⚠ {sym}: {percent}% (below {min_score_percent}% threshold)")
            else:
                filtered_count += 1
                results.append(row)
                if debug_mode:
                    print(f"✅ {sym}: total={scores.get('total', 0)}, percent={percent}%")

                if realtime_csv and write_csv:
                    with open(output_path, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=csv_headers)
                        writer.writerow(row)
                        f.flush()

                # Accumulate message instead of sending immediately
                if discord_notify:
                    try:
                        msg = format_trade_message(row)
                        accumulate_message(msg)
                    except Exception as exc:
                        if debug_mode:
                            print(f"Discord message accumulation failed for {sym}: {exc}")

        except Exception as e:
            error_msg = str(e)
            if debug_mode and ("does not have market symbol" in error_msg or "could not fetch ohlcv" in error_msg):
                print(f"❌ {sym}: {error_msg}")
            elif debug_mode:
                print(f"❌ {sym}: {error_msg}")

        if rate_limit_sleep > 0:
            time.sleep(rate_limit_sleep)

        pct = round((idx / total) * 100, 1) if total else 0
        status = "✅" if sym in [r["symbol"] for r in results] else "⚠"
        print(f"{status} {idx}/{total} ({pct}%) - {sym}")

    # Send accumulated messages at the end
    if discord_notify:
        try:
            # If we have results, send them
            if filtered_count > 0:
                send_bulk_discord_message()
                if debug_mode:
                    print(f"Sent {filtered_count} results to Discord in bulk.")
            
            # If no results and top3 enabled, send top3
            elif enable_top3 and all_evaluated:
                top3 = sorted(all_evaluated, key=lambda r: (r.get("percent") or 0), reverse=True)[:3]
                summary = format_top3_message(top3)
                send_discord_message(summary)
                if debug_mode:
                    print("Sent top-3 summary to Discord.")
        except Exception as exc:
            if debug_mode:
                print(f"Discord bulk send failed: {exc}")

    # Write top3 to CSV if needed
    if filtered_count == 0 and all_evaluated and enable_top3 and realtime_csv and write_csv:
        top3 = sorted(all_evaluated, key=lambda r: (r.get("percent") or 0), reverse=True)[:3]
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


# --- Added: Discord message formatting helpers ---
def _format_value(v):
    try:
        return f"{float(v):.6f}"
    except Exception:
        return str(v) if v is not None else "N/A"


def format_trade_message(row: dict) -> str:
    """Return a compact, readable message for a single symbol result."""
    symbol = row.get('symbol', '?')
    total_score = row.get('total', 0)
    percent = row.get('percent', 0)
    
    # Header line
    header = f"**{symbol}**\nTrade Plan Score: **{total_score}** pts • **{percent}%**"
    
    # Trade plan section
    plan_lines = [
        f"• Entry: {_format_value(row.get('entry'))}  • Stop: {_format_value(row.get('stop'))}",
        f"• TP1: {_format_value(row.get('tp1'))}  • TP2: {_format_value(row.get('tp2'))}",
        f"• Risk: {_format_value(row.get('risk'))}  • RR1: {_format_value(row.get('rr1'))}  • RR2: {_format_value(row.get('rr2'))}"
    ]
    
    # Combine with separator
    return header + "\n" + "\n".join(plan_lines) + "\n" + "-" * 43


def format_top3_message(rows: list) -> str:
    """Format a compact Top-3 summary."""
    lines = ["**Top 3 Candidates (No Symbol Met Threshold)**\n"]
    
    for i, r in enumerate(rows, start=1):
        symbol = r.get('symbol', '?')
        total_score = r.get('total', 0)
        percent = r.get('percent', 0)
        
        lines.append(f"**{i}. {symbol}**")
        lines.append(f"Score: **{total_score}** pts • **{percent}%**")
        lines.append(f"Entry: {_format_value(r.get('entry'))} • TP1: {_format_value(r.get('tp1'))} • Stop: {_format_value(r.get('stop'))}")
        lines.append("-" * 30)
    
    lines.append(f"*Generated: {now_iso()}*")
    return "\n".join(lines)
# --- end added helpers ---

if __name__ == "__main__":
    main()
