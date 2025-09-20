import os
import asyncio
import logging

try:
    import discord
except Exception:  # pragma: no cover - optional dependency
    discord = None

logger = logging.getLogger(__name__)


async def _send(token: str, channel_id: int, content: str) -> None:
    if discord is None:
        raise RuntimeError("discord.py is not installed")

    intents = discord.Intents.none()
    # We only need the ability to send messages.
    client = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        try:
            channel = client.get_channel(channel_id)
            if channel is None:
                # try fetch
                channel = await client.fetch_channel(channel_id)
            await channel.send(content)
        except Exception as exc:
            logger.exception("Failed to send discord message: %s", exc)
        finally:
            await client.close()

    await client.start(token)


def send_discord_message(content: str) -> None:
    """Send a short message to the channel configured via env vars.

    Reads `DISCORD_TOKEN` and `DISCORD_CHANNEL_ID` from the environment. This
    function is synchronous and will run an asyncio event loop to deliver the
    message. It raises RuntimeError when `discord.py` is not installed or when
    env vars are missing.
    """
    token = os.getenv("DISCORD_TOKEN")
    channel = os.getenv("DISCORD_CHANNEL_ID")
    if not token or not channel:
        raise RuntimeError("DISCORD_TOKEN and DISCORD_CHANNEL_ID must be set in env")

    try:
        channel_id = int(channel)
    except ValueError:
        raise RuntimeError("DISCORD_CHANNEL_ID must be an integer channel id")

    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_send(token, channel_id, content))
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception:
            pass
        loop.close()


def process_symbols(symbols_subset, ex_client, timeframe, limit_bars, rate_limit_sleep,
                   debug_mode, min_score_percent, realtime_csv, output_path, csv_headers, discord_notify=False):
    """Process symbols and optionally write to CSV in real-time"""
    results = []
    total = len(symbols_subset)
    filtered_count = 0

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

            # Check if score meets minimum threshold
            percent = detail.get("percent", 0)
            if percent < min_score_percent:
                if debug_mode:
                    print(f"⚠ {sym}: {percent}% (below {min_score_percent}% threshold)")
                continue  # Skip this result

            filtered_count += 1
            current_time = pd.Timestamp.now().isoformat()
            row = {"timestamp": current_time, "symbol": sym, **scores}
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
                "error": ""
            })
            results.append(row)

            if debug_mode:
                print(f"✅ {sym}: total={scores.get('total', 0)}, percent={percent}%")

            # Write to CSV immediately if real-time mode
            if realtime_csv:
                import csv
                with open(output_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=csv_headers)
                    writer.writerow(row)
                    f.flush()  # Ensure immediate write

            # Send discord notification if enabled
            if discord_notify:
                try:
                    msg = f"[{current_time}] {sym} — score: {scores.get('total', 0)} pts, {percent}% | entry={row.get('entry'):.6f} tp1={row.get('tp1'):.6f}"
                    send_discord_message(msg)
                except Exception as exc:
                    if debug_mode:
                        print(f"Discord notify failed for {sym}: {exc}")

        except Exception as e:
            error_msg = str(e)
            # log more detail for missing symbols
            if "does not have market symbol" in error_msg or "could not fetch ohlcv" in error_msg:
                if debug_mode:
                    print(f"❌ {sym}: {error_msg}")
            elif debug_mode:
                print(f"❌ {sym}: {error_msg}")
            # Don't add errors to results when filtering by score

        if rate_limit_sleep > 0:
            time.sleep(rate_limit_sleep)

        # print progress
        pct = round((idx / total) * 100, 1) if total else 0
        status = f"✅" if sym in [r["symbol"] for r in results] else "⚠"
        print(f"{status} {idx}/{total} ({pct}%) - {sym}")

    # Write final results if not in real-time mode
    if not realtime_csv and results:
        out = pd.DataFrame(results)
        out.to_csv(output_path, index=False)
        print(f"\nWrote {len(results)} results to {output_path}")
