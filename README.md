Trading Checklist
=================

Simple Python package implementing the checklist described by the user.

      +-------------------------+
      |  Fetch USDT pairs       |
      |  from Binance API       |
      +-----------+-------------+
                  |
                  v
      +-------------------------+
      |  Analyze each pair      |
      |  Calculate breakhigh    |
      |  score using CHECK_ITEMS|
      +-----------+-------------+
                  |
                  v
      +-------------------------+
      |  Filter results         |
      |  - Score >= threshold   |
      |  - Top 3 if below      |
      +-----------+-------------+
                  |
                  v
      +-------------------------+
      |  Log results            |
      +-----------+-------------+
                  |
                  v
      +-------------------------+
      |  Send asynchronously    |
      |  to Discord             |
      +-------------------------+


Install
-------
<img width="1421" height="907" alt="image" src="https://github.com/user-attachments/assets/e6d3e437-0592-4919-9108-04508133bb77" />

Create a virtualenv and install pytest:

```bash
python -m venv .venv
source .venv/Scripts/activate  # on Windows use `.venv\\Scripts\\activate`
pip install -r requirements.txt
```

Run tests:

```bash
pytest -q
```

Files
-----
- `trading_checklist/indicators.py`: EMA, SMA, ATR, ADX implementations
- `trading_checklist/scoring.py`: checklist evaluation and scoring
- `trading_checklist/utils.py`: helpers
- `tests/`: pytest tests

Discord notifier
----------------
You can send notifications to a Discord channel by setting `DISCORD_TOKEN`
and `DISCORD_CHANNEL_ID` in your environment (or in a `.env` file at the repo root)
and calling the helper in `scripts/discord_notify.py`:

```python
from scripts.discord_notify import send_discord_message

send_discord_message("Hello from the trading checklist runner")
```

Install the dependency with `pip install -r requirements.txt` (the file now
contains `discord.py`).
