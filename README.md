Trading Checklist
=================

Simple Python package implementing the checklist described by the user.

Install
-------

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
