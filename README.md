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
