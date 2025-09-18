from typing import List, Tuple
import re


def load_symbols(path: str) -> List[str]:
    out = []
    market_re = re.compile(r'([A-Za-z0-9\-_.]+/USDT(?::USDT)?)', re.IGNORECASE)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("//"):
                continue
            # take the first token (strip trailing comments/columns)
            token_raw = s.split()[0]
            # if it already looks like "TOKEN/USDT" or "TOKEN/USDT:USDT", keep it
            m = market_re.search(token_raw)
            if m:
                sym = m.group(1).upper()
                out.append(sym)
                continue
            # fallback: remove leading digits from a bare token and build futures-style pair
            base = token_raw.split("/")[0]
            base = re.sub(r'^\d+', '', base).strip().upper()
            if not base:
                continue
            out.append(f"{base}/USDT:USDT")
    return sorted(set(out))


def ensure_lengths(*lists) -> bool:
    lengths = [len(l) for l in lists]
    return all(l == lengths[0] for l in lengths)

# Added: simpler reader per user's logic
def read_symbols(path: str) -> List[str]:
    """Read symbols using logic: token before '/', strip leading digits, return 'TOKEN/USDT'."""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("//"):
                continue
            token = s.split("/")[0].strip().upper()
            token = re.sub(r'^\d+', '', token)
            if not token:
                continue
            out.append(f"{token}/USDT")
    return sorted(set(out))
