import re
from typing import Optional

# regex mirrors original pattern from CryptoHunter; allows optional '$' and
# skips absurdly large numbers by later ratio filtering.
_PATTERN = re.compile(r"(?:\$)?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*([kKmMbB])?")


def extract_crypto_strike(text: str, anchor: float) -> Optional[float]:
    """Scan ``text`` for a plausible crypto strike value.

    The function returns the first candidate whose ratio to ``anchor`` lies
    within the STRIKE_RATIO bounds (0.2–2.0).  If ``anchor`` is zero or
    missing, only syntactic extraction is performed.
    """
    candidates = []
    for match in _PATTERN.finditer(text):
        try:
            base = float(match.group(1).replace(",", ""))
        except Exception:
            continue
        suffix = (match.group(2) or "").lower()
        if suffix == "k":
            base *= 1_000
        elif suffix == "m":
            base *= 1_000_000
        elif suffix == "b":
            base *= 1_000_000_000
        candidates.append(base)

    # Ratio bounds for crypto strikes
    STRIKE_RATIO_MIN = 0.2
    STRIKE_RATIO_MAX = 2.0

    for val in candidates:
        try:
            ratio = val / anchor if anchor else 0
        except Exception:
            continue
        if STRIKE_RATIO_MIN < ratio < STRIKE_RATIO_MAX:
            return val
    return None
