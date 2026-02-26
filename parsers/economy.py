import re
from typing import Optional

# three patterns used by the original EconomyHunter
_PCT = re.compile(r"(\d{1,4}(?:\.\d{1,2})?)\s*(%|percent)")
_BPS = re.compile(r"(\d{1,4})\s*bps?\b")
_DECIMAL = re.compile(r"(\d{1,3}(?:\.\d{1,2})?)(?=\s|$|(\+|\-|rate|percent|level))")


def extract_economy_strike(question: str, anchor_val: float) -> Optional[float]:
    """Look for a strike embedded in ``question``.

    Returns a float if a candidate is found that lies within ~5 units of
    ``anchor_val``; otherwise ``None``.
    """
    candidates = []

    for match in _PCT.finditer(question):
        try:
            candidates.append(float(match.group(1)))
        except Exception:
            pass

    for match in _BPS.finditer(question):
        try:
            candidates.append(float(match.group(1)) / 100.0)
        except Exception:
            pass

    for match in _DECIMAL.finditer(question):
        try:
            val = float(match.group(1))
            if val <= 100:
                candidates.append(val)
        except Exception:
            pass

    valid = [c for c in candidates if abs(c - anchor_val) < 5.0]
    return valid[0] if valid else None
