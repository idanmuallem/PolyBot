"""Strike value extraction from market text, one function per asset domain."""

import re
from typing import Optional


# ── Economy ──────────────────────────────────────────────────────────────────

_PCT     = re.compile(r"(\d{1,4}(?:\.\d{1,2})?)\s*(%|percent)")
_BPS     = re.compile(r"(\d{1,4})\s*bps?\b")
_DECIMAL = re.compile(r"(\d{1,3}(?:\.\d{1,2})?)(?=\s|$|(\+|\-|rate|percent|level))")


def extract_economy_strike(question: str, anchor_val: float) -> Optional[float]:
    """Return a strike embedded in *question* within ~5 units of *anchor_val*."""
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


# ── Crypto ───────────────────────────────────────────────────────────────────

def extract_crypto_strike(text: str, anchor: float) -> Optional[float]:
    """Return the most logical crypto strike from market text."""
    if not text:
        return None

    clean = text.upper()
    for y in ["2024", "2025", "2026", "2027", "2028"]:
        clean = clean.replace(y, "")
    clean = clean.replace(",", "").replace("$", "")
    clean = re.sub(r"(\d+)M", r"\g<1>000000", clean)
    clean = re.sub(r"(\d+)B", r"\g<1>000000000", clean)

    matches = re.findall(r"(\d+(?:\.\d+)?)", clean)
    if not matches:
        print(f"[Parser] FAILED to extract strike from: {text}")
        return None

    try:
        numbers = [float(m) for m in matches]
        if anchor and anchor > 0:
            valid = [n for n in numbers if 0.05 <= n / anchor <= 50.0]
        else:
            valid = numbers

        if not valid:
            print(f"[Parser] FAILED sanity check (all numbers were dates/noise): {text}")
            return None

        return min(valid, key=lambda x: abs(x - anchor)) if anchor and anchor > 0 else max(valid)
    except Exception as e:
        print(f"[Parser] Exception during parsing: {e}")
        return None
