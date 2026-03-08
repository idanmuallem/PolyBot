import re
from typing import Optional

def extract_crypto_strike(text: str, anchor: float) -> Optional[float]:
    """Extract the most logical crypto strike from market text."""
    if not text:
        return None

    clean_text = text.upper()

    # 1. Strip out future years so they aren't confused as prices
    for y in ["2024", "2025", "2026", "2027", "2028"]:
        clean_text = clean_text.replace(y, "")

    # 2. Strip out currency symbols and commas
    clean_text = clean_text.replace(",", "").replace("$", "")

    # 3. Convert M (Millions) and B (Billions) into standard zeroes
    clean_text = re.sub(r'(\d+)M', r'\g<1>000000', clean_text)
    clean_text = re.sub(r'(\d+)B', r'\g<1>000000000', clean_text)

    # 4. Extract all remaining numbers
    matches = re.findall(r"(\d+(?:\.\d+)?)", clean_text)

    if not matches:
        print(f"[Parser] FAILED to extract strike from: {text}")
        return None

    try:
        numbers = [float(m) for m in matches]
        valid_candidates = []

        # 5. Sanity Check: Throw out dates and noise using a ratio
        for num in numbers:
            if anchor and anchor > 0:
                ratio = num / anchor
                # The strike must be between 5% and 5,000% of the live price
                # e.g., If BTC is 65k, this allows strikes from $3,250 to $3.25 Million
                # This instantly deletes dates like "31" (ratio = 0.0004)
                if 0.05 <= ratio <= 50.0:
                    valid_candidates.append(num)
            else:
                valid_candidates.append(num)

        if not valid_candidates:
            print(f"[Parser] FAILED sanity check (all numbers were dates/noise): {text}")
            return None

        # 6. Now pick the valid number closest to the live price
        if anchor and anchor > 0:
            return min(valid_candidates, key=lambda x: abs(x - anchor))
        else:
            return max(valid_candidates)

    except Exception as e:
        print(f"[Parser] Exception during parsing: {e}")
        return None
