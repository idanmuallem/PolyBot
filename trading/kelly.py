"""The Kelly formula for sizing a single binary bet.

Caps on the result (the max-bet ceiling, daily budgets) are applied by the
Account (trading/account.py, Account.reserve()), not here.
"""


def kelly_bet_size(
    edge: float, odds: float, confidence: float, kelly_fraction: float, bankroll_usd: float,
) -> float:
    """Fractional-Kelly bet size in dollars, before any cap.

    The confidence-scale -> kelly_fraction-scale chain mirrors oracle3's
    ModelInformedSizer.compute_size() (oracle3/trading/sizing.py) -
    https://github.com/YichengYang-Ethan/oracle3, licensed under the
    Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0).

    kelly_optimal = edge / odds: the binary-outcome form of the Kelly
    criterion f* = (p*b - q) / b, where `edge` is (true_prob - price) and
    `odds` is the price move against you if wrong (1 - price for a YES bet
    at `price`, or `price` for the equivalent NO bet).

    Scaled down by model confidence (clamped to [0, 1]) and kelly_fraction
    (quarter-Kelly by default: full Kelly is optimal in expectation but has
    extreme variance). Returns 0.0 for a non-positive edge or degenerate
    odds, so it never sizes a bet against the edge.
    """
    if odds <= 0.0 or edge <= 0.0:
        return 0.0
    kelly_raw = edge / odds
    kelly_scaled = kelly_raw * max(0.0, min(1.0, float(confidence))) * float(kelly_fraction)
    return max(0.0, kelly_scaled * float(bankroll_usd))
