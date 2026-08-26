import pytest

from core.trading_config import TradingConfig
from trading.strategies.base import StrategySignal
from trading.strategies.event_sum import EventSumStrategy


def _market(token_id="tok1", price=0.30, closed=False, no_id="tok1_no", volume=100_000.0):
    return {
        "closed": closed,
        "clobTokenIds": [token_id, no_id],
        "lastTradePrice": price,
        "question": f"Outcome {token_id}?",
        "groupItemTitle": f"Outcome {token_id}",
        "volume": volume,
    }


def _event(event_id="evt1", title="Test Event", markets=None):
    return {
        "id": event_id,
        "title": title,
        "markets": markets or [],
    }


def _strategy(**kwargs):
    kwargs.setdefault("min_edge", 0.02)
    kwargs.setdefault("trade_size_usd", 1.0)
    kwargs.setdefault("fee_rate", 0.005)
    kwargs.setdefault("config", TradingConfig())
    return EventSumStrategy(**kwargs)


# ── Arbitrage condition met (underpriced) ────────────────────────────────────

@pytest.mark.asyncio
async def test_underpriced_event_emits_one_signal_per_leg():
    # 3 outcomes summing to 0.85 -> well below 1.0, plenty of edge after fees.
    event = _event(markets=[
        _market("tokA", 0.30),
        _market("tokB", 0.25),
        _market("tokC", 0.30),
    ])
    strategy = _strategy()

    signals = await strategy.scan([event])

    assert len(signals) == 3
    assert all(isinstance(s, StrategySignal) for s in signals)
    assert all(s.side == "YES" for s in signals)
    assert all(s.strategy_type == "arbitrage" for s in signals)
    assert {s.market.market_id for s in signals} == {"tokA", "tokB", "tokC"}
    assert {s.price for s in signals} == {0.30, 0.25, 0.30}


@pytest.mark.asyncio
async def test_underpriced_event_all_legs_share_group_id_and_edge():
    event = _event(event_id="evt42", markets=[_market("tokA", 0.30), _market("tokB", 0.30)])
    strategy = _strategy()

    signals = await strategy.scan([event])

    assert len(signals) == 2
    group_ids = {s.group_id for s in signals}
    edges = {s.edge for s in signals}
    assert group_ids == {"event_sum:evt42"}
    assert len(edges) == 1

    sum_yes = 0.60
    expected_edge = 1.0 - sum_yes - (strategy.fee_rate * 2)
    assert signals[0].edge == pytest.approx(expected_edge)


@pytest.mark.asyncio
async def test_bet_amount_matches_configured_trade_size():
    event = _event(markets=[_market("tokA", 0.30), _market("tokB", 0.30)])
    strategy = _strategy(trade_size_usd=2.5)

    signals = await strategy.scan([event])

    assert all(s.bet_amount_usd == pytest.approx(2.5) for s in signals)


# ── Arbitrage condition NOT met ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_fairly_priced_event_emits_no_signals():
    # Sums to ~1.0 -> no edge.
    event = _event(markets=[_market("tokA", 0.50), _market("tokB", 0.49)])
    strategy = _strategy()

    signals = await strategy.scan([event])

    assert signals == []


@pytest.mark.asyncio
async def test_overpriced_event_emits_no_signals():
    # Sums to > 1.0 -> event_sum (buy-YES-only) has nothing to do here;
    # that's exclusivity arbitrage's job, not implemented in this strategy.
    event = _event(markets=[_market("tokA", 0.60), _market("tokB", 0.60)])
    strategy = _strategy()

    signals = await strategy.scan([event])

    assert signals == []


@pytest.mark.asyncio
async def test_edge_below_min_edge_threshold_emits_no_signals():
    # sum=0.975 -> raw edge=0.025, minus fees (0.005 * 2 = 0.01) = 0.015,
    # which is under a min_edge of 0.02.
    event = _event(markets=[_market("tokA", 0.50), _market("tokB", 0.475)])
    strategy = _strategy(min_edge=0.02, fee_rate=0.005)

    signals = await strategy.scan([event])

    assert signals == []


@pytest.mark.asyncio
async def test_fee_rate_reduces_edge():
    event = _event(markets=[_market("tokA", 0.30), _market("tokB", 0.30)])
    cheap_fees = _strategy(fee_rate=0.0, min_edge=0.0)
    expensive_fees = _strategy(fee_rate=0.10, min_edge=0.0)

    cheap_signals = await cheap_fees.scan([event])
    expensive_signals = await expensive_fees.scan([event])

    assert cheap_signals[0].edge > expensive_signals[0].edge


# ── Outcome-count bounds ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_below_min_outcomes_emits_no_signals():
    event = _event(markets=[_market("tokA", 0.30)])  # only 1 outcome
    strategy = _strategy(min_outcomes=2)

    signals = await strategy.scan([event])

    assert signals == []


@pytest.mark.asyncio
async def test_above_max_outcomes_emits_no_signals():
    event = _event(markets=[_market(f"tok{i}", 0.01) for i in range(25)])
    strategy = _strategy(max_outcomes=20, min_edge=0.0)

    signals = await strategy.scan([event])

    assert signals == []


# ── Leg filtering ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_closed_markets_are_excluded():
    event = _event(markets=[
        _market("tokA", 0.30),
        _market("tokB", 0.30, closed=True),
        _market("tokC", 0.30),
    ])
    strategy = _strategy(min_outcomes=2, min_edge=0.0)

    signals = await strategy.scan([event])

    assert {s.market.market_id for s in signals} == {"tokA", "tokC"}


@pytest.mark.asyncio
async def test_missing_token_ids_excluded():
    market_no_tokens = _market("tokA", 0.30)
    market_no_tokens["clobTokenIds"] = None
    event = _event(markets=[market_no_tokens, _market("tokB", 0.30), _market("tokC", 0.30)])
    strategy = _strategy(min_outcomes=2, min_edge=0.0)

    signals = await strategy.scan([event])

    assert "tokA" not in {s.market.market_id for s in signals}


@pytest.mark.asyncio
async def test_degenerate_prices_excluded():
    event = _event(markets=[
        _market("tokA", 0.0),   # resolved NO / degenerate
        _market("tokB", 1.0),   # resolved YES / degenerate
        _market("tokC", 0.30),
        _market("tokD", 0.30),
    ])
    strategy = _strategy(min_outcomes=2, min_edge=0.0)

    signals = await strategy.scan([event])

    assert {s.market.market_id for s in signals} == {"tokC", "tokD"}


@pytest.mark.asyncio
async def test_legs_outside_price_floor_ceiling_are_not_filtered():
    # Multi-outcome events routinely have most legs priced well outside the
    # single-market PRICE_FLOOR/PRICE_CEILING (0.30-0.85) band -- that's
    # normal and must NOT be filtered out here.
    event = _event(markets=[
        _market("tokA", 0.02),
        _market("tokB", 0.05),
        _market("tokC", 0.15),
        _market("tokD", 0.60),
    ])
    strategy = _strategy(min_outcomes=2, min_edge=0.0)

    signals = await strategy.scan([event])

    assert {s.market.market_id for s in signals} == {"tokA", "tokB", "tokC", "tokD"}


# ── Multiple events / JSON-string clobTokenIds ────────────────────────────────

@pytest.mark.asyncio
async def test_scans_multiple_events_independently():
    underpriced = _event(event_id="evt1", markets=[_market("tokA", 0.30), _market("tokB", 0.30)])
    fair = _event(event_id="evt2", markets=[_market("tokC", 0.50), _market("tokD", 0.49)])
    strategy = _strategy()

    signals = await strategy.scan([underpriced, fair])

    assert {s.market.market_id for s in signals} == {"tokA", "tokB"}


@pytest.mark.asyncio
async def test_clob_token_ids_as_json_string():
    import json
    market = _market("tokA", 0.30)
    market["clobTokenIds"] = json.dumps(["tokA", "tokA_no"])
    event = _event(markets=[market, _market("tokB", 0.30)])
    strategy = _strategy(min_edge=0.0)

    signals = await strategy.scan([event])

    assert {s.market.market_id for s in signals} == {"tokA", "tokB"}


@pytest.mark.asyncio
async def test_empty_markets_list_returns_empty():
    strategy = _strategy()
    assert await strategy.scan([]) == []
    assert await strategy.scan([_event(markets=[])]) == []


# ── condition_id / slug propagation (paper-trading Bug 2 fix) ────────────────
# EventSumStrategy builds its own MarketData directly from the raw Gamma API
# event payload (unlike CryptoHunter/WeatherHunter/EconomyHunter, which all
# go through BasePolymarketHunter._scan_polymarket()) — it must independently
# carry condition_id/slug through, or every arbitrage leg silently fails to
# reach the paper engine (PaperAdapter.execute_buy() requires one of the two).

@pytest.mark.asyncio
async def test_leg_market_carries_condition_id_and_slug():
    market = _market("tokA", 0.30)
    market["conditionId"] = "0xcond-a"
    market["slug"] = "outcome-a-slug"
    event = _event(markets=[market, _market("tokB", 0.30)])
    strategy = _strategy()

    signals = await strategy.scan([event])

    leg_a = next(s for s in signals if s.market.market_id == "tokA")
    assert leg_a.market.condition_id == "0xcond-a"
    assert leg_a.market.slug == "outcome-a-slug"


@pytest.mark.asyncio
async def test_leg_market_falls_back_to_event_level_slug():
    # Per-market "slug" absent -> falls back to the event's own slug, same
    # pattern as hunters/base.py's _scan_polymarket().
    market = _market("tokA", 0.30)
    event = _event(event_id="evt-fallback", markets=[market, _market("tokB", 0.30)])
    event["slug"] = "event-level-slug"
    strategy = _strategy()

    signals = await strategy.scan([event])

    leg_a = next(s for s in signals if s.market.market_id == "tokA")
    assert leg_a.market.slug == "event-level-slug"


@pytest.mark.asyncio
async def test_leg_market_condition_id_and_slug_none_when_absent_from_payload():
    event = _event(markets=[_market("tokA", 0.30), _market("tokB", 0.30)])
    strategy = _strategy()

    signals = await strategy.scan([event])

    assert all(s.market.condition_id is None for s in signals)
    assert all(s.market.slug is None for s in signals)


# ── Phase 4: crypto-first scan order ──────────────────────────────────────────

def _crypto_first_events():
    # Two crypto events (matched via title keyword "bitcoin"/slug "eth-"),
    # interleaved with two general events, all equally underpriced.
    general_a = _event(event_id="gen_a", title="Who wins the election?",
                        markets=[_market("gA1", 0.30), _market("gA2", 0.30)])
    crypto_a = _event(event_id="crypto_a", title="Will Bitcoin hit $100k?",
                       markets=[_market("cA1", 0.30), _market("cA2", 0.30)])
    general_b = _event(event_id="gen_b", title="Next Fed rate decision",
                        markets=[_market("gB1", 0.30), _market("gB2", 0.30)])
    crypto_b = _event(event_id="crypto_b", title="Ethereum event",
                       markets=[_market("cB1", 0.30), _market("cB2", 0.30)])
    return [general_a, crypto_a, general_b, crypto_b]


@pytest.mark.asyncio
async def test_crypto_first_true_scans_crypto_events_before_general():
    strategy = _strategy(config=TradingConfig(arbitrage_crypto_first=True))

    signals = await strategy.scan(_crypto_first_events())

    group_order = list(dict.fromkeys(s.group_id for s in signals))
    assert group_order == [
        "event_sum:crypto_a", "event_sum:crypto_b",
        "event_sum:gen_a", "event_sum:gen_b",
    ]


@pytest.mark.asyncio
async def test_crypto_first_false_preserves_original_scan_order():
    strategy = _strategy(config=TradingConfig(arbitrage_crypto_first=False))

    signals = await strategy.scan(_crypto_first_events())

    group_order = list(dict.fromkeys(s.group_id for s in signals))
    assert group_order == [
        "event_sum:gen_a", "event_sum:crypto_a",
        "event_sum:gen_b", "event_sum:crypto_b",
    ]


@pytest.mark.asyncio
async def test_crypto_match_checks_slug_too():
    strategy = _strategy(config=TradingConfig(arbitrage_crypto_first=True))
    general = _event(event_id="gen", title="Some general event",
                      markets=[_market("g1", 0.30), _market("g2", 0.30)])
    crypto_by_slug = {
        "id": "crypto_slug", "title": "Ambiguous title",
        "slug": "solana-price-eoy", "markets": [_market("s1", 0.30), _market("s2", 0.30)],
    }

    signals = await strategy.scan([general, crypto_by_slug])

    group_order = list(dict.fromkeys(s.group_id for s in signals))
    assert group_order == ["event_sum:crypto_slug", "event_sum:gen"]
