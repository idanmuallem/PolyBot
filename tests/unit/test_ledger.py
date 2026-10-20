"""Ledger: the typed record of trades and positions (spec #12, ticket #13).

Exercised only through its public interface, against real SQLite in a temp
directory — no fakes."""
import pytest

from trading.ledger import Ledger


@pytest.fixture
def ledger(tmp_path):
    return Ledger(str(tmp_path / "trades.db"))


def _enter(ledger, token_id="tok_yes", side="YES", shares=10.0, price=0.40, **kw):
    fields = dict(
        token_id=token_id, condition_id="cond1", side=side, shares=shares, price=price,
        venue="paper", strategy="crypto", asset_type="Crypto::BTCUSDT",
        market_name="Will BTC exceed $100k?", pre_prob=0.55, post_prob=0.50, wang_edge=0.10,
    )
    fields.update(kw)
    return ledger.record_entry(**fields)


def test_recorded_entry_is_an_open_holding_with_its_entry_details(ledger):
    _enter(ledger)

    holding = ledger.holding("tok_yes")

    assert holding is not None
    assert holding.condition_id == "cond1"
    assert holding.side == "YES"
    assert holding.shares == pytest.approx(10.0)
    assert holding.entry_price == pytest.approx(0.40)
    assert holding.pre_prob == pytest.approx(0.55)
    assert holding.post_prob == pytest.approx(0.50)
    assert holding.wang_edge == pytest.approx(0.10)
    assert holding.asset_type == "Crypto::BTCUSDT"
    assert holding.venue == "paper"
    assert holding.strategy == "crypto"


def test_unknown_token_has_no_holding(ledger):
    assert ledger.holding("nope") is None


def test_no_side_holding_is_found_by_condition_and_side(ledger):
    _enter(ledger, token_id="tok_no", side="NO", price=0.30)

    assert ledger.holding_for("cond1", "NO").token_id == "tok_no"
    assert ledger.holding_for("cond1", "no").token_id == "tok_no"
    assert ledger.holding_for("cond1", "YES") is None


def test_open_positions_lists_every_open_holding(ledger):
    _enter(ledger, token_id="a", condition_id="c1")
    _enter(ledger, token_id="b", condition_id="c2", side="NO")

    assert sorted(h.token_id for h in ledger.open_positions()) == ["a", "b"]


def test_full_sale_closes_the_holding_and_realizes_pnl(ledger):
    _enter(ledger, shares=10.0, price=0.40)

    ledger.record_exit(token_id="tok_yes", shares=10.0, price=0.65, reason="TAKE-PROFIT", venue="paper")

    assert ledger.holding("tok_yes") is None
    assert ledger.realized_pnl() == pytest.approx(2.50)  # (0.65 - 0.40) * 10


def test_partial_sale_leaves_the_rest_open(ledger):
    _enter(ledger, shares=10.0, price=0.40)

    ledger.record_exit(token_id="tok_yes", shares=4.0, price=0.30, reason="STOP-LOSS", venue="paper")

    assert ledger.holding("tok_yes").shares == pytest.approx(6.0)
    assert ledger.realized_pnl() == pytest.approx(-0.40)  # (0.30 - 0.40) * 4


def test_sale_of_a_position_the_ledger_never_saw_uses_the_given_entry_price(ledger):
    ledger.record_exit(token_id="legacy", shares=5.0, price=0.60, reason="EV-CONVERGENCE",
                       venue="live", entry_price=0.50, side="YES")

    assert ledger.realized_pnl() == pytest.approx(0.50)
    assert ledger.closed_trade_deltas() == pytest.approx([0.10])


def test_resolution_closes_the_holding_at_the_payout_price(ledger):
    _enter(ledger, shares=10.0, price=0.40)

    ledger.record_resolution(token_id="tok_yes", shares=10.0, payout_price=1.0, venue="paper")

    assert ledger.holding("tok_yes") is None
    assert ledger.realized_pnl() == pytest.approx(6.0)
    assert ledger.closed_trade_deltas() == pytest.approx([0.60])


def test_unconfirmed_expiry_closes_the_holding_without_realizing_pnl(ledger):
    _enter(ledger, shares=10.0, price=0.40)

    ledger.record_resolution(token_id="tok_yes", shares=10.0, payout_price=None, venue="live")

    assert ledger.holding("tok_yes") is None
    assert ledger.realized_pnl() == 0.0
    assert ledger.closed_trade_deltas() == []


def test_unrealized_pnl_marks_open_holdings_to_the_given_prices(ledger):
    _enter(ledger, token_id="a", shares=10.0, price=0.40)
    _enter(ledger, token_id="b", condition_id="c2", side="NO", shares=5.0, price=0.20)

    pnl = ledger.unrealized_pnl({"a": 0.50, "b": 0.10})

    assert pnl == pytest.approx(10 * 0.10 + 5 * -0.10)


def test_unrealized_pnl_ignores_holdings_without_a_mark(ledger):
    _enter(ledger, token_id="a", shares=10.0, price=0.40)

    assert ledger.unrealized_pnl({}) == 0.0


# ── trade stats ──────────────────────────────────────────────────────────────

def test_trade_stats_on_an_empty_ledger(ledger):
    stats = ledger.trade_stats()

    assert stats == {
        "win_rate": 0.0, "total_trades": 0, "avg_win": 0.0, "avg_loss": 0.0,
        "total_yes_trades": 0, "yes_win_rate": 0.0, "total_no_trades": 0, "no_win_rate": 0.0,
    }


def test_trade_stats_count_entries_by_side(ledger):
    _enter(ledger, token_id="y", side="YES")
    _enter(ledger, token_id="n", condition_id="c2", side="NO")

    stats = ledger.trade_stats()

    assert stats["total_trades"] == 2
    assert stats["total_yes_trades"] == 1
    assert stats["total_no_trades"] == 1


def test_signal_exits_are_won_or_lost_by_price_move(ledger):
    """EV-CONVERGENCE / WANG-EDGE-DECAY aren't P&L-threshold exits, so the
    win/loss comes from the realized price move on the held side."""
    _enter(ledger, token_id="n", side="NO", shares=58.14, price=0.43)
    _enter(ledger, token_id="y", condition_id="c2", side="YES", shares=10.0, price=0.50)
    ledger.record_exit(token_id="n", shares=58.14, price=0.38, reason="EV-CONVERGENCE", venue="paper")
    ledger.record_exit(token_id="y", shares=10.0, price=0.55, reason="WANG-EDGE-DECAY", venue="paper")

    stats = ledger.trade_stats()

    assert stats["total_trades"] == 4  # two entries + two closes
    assert stats["win_rate"] == pytest.approx(50.0)
    assert stats["avg_win"] == pytest.approx(5.50)    # 0.55 * 10 proceeds
    assert stats["avg_loss"] == pytest.approx(22.09)  # 0.38 * 58.14 proceeds
    assert stats["yes_win_rate"] == pytest.approx(100.0)
    assert stats["no_win_rate"] == 0.0


def test_take_profit_is_a_win_and_stop_loss_a_loss_by_definition(ledger):
    _enter(ledger, token_id="a", price=0.50)
    _enter(ledger, token_id="b", condition_id="c2", price=0.50)
    ledger.record_exit(token_id="a", shares=10.0, price=0.50, reason="TAKE-PROFIT", venue="paper")
    ledger.record_exit(token_id="b", shares=10.0, price=0.50, reason="STOP-LOSS", venue="paper")

    stats = ledger.trade_stats()

    assert stats["win_rate"] == pytest.approx(50.0)


def test_paid_out_resolution_counts_but_unconfirmed_expiry_does_not(ledger):
    _enter(ledger, token_id="won", price=0.40)
    _enter(ledger, token_id="dead", condition_id="c2", price=0.30)
    ledger.record_resolution(token_id="won", shares=10.0, payout_price=1.0, venue="paper")
    ledger.record_resolution(token_id="dead", shares=10.0, payout_price=None, venue="live")

    stats = ledger.trade_stats()

    assert stats["total_trades"] == 3  # two entries + one paid-out resolution
    assert stats["win_rate"] == pytest.approx(100.0)
    assert stats["avg_win"] == pytest.approx(10.0)


def test_trade_stats_can_be_limited_to_one_venue(ledger):
    _enter(ledger, token_id="p", venue="paper")
    _enter(ledger, token_id="l", condition_id="c2", venue="live")

    assert ledger.trade_stats(venue="live")["total_trades"] == 1
    assert ledger.trade_stats()["total_trades"] == 2


# ── transactions ─────────────────────────────────────────────────────────────

def test_transactions_list_buys_and_sales_newest_first(ledger):
    _enter(ledger, shares=10.0, price=0.40)
    ledger.record_exit(token_id="tok_yes", shares=10.0, price=0.65, reason="TAKE-PROFIT", venue="paper")

    sale, buy = ledger.transactions()

    assert buy["kind"] == "BUY"
    assert buy["amount_usd"] == pytest.approx(4.0)
    assert buy["pnl_per_share"] is None
    assert sale["kind"] == "SELL"
    assert sale["reason"] == "TAKE-PROFIT"
    assert sale["side"] == "YES"
    assert sale["market_name"] == "Will BTC exceed $100k?"
    assert sale["pnl_per_share"] == pytest.approx(0.25)
    assert sale["confirmed"] is True


def test_transactions_keep_an_unconfirmed_expiry(ledger):
    _enter(ledger, shares=5.0, price=0.30)
    ledger.record_resolution(token_id="tok_yes", shares=5.0, payout_price=None, venue="live")

    expiry = ledger.transactions()[0]

    assert expiry["kind"] == "EXPIRED"
    assert expiry["confirmed"] is False
    assert expiry["pnl_per_share"] is None


def test_transactions_respect_the_limit(ledger):
    for i in range(3):
        _enter(ledger, token_id=f"t{i}", condition_id=f"c{i}")

    assert [t["token_id"] for t in ledger.transactions(limit=2)] == ["t2", "t1"]
