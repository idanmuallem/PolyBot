# Dead-code review — 2026-09-28

Scope: a responsible dead-code pass, not an aggressive one. In a trading bot,
most of what a static tool (vulture) flags is either load-bearing via dynamic
dispatch / dataclass-field reads, or intentionally dormant. Blind deletion
would destroy features. This documents what was removed, what was deliberately
kept, and what needs a human decision.

## Removed (genuinely dead, zero references anywhere)

- `core/wallet_context.py`: `DATA_ROOT = "data"` — module constant, referenced
  nowhere (not even within its own file). Removed.

## Fixed (misleading leftover from the Wang entry-side redesign)

- `brains/pricing_engine.py`: corrected the docstrings that described
  `wang_transform()` as "exit-side only." The exit-side `PricingEngine`
  inlines its own probit shift and does not call that function. It is now
  accurately documented as reference/test-only (used by the characterization
  tests and `scripts/phase3_validation.py` to document the old crossover bug).
  The function is retained on purpose — those tests depend on it.

## Deliberately NOT removed — false positives (do not delete in a future pass)

vulture flags these as unused; they are not:

- `DataBridge` attributes (`market_actual`, `market_poly`, `forecast`,
  `last_update`, `current_drawdown_pct`, `open_positions_value`, etc.): written
  in `decision_pipeline.py` / `risk_manager.py` and read by the Streamlit
  dashboard. The tool can't trace the cross-process read.
- `row_factory` assignments (`ui/data_manager.py`): real side-effecting sqlite
  config, not a dead attribute.
- `EconomyHunter`, `WeatherHunter` (and their `get_anchor_value`): dormant
  strategy classes — only the crypto strategy is active right now, but these
  are intentional, registered hunters, not dead code.
- `BaseApiClient.get_latest_value(self, *args, **kwargs)`: `*args` flagged as
  unused; it's an `@abstractmethod` interface signature.

## NEEDS A HUMAN DECISION — tested but never called in production

These have test coverage (so they're validated API surface) but no production
call site. That is a different problem from dead code — either they're
obsolete and their tests should go with them, or they were meant to be wired
in and aren't. I did not touch them. Triage each:

- **`trading/budget_manager.py::check_and_cap_bet` (6 tests, 0 callers)** —
  the most important one. It's a budget-cap safety method: caps a bet to the
  remaining strategy budget. The pipeline currently sizes bets via
  `compute_kelly_bet_size` instead. A tested budget cap that's never called is
  either dead (superseded by Kelly + `max_bet_size_usd` clamp) or a safety net
  that should be wired in. **Worth a deliberate decision before dry-run
  conclusions, given it's about not over-betting a budget.**
- `trading/paper_adapter.py::get_total_value` (2 tests, 0 callers)
- `hunters/clients/ccxt_client.py::clear_cache` (3 tests, 0 callers)
- `ui/components.py::render_correlation_matrix` (4 tests, 0 callers) — a
  dashboard panel that's built and tested but not mounted in the current
  dashboard layout.

Recommendation: leave all four in place until you decide per-item. Deleting a
tested safety/monitoring method to satisfy a linter is the wrong trade in a
system that moves money.
