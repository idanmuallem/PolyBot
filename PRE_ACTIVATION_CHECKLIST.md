# Pre-activation checklist — before reactivating EC2 and letting the dry-run run

Context: the EC2 instance is currently inactive. Before bringing it back up
and letting the crypto dry-run accumulate data, work through the BLOCKERS.
The whole point of this run is to gather the un-confounded resolved-outcome
data that Phase 1/3 couldn't get (the historical backtest was 94% NO), so
the bar is: don't activate until we're sure the run will actually produce
that data, trade at all, and price in the right direction.

Grouped by whether it gates activation.

## A. BLOCKERS — the run is wasted or unsafe without these

### A1. DONE (analysis) — and the original worry was INVERTED
Original worry: `entry_k=0.5` shrinks edges below `min_ev` -> zero trades.
Running the numbers (scripts/trade_rate_check.py, A1_TRADE_RATE_FINDINGS.md)
showed the OPPOSITE: `logit_shrink` pulls the brain toward 0.5, which on
cheap markets (price ≲0.12) lifts the 40% model component ABOVE the price and
manufactures a tradable YES edge even when the brain AGREES with the market
(e.g. market=0.10, brain=0.10 -> ev_yes=+0.60). So:
- `min_ev` is NOT too high; do NOT lower it.
- Expect the bot to readily take cheap-market YES bets with little brain
  conviction. Cheap longshots historically resolve NO ~94% of the time, so
  this could be systematically bad — read it as the known behavior of this
  setting, not a bug, and let forward_calibration.py measure whether these
  win. May argue for a LOWER entry_k later; don't change it blind now.
- Still worth running `scripts/trade_rate_check.py --csv <live export>` on a
  current live-market dump from the instance for the real trade-rate number
  (the historical CSV is far-OTM and unrepresentative).

### A2. Confirm forward-calibration data is actually captured and joinable
The resolution loop (`resolve_closed_markets`, every 15 min in dry-run) does
resolve entered positions, and the TRACK log records entry-time `pre_prob`,
`post_prob`, `wang_lambda` (= entry_k now), `wang_fair_value`, `model_used`.
Before activating, confirm:
- the entry snapshot (pre_prob, post_prob, market_price, entry_k, model_used,
  strike, spot, tte) is PERSISTED per trade in the DB, not only written to a
  rotating log that ages out.
- resolution outcomes are persisted and joinable back to those entries by
  market/token id.
- **Known limitation to accept up front:** only ENTERED positions resolve, so
  forward data is (a) low-volume and (b) selection-biased toward markets that
  cleared min_ev. If you want calibration across all EVALUATED markets (like
  Phase 1 had), you must log evaluated-but-not-entered snapshots too and
  resolve those markets later — and you CANNOT recover un-logged snapshots
  after the fact. Decide now whether to add that logging.

### A3. EC2 `.env`: rename `WANG_LAMBDA` -> `ENTRY_K`
After the redesign, `WANG_LAMBDA` in the deployed `.env` is silently ignored;
the bot falls back to `DEFAULT_ENTRY_K=0.5`. That happens to be the intended
value, so it's not dangerous — but it's undocumented config drift and will
confuse the next person who reads the `.env`. Rename it, and add `ENTRY_K` to
any `.env` template.

### A4. Confirm the deploy pipeline completes AND the engine actually starts
The last deploy hung because the instance was off. When reactivated, confirm
the full chain (test -> build -> ECR push -> SSM deploy) goes green AND the
engine process is actually RUNNING afterward — not just that CI passed. This
is the "silent 18-minute blackout" lesson: verify execution (engine heartbeat
in the DB / dashboard), not just that SSM queued the command.

## B. CORRECTNESS / ROBUSTNESS — latent bugs worth clearing

### B1. The "dip" inversion bug (brains/crypto.py:171)
`invert_keywords = ["↓", "below", "under", "less", "down", "lower"]` — "dip"
is NOT in the list. A market phrased "dip to $X" would be priced in the WRONG
DIRECTION (the brain answers the opposite question). Live markets currently
use arrow/↓ phrasing so may not hit it right now — but before activating,
either confirm no current live crypto market title uses "dip"-style phrasing,
or add "dip" to the list. A silent wrong-direction probability is a serious
mispricing, not a cosmetic bug.

### B2. Resolution-type router audit (early in the run)
`classify_resolution_type` falls back to "unknown" -> terminal Black-Scholes
for anything it can't classify. `model_used` is logged per trade (good). Early
in the dry-run, spot-check the `model_used` distribution: confirm live markets
classify as `first_passage` as expected and nothing is silently falling to
`unknown`/terminal (which would mean mispricing touch markets with the old
wrong formula).

### B3. Dry-run parity re-confirmation
Parity is non-negotiable (past gaps: cross-thread sqlite crashes, phantom
sells, incomplete baskets, restart double-buy). This session's changes were
pricing-only (entry_k) + a dead constant + docs, so parity shouldn't be
affected — but run the e2e/parity tests against the deployed image, and
confirm the restart-double-buy guard still holds on the real instance.

### B4. Instance sanity on reactivation
Confirm t3.small (a prior resize to t3.small once didn't persist and caused
OOM kills on t3.micro), 1 GB swapfile present, ECR lifecycle (keep-5) intact.

## C. EXPLORATORY / DECISIONS — not blockers, decide don't necessarily do

### C1. Set a concrete review trigger for entry_k
Leave `entry_k=0.5` for this run by design. Define the trigger to revisit it,
e.g. "after N resolved trades, re-run the divergence-vs-correctness split from
PHASE3_VALIDATION.md on forward data." Without a trigger it just sits at the
placeholder indefinitely.

### E. Flat-stake SIZING_MODE via check_and_cap_bet (with the clamp)
Optional A/B against Kelly, mirroring `PRICING_MODE=wang|legacy`: a
`SIZING_MODE=kelly|flat` where `flat` routes through `check_and_cap_bet`.
Thematically aligned with the redesign (distrust an unvalidated edge ->
don't let it drive bet size). **If wired in, `check_and_cap_bet` MUST also
clamp to `max_bet_size_usd`** — it currently skips that ceiling, so
`bankroll * fraction` could exceed the intended per-bet cap, bounded only by
the daily budget. See DEAD_CODE_REVIEW.md.

### E2. Triage the other tested-but-uncalled functions
`get_total_value`, `clear_cache`, `render_correlation_matrix` — cosmetic,
do anytime.
