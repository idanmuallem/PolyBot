# Wang Transform Redesign — Progress

## Iteration 1 — Entry-side redesign (exit side untouched, per explicit scope decision)

### What changed

Replaced the entry-side Wang Transform (`brains/base.py`'s call into
`wang_transform()`, a flat probit-space shift with `wang_lambda=-0.75`) with
`logit_shrink()` (`brains/pricing_engine.py`), a proportional shrink toward
0.5 in log-odds space: `p' = sigmoid(logit(p) * k)`, `k` in `[0, 1]`.

**The bug this fixes:** `wang_transform`'s constant probit-space shift is
only "toward 0.5" on one side of a lambda-dependent crossover point. For
`lambda=-0.75`, that crossover sits at p≈0.62-0.65 (verified numerically,
see `test_wang_transform_crossover_bug_at_entry_lambda` in
`test_pricing_engine.py`). Every raw probability the crypto brain has ever
actually produced sits in 0.25-0.45 — entirely on the wrong side — so in
production this transform moved every real input *away* from 0.5, the
opposite of its own docstring's claim. `logit_shrink()` is monotonic toward
0.5 for every `p` in `(0,1)` and every `k` in `[0,1]`, by construction —
proven by a dense-grid property test
(`test_logit_shrink_monotonic_toward_half_full_sweep`), not just checked
against the historical output range.

**Scope, explicitly:** only entry-side pricing (`BaseBrain.evaluate()`,
`config.entry_k`) was touched. The exit side (`trading/risk_manager.py` →
`PricingEngine.compute_edge()`) is a hierarchical model fit on 13,274 real
contracts (volume/duration/extremity-dependent lambda, oracle3 Yang 2026),
not a flat guess like entry's old -0.75 — replacing it with the same flat
shrink would have discarded that fit. Left entirely alone. `wang_transform()`
itself is unchanged and still used there.

### entry_k default: 0.5 — a documented neutral placeholder, NOT a fitted calibration

Phase 1 (`scripts/backtest_crypto_brain.py`, `scripts/phase1_analysis.py`,
`PHASE1_FINDINGS.md` — 56 resolved BTC/ETH markets, 164 snapshots) tested
whether the brain's divergence from the market predicts which one is closer
to the real outcome:
- Divergence **magnitude** does not predict correctness (r=-0.103, p=0.189
  — not significant).
- Divergence **direction** shows a correlation (r=0.375, p<0.0001) but is
  confounded: the sample is 94% NO outcomes (far-OTM strikes), so a bearish
  correction wins near-mechanically regardless of real skill.

No evidence basis exists for a stronger or weaker `k`. At the same time,
the fixed brain's **aggregate** calibration (Brier ≈0.047) is on par with
or slightly better than the market's own (≈0.057) — independently
reproduced against the committed dataset before this work started (see
commit history), not just taken on faith from the earlier chat summary —
which argues against defaulting to near-total distrust (`k` close to 0)
either.

`entry_k=0.5` treats the brain's raw signal and 0.5 symmetrically in
log-odds space, pending more resolved dry-run trades to calibrate this
properly with less confounded data than the historical backtest's
94%-NO-outcome sample. **Revisit once enough dry-run trades resolve** to
re-run the same divergence-vs-correctness analysis on forward data.

### Naming decisions (why some things kept the "wang_" name)

- **Renamed** (config-facing, low blast radius): `DEFAULT_WANG_LAMBDA` →
  `DEFAULT_ENTRY_K`, `TradingConfig.wang_lambda` → `.entry_k`, env var
  `WANG_LAMBDA` → `ENTRY_K`. **Deployment note:** the EC2 `.env` almost
  certainly still sets `WANG_LAMBDA` — after this merges, that line has no
  effect (silently falls back to the new default) until it's renamed to
  `ENTRY_K` in the deployed `.env`. Bot is in dry-run mode, so no live-capital
  risk from this, but worth fixing before the next deploy.
- **Kept** `TradeSignal.wang_lambda`, `TradeCandidate.wang_lambda`, and the
  TRACK-log/dashboard dict key `"wang_lambda"` (storage/dashboard/DB-schema
  continuity — renaming these touches decision_pipeline.py's logging,
  the dashboard, and historical SQLite rows for no functional benefit).
  Each site is commented explaining it now holds `entry_k`, not a probit
  Wang lambda. Dashboard *display* label changed from "Wang λ" → "Entry k"
  (`ui/data_manager.py`, `ui/dashboard.py`) since that's a free, zero-risk
  fix for the same misleading-name problem this whole task exists to solve.
- **Critical semantic flip caught before landing:** the old escape hatch
  (`decision_pipeline.py`'s "legacy" pricing mode) passed `wang_lambda=0.0`
  meaning "disable the shift, passthrough." For `logit_shrink`, `k=0.0`
  means "collapse everything to 0.5" — the *opposite* of passthrough.
  Passthrough for the new function is `k=1.0`. Fixed at the call site with
  an explicit comment; would otherwise have silently broken the
  `PRICING_MODE=legacy` A/B comparison path.

### Docs updated (were already stale before this change)

- `README.md`: Wang Transform section, escape-hatch section, `.env` example
  block, and the config table — all previously described `WANG_LAMBDA`/the
  old crossover-broken behavior.
- Also fixed, while touching the same section: the README's crypto-model
  description still said "Black-Scholes for every TTE," which predates the
  first-passage/resolution-router fix (`brains/crypto.py`,
  `classify_resolution_type`) — that fix's own commit never updated the
  README. Updated to describe the actual current routing.

### Tests

- `tests/unit/test_pricing_engine.py`: added the crossover-bug
  characterization test (proves the OLD bug numerically, doesn't just
  assert against production code) and the `logit_shrink` property-test
  suite (monotonicity, fixed point at 0.5, `k=1`/`k=0` boundary behavior,
  order-preservation, direct contrast against the old bug's failure cases)
  — dense sweep over `p` (including near-0/near-1) × `k` in `[0,1]`.
- `tests/unit/test_brains_base.py`: rewrote the Wang-transform test section
  for the new mechanism. Two properties from the old tests do **not**
  reproduce for logit_shrink and were replaced rather than forced to pass:
  "shrink is largest near p=0.5" (true of the old additive probit shift;
  logit_shrink has a *fixed point* at 0.5 — zero shift there, growing
  toward the extremes) and any assumption that `wang_lambda=0.0` disables
  the layer (flipped to `k=1.0`, see above). Added
  `test_shrink_reduces_low_probability_toward_half` as the actual
  end-to-end bug regression test, using 0.30 — inside the brain's real
  historical output range — and contrasting directly against the old
  transform's behavior on the same input.
- `tests/integration/test_evaluate_to_execute.py`,
  `tests/unit/test_data_manager.py`: updated call sites and display-column
  assertions for the rename; no behavioral rewrite needed.
- `tests/e2e/test_full_pipeline_dryrun.py`: the Wang-mode e2e test used a
  real (unmocked) brain and asserted a fill against `min_ev=0.50`, tuned
  implicitly to the old lambda's ~0.71 edge on that scenario. Under
  `entry_k=0.5` the same scenario's real edge is ~0.33 — still genuinely
  favorable, just a different number, because the calibration actually
  changed. Relaxed `min_ev` to 0.10 for this test (documented as testing
  pipeline plumbing, not pinning a calibration value — same rationale the
  test already used for `wang_min_edge=0.0`), rather than tuning the
  scenario to reproduce the old number artificially.

### Verification

`grep -rn "DEFAULT_WANG_LAMBDA\|wang_lambda="` across the repo: only
historical/comment references and the deliberately-kept storage field
names remain (see above), no live behavioral reference to the old flat
lambda in the entry path.

Full suite: **428 passed** (418 baseline + 10 new), 0 failed.

### Out of scope, confirmed untouched

`min_ev` (except the one test's threshold, not the production default),
Kelly/position sizing, the crypto brain pricing formula, the resolution-type
router, `model_weight` (still 0.40), and all exit-side code
(`trading/risk_manager.py`, `PricingEngine`'s hierarchical model).

### Next: Phase 3

Backtest this mechanism against the same 56-market/164-snapshot dataset,
check for reintroduced directional bias, decide (a human decision, not an
automated exit condition) whether it's ready for dry-run.
