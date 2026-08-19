"""
Trade execution engine: order submission, position tracking, and balance fetching.

Wraps the Polymarket CLOB client. Execution is gated by EV threshold, daily
trade count, and price bounds. Supports dry-run, paper-trade, and live modes.
"""

import asyncio
import logging
import math
import time
from typing import Optional, Dict, Any, Callable, List

import requests
from eth_utils import to_checksum_address

from core.models import MarketData, Position, PRICE_FLOOR, PRICE_CEILING
from core.trading_config import TradingConfig

try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import OrderArgs, AssetType, BalanceAllowanceParams
    from py_clob_client.order_builder.constants import BUY, SELL
    CLOB_IMPORT_OK = True
except Exception as e:
    logging.error(f"Import Error: {e}")
    ClobClient = Any
    OrderArgs = Any
    AssetType = Any
    BalanceAllowanceParams = Any
    BUY = "BUY"
    SELL = "SELL"
    CLOB_IMPORT_OK = False


class TradeExecutor:
    """Handles trade execution with risk management.

    Modes: dry_run (simulate only), paper_trade (log only), live (real orders).
    """

    def __init__(self, config: TradingConfig):
        self.config = config
        self.trade_count_today = 0
        self.trades_by_strategy: Dict[str, int] = {}
        self.dry_run = self.config.dry_run
        self.paper_trade_mode = self.config.paper_trade_mode
        self.proxy_address = self.config.proxy_address
        self.client = None

        if not CLOB_IMPORT_OK:
            print("[FATAL] py-clob-client is not loaded correctly!")
            return

        if self.config.proxy_address and self.config.private_key:
            try:
                print(f"Signing for Proxy: {self.config.proxy_address}")
                self.client = ClobClient(
                    host="https://clob.polymarket.com",
                    chain_id=137,
                    key=self.config.private_key,
                    funder=to_checksum_address(self.config.proxy_address),
                    signature_type=self.config.signature_type,
                )
                try:
                    creds = self.client.create_or_derive_api_creds()
                    self.client.set_api_creds(creds)
                    print("[AUTH] Credentials successfully derived and set!")
                except Exception as e:
                    print(f"[AUTH-ERROR] Failed to derive: {e}")
                    self.client = None
                    return
                print("[SUCCESS] Live CLOB Client is fully armed and operational!")
            except Exception as e:
                print(f"[FATAL ERROR] ClobClient failed to build: {e}")
                self.client = None
        else:
            print("[FATAL] Missing keys in Config! Cannot build Client.")

    def _resolve_positions_user_address(self) -> Optional[str]:
        explicit = str(self.proxy_address or "").strip()
        return explicit if explicit else None

    @staticmethod
    def _pick_float(payload: Dict[str, Any], *keys: str) -> float:
        for key in keys:
            raw = payload.get(key)
            if raw is None:
                continue
            try:
                return float(raw)
            except Exception:
                continue
        return 0.0

    def _submit_order(self, token_id: str, price: float, side: str, size: float):
        """Submit a live order to the Polymarket CLOB."""
        try:
            price = round(float(price), 2)
            size = round(float(size), 2)
            print(f"[EXECUTION] Attempting {side} order: {size} shares at ${price}")
            order = OrderArgs(token_id=token_id, price=price, side=side, size=size)
            return self.client.create_and_post_order(order)
        except Exception as e:
            print(f"[LIVE-TRADE-ERROR] {token_id} - {str(e)}")
            return None

    @staticmethod
    def _is_valid_order_response(response: Any) -> bool:
        if not response:
            return False
        if isinstance(response, dict):
            if response.get("error") or response.get("errors") or response.get("errorMsg"):
                return False
            order_id = (
                response.get("orderID")
                or response.get("orderId")
                or response.get("order_id")
                or response.get("id")
            )
            return bool(order_id)
        return True

    @staticmethod
    def _format_order_exception(exc: Exception) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "error": str(exc),
            "exception_type": type(exc).__name__,
            "exception_repr": repr(exc),
        }
        response = getattr(exc, "response", None)
        if response is not None:
            if getattr(response, "status_code", None) is not None:
                payload["status_code"] = response.status_code
            if getattr(response, "text", None):
                payload["response_text"] = response.text
        return payload

    def get_balance(self) -> float:
        """Fetch available CLOB collateral balance (deployable cash)."""
        if self.dry_run or self.client is None:
            return float(self.config.paper_balance_usd)

        try:
            if hasattr(self.client, "get_collateral_balance"):
                raw = float(self.client.get_collateral_balance())
                return raw / 1_000_000.0 if raw > 1_000_000 else raw

            if hasattr(self.client, "get_balance_allowance"):
                params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
                resp = self.client.get_balance_allowance(params=params)
                if isinstance(resp, dict):
                    balance_raw = resp.get("balance", resp)
                    if isinstance(balance_raw, dict):
                        for key in ("balance", "amount", "available", "usdc", "USDC"):
                            if balance_raw.get(key) is not None:
                                raw = float(balance_raw[key])
                                return raw / 1_000_000.0 if raw > 1_000_000 else raw
                    raw = float(balance_raw)
                    return raw / 1_000_000.0 if raw > 1_000_000 else raw
        except Exception as exc:
            logging.warning(f"Live collateral balance fetch failed: {exc}")

        return 0.0

    def get_open_positions(self) -> List[Position]:
        """Fetch open positions from Polymarket Data API with mark-to-mid PnL.

        Uses Data API because py-clob-client does not expose a stable positions
        listing method across versions.
        """
        wallet_address = self._resolve_positions_user_address()
        if not wallet_address:
            return []

        positions: List[Position] = []
        url = f"https://data-api.polymarket.com/positions?user={wallet_address}"

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            if status_code != 404:
                logging.debug(f"Gamma positions HTTP error: {exc}")
            return []
        except requests.exceptions.RequestException as exc:
            logging.debug(f"Gamma positions request error: {exc}")
            return []

        try:
            raw_positions = response.json()
            if isinstance(raw_positions, dict):
                raw_positions = raw_positions.get("positions", [])
            if not isinstance(raw_positions, list):
                return []

            for raw in raw_positions:
                if not isinstance(raw, dict):
                    continue

                token_id = str(
                    raw.get("asset") or raw.get("token_id") or raw.get("tokenId") or ""
                )
                if not token_id:
                    continue

                shares = abs(float(self._pick_float(
                    raw, "size", "shares", "quantity", "balance", "positionSize", "numShares"
                )))
                initial_price = self._pick_float(
                    raw, "avgPrice", "avg_price", "entry_price", "initial_price", "price"
                )
                current_price = self._pick_float(
                    raw, "currentPrice", "current_price", "markPrice", "mark_price"
                )
                current_value = abs(self._pick_float(
                    raw, "currentValue", "current_value", "positionValue",
                    "position_value", "value", "usdValue",
                ))

                if current_value <= 0.0 and shares > 0.0 and current_price > 0.0:
                    current_value = shares * current_price
                if current_price <= 0.0 and shares > 0.0 and current_value > 0.0:
                    current_price = current_value / shares
                if shares <= 0.0 and current_value > 0.0 and initial_price > 0.0:
                    shares = current_value / initial_price
                if shares <= 0.0 and current_value <= 0.0:
                    continue

                value = current_value if current_value > 0.0 else (
                    shares * (current_price if current_price > 0.0 else initial_price)
                )
                if current_price <= 0.0:
                    current_price = initial_price

                pnl_ratio = (
                    (current_price - initial_price) / initial_price
                    if initial_price > 0 else 0.0
                )

                positions.append(Position(
                    market_id=str(
                        raw.get("conditionId") or raw.get("condition_id")
                        or raw.get("market_id") or token_id
                    ),
                    token_id=token_id,
                    initial_price=initial_price,
                    current_price=current_price,
                    shares=shares,
                    value=value,
                    pnl_ratio=pnl_ratio,
                    side=str(raw.get("outcome") or raw.get("side") or "UNKNOWN"),
                    live_ev=float(pnl_ratio),
                ))
        except Exception as exc:
            logging.warning(f"Could not parse open positions: {exc}")

        return positions

    def execute_trade(
        self,
        token_id: str,
        current_poly_price: float,
        shares: float,
        bet_amount: float,
        asset_type: str,
        side: str,
        no_token_id: Optional[str],
        log_func: Callable,
    ) -> bool:
        """Execute a live order or simulate it based on configuration."""
        execution_side = str(side or "YES").upper()
        execution_token_id = str(token_id)
        execution_price = float(current_poly_price)

        if execution_side == "NO":
            if no_token_id:
                execution_token_id = str(no_token_id)
            execution_price = max(1e-6, 1.0 - float(current_poly_price))

        if self.dry_run:
            print(
                f"[DRY-RUN] Would buy {shares} of {execution_token_id} "
                f"({execution_side}) for ${bet_amount}."
            )
            log_func("DRY-RUN", asset_type, execution_token_id, {
                "price": execution_price, "shares": shares,
                "bet_amount_usd": bet_amount, "side": execution_side,
            })
            return True

        if self.client is None:
            log_func("PAPER-TRADE", asset_type, execution_token_id, {
                "price": execution_price, "shares": shares,
                "bet_amount_usd": bet_amount, "side": execution_side,
                "reason": "No live CLOB client configured",
            })
            return True

        try:
            order_resp = self._submit_order(
                token_id=execution_token_id,
                price=execution_price,
                side=BUY,
                size=shares,
            )
            if not self._is_valid_order_response(order_resp):
                error_payload = {
                    "error": "order_rejected_or_unconfirmed",
                    "response": order_resp, "side": execution_side,
                    "price": execution_price, "shares": shares,
                }
                logging.error(f"LIVE order rejected: {error_payload}")
                log_func("LIVE-TRADE-ERROR", asset_type, execution_token_id, error_payload)
                return False

            log_func("LIVE-TRADE", asset_type, execution_token_id, {
                "success": True, "response": order_resp,
                "side": execution_side, "price": execution_price,
            })
            return True
        except Exception as exc:
            error_payload = self._format_order_exception(exc)
            logging.error(f"LIVE order rejected: {error_payload}")
            log_func("LIVE-TRADE-ERROR", asset_type, execution_token_id, error_payload)
            return False

    def sell_position(self, token_id: str, shares: float, price: float, log_func: Callable) -> bool:
        """Sell an existing position (dry-run, paper, or live)."""
        if self.dry_run:
            msg = f"[DRY-RUN] Would SELL {shares} of {token_id} at ${price}"
            print(msg)
            log_func("DRY-RUN-SELL", "Portfolio", token_id, {
                "message": msg, "price": price, "shares": shares,
            })
            return True

        if self.client is None:
            log_func("PAPER-SELL", "Portfolio", token_id, {
                "price": price, "shares": shares,
                "reason": "No live CLOB client configured",
            })
            return True

        try:
            order_resp = self._submit_order(token_id=token_id, price=price, side=SELL, size=shares)
            if not self._is_valid_order_response(order_resp):
                error_payload = {
                    "error": "order_rejected_or_unconfirmed",
                    "response": order_resp, "price": price, "shares": shares,
                }
                logging.error(f"SELL order rejected: {error_payload}")
                log_func("SELL-ERROR", "Portfolio", token_id, error_payload)
                return False

            log_func("SELL", "Portfolio", token_id, {
                "success": True, "response": order_resp, "price": price, "shares": shares,
            })
            return True
        except Exception as exc:
            error_payload = self._format_order_exception(exc)
            logging.error(f"SELL order rejected: {error_payload}")
            log_func("SELL-ERROR", "Portfolio", token_id, error_payload)
            return False

    # ------------------------------------------------------------------
    # Arbitrage group execution — limit orders + fill timeout (Phase 3)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_order_id(order_resp: Any) -> Optional[str]:
        if not isinstance(order_resp, dict):
            return None
        for key in ("orderID", "orderId", "order_id", "id"):
            value = order_resp.get(key)
            if value:
                return str(value)
        return None

    def _get_order_filled_shares(self, order_id: str) -> float:
        """How many shares of a resting order have filled so far.

        py-clob-client's ClobClient.get_order(order_id) (confirmed present
        on the installed client — GET /order under the hood) is synchronous,
        not async: there's no native async order-status polling to hook
        into. This blocks the event loop for one REST round-trip per poll,
        which is acceptable here since execute_arbitrage_group's polling
        loop is the only thing running at that point in the pipeline (see
        SequentialTradingPipeline.run_forever — one sequential async loop,
        nothing else concurrent to starve). If that ever changes, wrap this
        call in asyncio.to_thread()/loop.run_in_executor() instead of
        calling it directly.

        Tries several field-name candidates (same defensive-pick pattern as
        _pick_float/get_open_positions) since the exact CLOB order-status
        response schema (size_matched vs. matched_size, etc.) isn't pinned
        by py-clob-client's thin wrapper — it just returns the raw REST
        JSON, whose exact field names aren't documented client-side.
        """
        try:
            order = self.client.get_order(order_id)
        except Exception as exc:
            logging.warning(f"get_order failed for {order_id}: {exc}")
            return 0.0
        if not isinstance(order, dict):
            return 0.0
        return self._pick_float(order, "size_matched", "matched_size", "sizeMatched", "filled_size", "filledSize")

    def _cancel_order(self, order_id: str, log_func: Callable, asset_type: str, group_id: Optional[str]) -> bool:
        """Cancel one resting order. ClobClient.cancel(order_id) (confirmed
        present on the installed client — DELETE /order under the hood) is
        also synchronous; same blocking-call trade-off as
        _get_order_filled_shares above."""
        try:
            resp = self.client.cancel(order_id)
            log_func("CANCEL", asset_type, order_id, {
                "reason": "partial_fill_no_arb",
                "group_id": group_id,
                "response": resp,
            })
            return True
        except Exception as exc:
            log_func("CANCEL-ERROR", asset_type, order_id, {
                "reason": "cancel_failed",
                "group_id": group_id,
                "error": str(exc),
            })
            return False

    def _record_strategy_trade(self, strategy_tag: str):
        self.trade_count_today += 1
        self.trades_by_strategy[strategy_tag] = self.trades_by_strategy.get(strategy_tag, 0) + 1

    def _simulate_full_fill_arbitrage_group(
        self, legs: List[Dict[str, Any]], log_func: Callable,
        asset_type: str, group_id: Optional[str], strategy_tag: str,
    ) -> Dict[str, Any]:
        """Dry-run/paper simulation: every leg fills completely, same as
        execute_trade()'s existing dry-run/no-client behavior — no real
        orders are placed, but the group structure is still logged clearly.
        """
        fills: Dict[str, float] = {}
        for leg in legs:
            token_id = str(leg["token_id"])
            shares = float(leg["shares"])
            price = float(leg["price"])
            self.execute_trade(
                token_id=token_id,
                current_poly_price=price,
                shares=shares,
                bet_amount=float(leg.get("bet_amount_usd", shares * price)),
                asset_type=str(leg.get("asset_type") or asset_type),
                side=str(leg.get("side", "YES")),
                no_token_id=leg.get("no_token_id"),
                log_func=log_func,
            )
            fills[token_id] = shares
            self._record_strategy_trade(strategy_tag)

        arb_sets = min(fills.values()) if fills else 0.0
        surplus = {tid: round(s - arb_sets, 4) for tid, s in fills.items() if s > arb_sets}

        log_func("ARBITRAGE-FILL", asset_type, group_id or "group", {
            "reason": "dry_run_full_fill_simulated",
            "group_id": group_id,
            "n_legs": len(legs),
            "fills": fills,
            "arb_sets": arb_sets,
            "surplus": surplus,
        })

        return {"success": True, "arb_sets": arb_sets, "fills": fills, "unfilled": [], "surplus": surplus}

    async def _execute_live_arbitrage_group(
        self, legs: List[Dict[str, Any]], timeout_seconds: float, log_func: Callable,
        strategy_tag: str, asset_type: str, group_id: Optional[str],
    ) -> Dict[str, Any]:
        order_ids: Dict[str, Optional[str]] = {}

        for leg in legs:
            token_id = str(leg["token_id"])
            order_resp = self._submit_order(
                token_id=token_id, price=float(leg["price"]), side=BUY, size=float(leg["shares"]),
            )
            order_id = self._extract_order_id(order_resp)
            order_ids[token_id] = order_id
            if order_id is None:
                log_func("EXECUTION-ERROR", asset_type, token_id, {
                    "reason": "order_placement_failed",
                    "group_id": group_id,
                    "response": order_resp,
                })

        fills: Dict[str, float] = {tid: 0.0 for tid in order_ids}
        poll_interval = max(0.01, min(1.0, timeout_seconds / 10.0)) if timeout_seconds > 0 else 0.0
        deadline = time.monotonic() + timeout_seconds

        while True:
            for token_id, order_id in order_ids.items():
                if order_id is not None:
                    fills[token_id] = self._get_order_filled_shares(order_id)
            if fills and all(filled > 0 for filled in fills.values()):
                break
            if time.monotonic() >= deadline:
                break
            await asyncio.sleep(poll_interval)

        unfilled = [tid for tid, filled in fills.items() if filled <= 0]

        if unfilled:
            for order_id in order_ids.values():
                if order_id is not None:
                    self._cancel_order(order_id, log_func, asset_type, group_id)

            for token_id in unfilled:
                log_func("REJECTED", asset_type, token_id, {
                    "reason": "insufficient_liquidity",
                    "group_id": group_id,
                    "timeout_seconds": timeout_seconds,
                })

            log_func("REJECTED", asset_type, group_id or "group", {
                "reason": "partial_fill_no_arb",
                "group_id": group_id,
                "unfilled": unfilled,
                "fills": fills,
            })

            # Shares that did fill before the group was voided are real,
            # unretractable trades (cancel() only stops further fills) —
            # they still count against this strategy's daily trade count.
            for token_id, filled in fills.items():
                if filled > 0:
                    self._record_strategy_trade(strategy_tag)

            return {"success": False, "arb_sets": 0, "fills": fills, "unfilled": unfilled, "surplus": {}}

        arb_sets = min(fills.values())
        surplus = {tid: round(filled - arb_sets, 4) for tid, filled in fills.items() if filled > arb_sets}

        for _token_id in fills:
            self._record_strategy_trade(strategy_tag)

        log_func("ARBITRAGE-FILL", asset_type, group_id or "group", {
            "reason": "all_legs_filled",
            "group_id": group_id,
            "fills": fills,
            "arb_sets": arb_sets,
            "surplus": surplus,
        })

        return {"success": True, "arb_sets": arb_sets, "fills": fills, "unfilled": [], "surplus": surplus}

    async def execute_arbitrage_group(
        self,
        legs: List[Dict[str, Any]],
        timeout_seconds: float,
        log_func: Callable,
        strategy_tag: str = "arbitrage",
    ) -> Dict[str, Any]:
        """Place limit orders for every leg of an arbitrage group, wait up
        to timeout_seconds for fills, then settle the group:

        - Every leg got >=1 fill: keep everything. The guaranteed
          arbitrage position size ("arb_sets") is the minimum fill count
          across all legs; surplus shares on the other legs are bonus
          directional exposure, not part of the guaranteed structure.
        - Any leg got 0 fills: no arbitrage exists. Every leg's resting
          order is cancelled and the zero-fill leg(s) are reported.

        *legs* is `[{token_id, price, shares, side, ...}, ...]` — optional
        per-leg keys `bet_amount_usd`, `asset_type`, `no_token_id`, and
        `group_id` (read from the first leg) enrich logging and let
        dry-run reuse execute_trade()'s existing simulation path.

        Dry-run/paper (no live client): every leg is simulated as a
        complete fill — same as execute_trade()'s existing behavior — and
        the timeout is never waited on, only the group structure is
        logged. There's no real order book to poll fills against in
        either case, so both collapse to the same simulated path.
        """
        if not legs:
            return {"success": False, "arb_sets": 0, "fills": {}, "unfilled": [], "surplus": {}}

        asset_type = str(legs[0].get("asset_type") or "Arbitrage")
        group_id = legs[0].get("group_id")

        if self.dry_run or self.client is None:
            return self._simulate_full_fill_arbitrage_group(legs, log_func, asset_type, group_id, strategy_tag)

        return await self._execute_live_arbitrage_group(
            legs, float(timeout_seconds), log_func, strategy_tag, asset_type, group_id,
        )

    def _strategy_max_daily_trades(self, strategy_tag: str) -> int:
        """Per-strategy daily trade cap (see core/trading_config.py).
        Unrecognized strategy tags fall back to the global max_daily_trades."""
        if strategy_tag == "arbitrage":
            return int(getattr(self.config, "arbitrage_max_daily_trades", self.config.max_daily_trades))
        if strategy_tag == "crypto":
            return int(getattr(self.config, "crypto_max_daily_trades", self.config.max_daily_trades))
        return int(self.config.max_daily_trades)

    def evaluate_and_execute(
        self,
        market: MarketData,
        fair_value: float,
        ev: float,
        current_poly_price: float,
        bet_amount_usd: float,
        side: str,
        log_func: Callable,
        strategy_tag: str = "crypto",
    ) -> bool:
        """Gate on EV + daily limit + price bounds, then fire the order."""
        asset_type = market.asset_type
        token_id = market.market_id
        execution_side = str(side or "YES").upper()

        execution_token_id = token_id
        execution_price = float(current_poly_price)
        execution_fair_value = float(fair_value)
        if execution_side == "NO":
            execution_token_id = str(getattr(market, "no_market_id", None) or token_id)
            execution_price = max(1e-6, 1.0 - float(current_poly_price))
            execution_fair_value = 1.0 - float(fair_value)

        if ev < self.config.min_ev:
            return False

        strategy_max_daily_trades = self._strategy_max_daily_trades(strategy_tag)
        if self.trades_by_strategy.get(strategy_tag, 0) >= strategy_max_daily_trades:
            log_func("RISK", asset_type, token_id,
                     f"Daily trade limit ({strategy_max_daily_trades}) reached for {strategy_tag}")
            return False

        if not self._validate_market(market, log_func):
            return False

        if execution_price < PRICE_FLOOR or execution_price > PRICE_CEILING:
            log_func("EXECUTION", asset_type, execution_token_id, {
                "reason": "entry price out of bounds",
                "side": execution_side,
                "execution_price": round(execution_price, 4),
                "price_floor": PRICE_FLOOR,
                "price_ceiling": PRICE_CEILING,
            })
            return False

        if execution_price <= 0:
            log_func("EXECUTION", asset_type, execution_token_id,
                     f"Invalid market price for execution: {execution_price}")
            return False

        shares = math.floor((bet_amount_usd / execution_price) * 100.0) / 100.0
        if shares <= 0:
            log_func("EXECUTION", asset_type, execution_token_id,
                     f"Calculated zero shares for bet_amount_usd={bet_amount_usd}")
            return False

        log_func("AUTO-TRADE", asset_type, execution_token_id, {
            "market_price": execution_price,
            "fair_value": execution_fair_value,
            "ev": round(ev, 4),
            "bet_amount_usd": bet_amount_usd,
            "shares": shares,
            "side": execution_side,
        })

        executed = self.execute_trade(
            token_id=token_id,
            current_poly_price=float(current_poly_price),
            shares=shares,
            bet_amount=bet_amount_usd,
            asset_type=asset_type,
            side=execution_side,
            no_token_id=getattr(market, "no_market_id", None),
            log_func=log_func,
        )
        if executed:
            self.trade_count_today += 1
            self.trades_by_strategy[strategy_tag] = self.trades_by_strategy.get(strategy_tag, 0) + 1
        return bool(executed)

    def _validate_market(self, market: MarketData, log_func: Callable) -> bool:
        for field_name, field_value in [
            ("market_id", market.market_id),
            ("asset_type", market.asset_type),
            ("strike_price", market.strike_price),
            ("question", market.question),
        ]:
            if field_value is None:
                log_func("VALIDATE", "Market", "Unknown", f"Missing {field_name}")
                return False
        return True

    def reset_daily_count(self):
        self.trade_count_today = 0
        self.trades_by_strategy = {}

    def get_execution_stats(self) -> Dict[str, Any]:
        return {
            "trades_today": self.trade_count_today,
            "daily_limit": self.config.max_daily_trades,
            "ev_threshold": self.config.min_ev,
            "trades_by_strategy": dict(self.trades_by_strategy),
        }
