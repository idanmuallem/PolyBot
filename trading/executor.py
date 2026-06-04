"""
Trade execution engine: order submission, position tracking, and balance fetching.

Wraps the Polymarket CLOB client. Execution is gated by EV threshold, daily
trade count, and price bounds. Supports dry-run, paper-trade, and live modes.
"""

import os
import logging
import math
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

    def __init__(self):
        self.config = TradingConfig.from_env()
        self.trade_count_today = 0
        self.dry_run = self.config.dry_run
        self.paper_trade_mode = self.config.paper_trade_mode
        self.proxy_address = self.config.proxy_address
        self.client = None

        print("\n" + "=" * 30)
        print("=== EXECUTOR BOOT DIAGNOSTIC ===")
        print(f"1. CLOB Import OK:    {CLOB_IMPORT_OK}")
        print(f"2. Private Key Found: {bool(self.config.private_key)}")
        print(f"3. Proxy Addr Found:  {bool(self.config.proxy_address)}")
        print(f"4. Dry Run Mode:      {self.dry_run}")
        print("=" * 30 + "\n")

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
            return float(os.getenv("PAPER_BALANCE_USD", "1000.0"))

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

    def evaluate_and_execute(
        self,
        market: MarketData,
        fair_value: float,
        ev: float,
        current_poly_price: float,
        bet_amount_usd: float,
        side: str,
        log_func: Callable,
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

        if self.trade_count_today >= self.config.max_daily_trades:
            log_func("RISK", asset_type, token_id,
                     f"Daily trade limit ({self.config.max_daily_trades}) reached")
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

    def get_execution_stats(self) -> Dict[str, Any]:
        return {
            "trades_today": self.trade_count_today,
            "daily_limit": self.config.max_daily_trades,
            "ev_threshold": self.config.min_ev,
        }
