"""
Executor: Trade execution and risk management.

Separates execution logic from the orchestration layer.
Handles position sizing, risk checks, and trade firing.
"""

import os
import logging
import math
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass
import requests
from eth_utils import to_checksum_address
from core.trading_config import DEFAULT_MIN_EV, TradingConfig
from core.models import MarketData, Position

ENTRY_PRICE_FLOOR = 0.30
ENTRY_PRICE_CEILING = 0.85

try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import ApiCreds, OrderArgs, AssetType, BalanceAllowanceParams
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

@dataclass
class RiskConfig:
    """Risk management configuration."""
    ev_threshold: float = DEFAULT_MIN_EV  # Minimum EV to execute trade
    max_position_size: float = 1.0  # Max position as fraction of capital
    max_daily_trades: int = 10  # Max trades per day
    stop_loss_pct: float = 0.05  # Stop loss percentage


@dataclass
class ExecutorAuthConfig:
    """Auth/config values required for Polymarket CLOB signing."""

    private_key: Optional[str]
    proxy_address: Optional[str]
    signature_type: int = 2

    @classmethod
    def from_env(cls) -> "ExecutorAuthConfig":
        private_key = str(os.getenv("POLYMARKET_PRIVATE_KEY") or "").strip() or None
        proxy_address = str(os.getenv("POLYMARKET_PROXY_ADDRESS") or "").strip() or None
        signature_type_raw = str(os.getenv("SIGNATURE_TYPE", "2")).strip()
        try:
            signature_type = int(signature_type_raw)
        except ValueError:
            signature_type = 2

        return cls(
            private_key=private_key,
            proxy_address=proxy_address,
            signature_type=signature_type,
        )


class TradeExecutor:
    """Handles trade execution with risk management.

    Responsibilities:
    - Evaluate expected value (EV) against thresholds
    - Check risk constraints
    - Fire execution callbacks
    - Position sizing (future enhancement)
    """

    def __init__(self, risk_config: Optional[RiskConfig] = None):
        self.config = TradingConfig.from_env()
        self.risk_config = risk_config or RiskConfig()
        self.trade_count_today = 0
        self.dry_run = self.config.dry_run
        self.client = None

        self.private_key = self.config.private_key
        self.proxy_address = self.config.proxy_address
        self.signature_type = self.config.signature_type
        self.paper_trade_mode = self.config.paper_trade_mode

        # --- DIAGNOSTIC BLOCK ---
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
                creds = ApiCreds(
                    api_key=os.getenv("POLY_API_KEY"),
                    api_secret=os.getenv("POLY_SECRET"),
                    api_passphrase=os.getenv("POLY_PASSPHRASE"),
                )
                self.client = ClobClient(
                    host="https://clob.polymarket.com",
                    chain_id=137,
                    key=self.config.private_key,
                    funder=to_checksum_address(self.config.proxy_address),
                    signature_type=self.config.signature_type,
                    creds=creds,
                )
                self.client.set_api_creds(creds)
                print("[SUCCESS] Live CLOB Client is fully armed and operational!")
            except Exception as e:
                print(f"[FATAL ERROR] ClobClient failed to build: {e}")
                self.client = None
        else:
            print("[FATAL] Missing keys in Config! Cannot build Client.")

    def _initialize_clob_client(self) -> None:
        """Initialize authenticated CLOB client with proxy wallet (funder)."""
        print(f"Signing for Proxy: {self.config.proxy_address}")
        creds = ApiCreds(
            api_key=os.getenv("POLY_API_KEY"),
            api_secret=os.getenv("POLY_SECRET"),
            api_passphrase=os.getenv("POLY_PASSPHRASE"),
        )
        self.client = ClobClient(
            host="https://clob.polymarket.com",
            chain_id=137,
            key=self.config.private_key,
            funder=to_checksum_address(self.config.proxy_address),
            signature_type=self.config.signature_type,
            creds=creds,
        )

        if not hasattr(self.client, "set_api_creds"):
            raise RuntimeError("py-clob-client does not expose set_api_creds()")

        self.client.set_api_creds(creds)

    def _resolve_positions_user_address(self) -> Optional[str]:
        """Resolve the wallet address used for Data API position queries."""
        explicit = str(self.proxy_address or "").strip()
        if explicit:
            return explicit

        return None

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
        """Execute a live order on the Polymarket CLOB."""
        try:
            print(f"[EXECUTION] Attempting {side} order: {size} shares at ${price}")

            order = OrderArgs(
                token_id=token_id,
                price=price,
                side=side,
                size=size,
            )
            
            resp = self.client.create_and_post_order(order)
            return resp
        except Exception as e:
            # This will catch 'invalid signature' or 'insufficient balance'
            print(f"[LIVE-TRADE-ERROR] {token_id} - {str(e)}")
            return None
        
    @staticmethod
    def _format_order_exception(exc: Exception) -> Dict[str, Any]:
        error_payload: Dict[str, Any] = {
            "error": str(exc),
            "exception_type": type(exc).__name__,
            "exception_repr": repr(exc),
        }

        response = getattr(exc, "response", None)
        if response is not None:
            status_code = getattr(response, "status_code", None)
            response_text = getattr(response, "text", None)
            if status_code is not None:
                error_payload["status_code"] = status_code
            if response_text:
                error_payload["response_text"] = response_text

        return error_payload

    def get_balance(self) -> float:
        """Fetch available CLOB collateral balance (true deployable cash)."""
        if self.dry_run or self.client is None:
            return float(os.getenv("PAPER_BALANCE_USD", "1000.0"))

        try:
            if hasattr(self.client, "get_collateral_balance"):
                raw_balance = float(self.client.get_collateral_balance())
                return raw_balance / 1_000_000.0 if raw_balance > 1_000_000 else raw_balance

            if hasattr(self.client, "get_balance_allowance"):
                params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
                resp = self.client.get_balance_allowance(params=params)
                if isinstance(resp, dict):
                    balance_raw = resp.get("balance", resp)
                    if isinstance(balance_raw, dict):
                        for key in ("balance", "amount", "available", "usdc", "USDC"):
                            if balance_raw.get(key) is not None:
                                raw_balance = float(balance_raw.get(key))
                                return raw_balance / 1_000_000.0 if raw_balance > 1_000_000 else raw_balance
                    raw_balance = float(balance_raw)
                    return raw_balance / 1_000_000.0 if raw_balance > 1_000_000 else raw_balance
        except Exception as exc:
            logging.warning(f"Live collateral balance fetch failed: {exc}")

        return 0.0
        
    def get_open_positions(self) -> List[Position]:
        """Fetch current open positions and calculate mark-to-mid PnL.

        Uses Polymarket Data API because py-clob-client does not expose
        a stable positions listing method across versions.
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
            if status_code == 404:
                logging.info("No open positions found (new wallet)")
            else:
                logging.info("No open positions found (new wallet)")
                logging.debug(f"Gamma positions HTTP error: {exc}")
            return []
        except requests.exceptions.RequestException as exc:
            logging.info("No open positions found (new wallet)")
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

                token_id = str(raw.get("asset") or raw.get("token_id") or raw.get("tokenId") or "")
                if not token_id:
                    continue

                shares_raw = self._pick_float(raw, "size", "shares", "quantity", "balance", "positionSize", "numShares")
                shares = abs(float(shares_raw))

                initial_price = self._pick_float(raw, "avgPrice", "avg_price", "entry_price", "initial_price", "price")
                current_price = self._pick_float(raw, "currentPrice", "current_price", "markPrice", "mark_price")
                current_value = abs(
                    self._pick_float(
                        raw,
                        "currentValue",
                        "current_value",
                        "positionValue",
                        "position_value",
                        "value",
                        "usdValue",
                    )
                )

                if current_value <= 0.0 and shares > 0.0 and current_price > 0.0:
                    current_value = float(shares) * float(current_price)

                if current_price <= 0.0 and shares > 0.0 and current_value > 0.0:
                    current_price = float(current_value) / float(shares)

                if shares <= 0.0 and current_value > 0.0 and initial_price > 0.0:
                    shares = float(current_value) / float(initial_price)

                if shares <= 0.0 and current_value <= 0.0:
                    continue

                if current_value > 0.0:
                    value = float(current_value)
                else:
                    ref_price = float(current_price if current_price > 0.0 else initial_price)
                    value = float(shares) * float(ref_price)

                if current_price <= 0.0:
                    current_price = float(initial_price)

                pnl_percent = 0.0
                if initial_price > 0:
                    pnl_percent = ((current_price - initial_price) / initial_price) * 100

                live_ev = pnl_percent / 100.0 if abs(pnl_percent) > 1.0 else pnl_percent

                positions.append(
                    Position(
                        market_id=str(raw.get("conditionId") or raw.get("condition_id") or raw.get("market_id") or token_id),
                        token_id=token_id,
                        initial_price=initial_price,
                        current_price=current_price,
                        shares=shares,
                        value=value,
                        pnl_percent=pnl_percent,
                        side=str(raw.get("outcome") or raw.get("side") or "UNKNOWN"),
                        live_ev=float(live_ev),
                    )
                )
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
                f"[DRY-RUN] Simulation: Would have purchased {shares} of {execution_token_id} "
                f"({execution_side}) for ${bet_amount}."
            )
            log_func(
                "DRY-RUN",
                asset_type,
                execution_token_id,
                {
                    "price": execution_price,
                    "shares": shares,
                    "bet_amount_usd": bet_amount,
                    "side": execution_side,
                },
            )
            return True

        if self.client is None:
            log_func(
                "PAPER-TRADE",
                asset_type,
                execution_token_id,
                {
                    "price": execution_price,
                    "shares": shares,
                    "bet_amount_usd": bet_amount,
                    "reason": "No live CLOB client configured",
                    "side": execution_side,
                },
            )
            return True

        try:
            resp = self._submit_order(
                token_id=execution_token_id,
                price=execution_price,
                side=BUY,
                size=shares,
            )
            live_success = True
            if isinstance(resp, dict):
                live_success = not resp.get("error") and not resp.get("errors")

            log_func(
                "LIVE-TRADE",
                asset_type,
                execution_token_id,
                {"success": live_success, "response": resp, "side": execution_side, "price": execution_price},
            )
            return live_success
        except Exception as exc:
            error_payload = self._format_order_exception(exc)
            logging.error(f"LIVE order rejected: {error_payload}")
            log_func(
                "LIVE-TRADE-ERROR",
                asset_type,
                execution_token_id,
                error_payload,
            )
            return False

    def sell_position(
        self,
        token_id: str,
        shares: float,
        price: float,
        log_func: Callable,
    ) -> bool:
        """Sell an existing position using dry-run or live execution."""
        if self.dry_run:
            msg = f"[DRY-RUN] Would have SOLD {shares} of {token_id} at ${price}"
            print(msg)
            log_func(
                "DRY-RUN-SELL",
                "Portfolio",
                token_id,
                {
                    "message": msg,
                    "price": price,
                    "shares": shares,
                },
            )
            return True

        if self.client is None:
            log_func(
                "PAPER-SELL",
                "Portfolio",
                token_id,
                {
                    "price": price,
                    "shares": shares,
                    "reason": "No live CLOB client configured",
                },
            )
            return True

        try:
            resp = self._submit_order(
                token_id=token_id,
                price=price,
                side=SELL,
                size=shares,
            )
            success = True
            if isinstance(resp, dict):
                success = not resp.get("error") and not resp.get("errors")

            log_func(
                "SELL",
                "Portfolio",
                token_id,
                {"success": success, "response": resp, "price": price, "shares": shares},
            )
            return success
        except Exception as exc:
            error_payload = self._format_order_exception(exc)
            logging.error(f"SELL order rejected: {error_payload}")
            log_func(
                "SELL-ERROR",
                "Portfolio",
                token_id,
                error_payload,
            )
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
        """Evaluate market conditions and execute trade if criteria are met.

        Args:
            market: MarketData object with market_id, asset_type, question, etc.
            fair_value: Fair value probability (our calculated price)
            ev: Expected value (fair_value - market_price) / market_price
            current_poly_price: Current Polymarket mid price
            bet_amount_usd: USD amount to allocate to this trade
            log_func: Logging callback function

        Returns:
            True if trade was executed, False otherwise
        """
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

        # ========================================
        # 1. Check EV threshold
        # ========================================
        if ev <= self.risk_config.ev_threshold:
            return False

        # ========================================
        # 2. Check daily trade limit
        # ========================================
        if self.trade_count_today >= self.risk_config.max_daily_trades:
            log_func(
                "RISK",
                asset_type,
                token_id,
                f"Daily trade limit ({self.risk_config.max_daily_trades}) reached",
            )
            return False

        # ========================================
        # 3. Check market validity
        # ========================================
        if not self._validate_market(market, log_func):
            return False

        # ========================================
        # 4. Calculate position size (future enhancement)
        # ========================================
        position_size = self._calculate_position_size(ev, fair_value)

        # ========================================
        # 5. Execute trade
        # ========================================
        if execution_price < ENTRY_PRICE_FLOOR or execution_price > ENTRY_PRICE_CEILING:
            log_func(
                "EXECUTION",
                asset_type,
                execution_token_id,
                {
                    "reason": "entry price out of bounds",
                    "side": execution_side,
                    "execution_price": round(float(execution_price), 4),
                    "price_floor": ENTRY_PRICE_FLOOR,
                    "price_ceiling": ENTRY_PRICE_CEILING,
                },
            )
            return False

        if execution_price <= 0:
            log_func(
                "EXECUTION",
                asset_type,
                execution_token_id,
                f"Invalid market price for execution: {execution_price}",
            )
            return False

        shares = math.floor((bet_amount_usd / execution_price) * 100.0) / 100.0

        if shares <= 0:
            log_func(
                "EXECUTION",
                asset_type,
                execution_token_id,
                f"Calculated zero shares for bet_amount_usd={bet_amount_usd}",
            )
            return False

        log_func(
            "AUTO-TRADE",
            asset_type,
            execution_token_id,
            {
                "market_price": execution_price,
                "fair_value": execution_fair_value,
                "ev": round(ev, 4),
                "position_size": position_size,
                "bet_amount_usd": bet_amount_usd,
                "shares": shares,
                "side": execution_side,
            },
        )

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
            return True
        return False

    def _validate_market(self, market: MarketData, log_func: Callable) -> bool:
        """Validate market conditions before execution.

        Args:
            market: MarketData object
            log_func: Logging callback

        Returns:
            True if market is valid for trading
        """
        # Check required fields
        required_fields = [
            ("market_id", market.market_id),
            ("asset_type", market.asset_type),
            ("strike_price", market.strike_price),
            ("question", market.question),
        ]
        for field_name, field_value in required_fields:
            if field_value is None:
                log_func("VALIDATE", "Market", "Unknown", f"Missing {field_name}")
                return False

        return True

    def _calculate_position_size(self, ev: float, fair_value: float) -> float:
        """Calculate position size based on EV and fair value.

        Simple Kelly-like approach (future enhancement).

        Args:
            ev: Expected value
            fair_value: Fair value probability

        Returns:
            Position size as fraction of max (0.0 to 1.0)
        """
        # Simple: size proportional to EV, capped at max
        # Higher EV = larger position
        position = min(ev * 2.0, self.risk_config.max_position_size)
        return max(0.01, position)  # Minimum 1% if trading

    def reset_daily_count(self):
        """Reset daily trade counter (call at start of each trading day)."""
        self.trade_count_today = 0

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get current execution statistics.

        Returns:
            Dict with execution stats (trades today, etc.)
        """
        return {
            "trades_today": self.trade_count_today,
            "daily_limit": self.risk_config.max_daily_trades,
            "av_ev_threshold": self.risk_config.ev_threshold,
        }
