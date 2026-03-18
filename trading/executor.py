"""
Executor: Trade execution and risk management.

Separates execution logic from the orchestration layer.
Handles position sizing, risk checks, and trade firing.
"""

import os
import logging
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass
import requests
from core.trading_config import DEFAULT_MIN_EV
from core.models import MarketData, Position

try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import OrderArgs, ApiCreds
    from py_clob_client.order_builder.constants import BUY, SELL
    CLOB_IMPORT_OK = True
except Exception as e:
    logging.error(f"Import Error: {e}")
    ClobClient = Any 
    OrderArgs = Any 
    ApiCreds = Any 
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


class TradeExecutor:
    """Handles trade execution with risk management.

    Responsibilities:
    - Evaluate expected value (EV) against thresholds
    - Check risk constraints
    - Fire execution callbacks
    - Position sizing (future enhancement)
    """

    def __init__(self, risk_config: Optional[RiskConfig] = None):
        """Initialize TradeExecutor.

        Args:
            risk_config: Risk configuration (uses defaults if not provided)
        """
        self.risk_config = risk_config or RiskConfig()
        self.trade_count_today = 0
        self.client = None
        self.proxy_address = os.getenv("POLY_ADDRESS")
        self.paper_trade_mode = str(os.getenv("PAPER_TRADE_MODE", "False")).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self.dry_run = str(os.getenv("DRY_RUN", "True")).strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        ) or self.paper_trade_mode

        proxy_address = self.proxy_address
        private_key = os.getenv("POLYGON_PRIVATE_KEY")

        if not CLOB_IMPORT_OK:
            logging.warning(
                "py-clob-client import not available. "
                "TradeExecutor will run in Paper Trading mode."
            )
            return

        if proxy_address and private_key:
            # Create the official credentials object the library expects.
            creds = ApiCreds(
                api_key=os.getenv("POLY_API_KEY"),
                api_secret=os.getenv("POLY_SECRET"),
                api_passphrase=os.getenv("POLY_PASSPHRASE"),
            )
            self.client = ClobClient(
                host="https://clob.polymarket.com",
                chain_id=137,
                key=private_key,
                creds=creds,
                funder=proxy_address,
                signature_type=1,
            )
            if self.dry_run:
                logging.warning("TradeExecutor initialized in DRY_RUN mode.")
        else:
            logging.warning(
                "TradeExecutor running in Paper Trading mode: "
                "missing POLY_ADDRESS and/or POLYGON_PRIVATE_KEY"
            )

    def get_balance(self) -> float:
        """Get available USDC balance directly from the Polygon blockchain."""
        paper_balance = float(os.getenv("PAPER_BALANCE_USD", "1000.0"))

        if self.dry_run:
            return paper_balance

        try:
            proxy_address = os.getenv("POLY_ADDRESS")
            if not proxy_address:
                logging.warning("No POLY_ADDRESS found in .env")
                return 0.0

            # The Ultimate Bypass: Read the blockchain directly instead of using the broken library
            rpc_url = "https://polygon-rpc.com"
            usdc_contract = "0x3c499c542cEF5E3811e1192ce70d8bC21B59FEe5"
            
            # Format the address for the ERC20 balanceOf function
            clean_address = proxy_address.lower().replace("0x", "")
            padded_address = clean_address.zfill(64)
            data = "0x70a08231" + padded_address
            
            payload = {
                "jsonrpc": "2.0",
                "method": "eth_call",
                "params": [{"to": usdc_contract, "data": data}, "latest"],
                "id": 1
            }
            
            resp = requests.post(rpc_url, json=payload).json()
            balance_hex = resp.get("result", "0x0")
            
            # USDC on Polygon has 6 decimal places
            real_balance = int(balance_hex, 16) / 1_000_000.0
            return real_balance
            
        except Exception as exc:
            logging.error(f"BLOCKCHAIN_FETCH_FAILURE: {exc}")
            return 0.0
        
        return paper_balance if self.dry_run else 0.0
    def get_open_positions(self) -> List[Position]:
        """Fetch current open positions and calculate mark-to-mid PnL.

        Uses Polymarket Data API because py-clob-client does not expose
        a stable positions listing method across versions.
        """
        if not self.proxy_address:
            return []

        positions: List[Position] = []
        wallet_address = str(self.proxy_address)
        url = f"https://gamma-api.polymarket.com/positions?user={wallet_address}"

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

                shares = float(raw.get("size") or raw.get("shares") or raw.get("quantity") or 0.0)
                if shares <= 0:
                    continue

                current_value = float(raw.get("currentValue") or raw.get("current_value") or 0.0)
                initial_price = float(raw.get("avgPrice") or raw.get("entry_price") or raw.get("initial_price") or 0.0)

                if current_value > 0.0 and shares > 0.0:
                    current_price = current_value / shares
                    value = current_value
                else:
                    current_price = initial_price
                    value = shares * current_price

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

        order = OrderArgs(
            price=execution_price,
            size=shares,
            side=BUY,
            token_id=execution_token_id,
        )
        try:
            resp = self.client.create_and_post_order(order)
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
            log_func(
                "LIVE-TRADE-ERROR",
                asset_type,
                execution_token_id,
                f"Order failed: {exc}",
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

        order = OrderArgs(
            price=price,
            size=shares,
            side=SELL,
            token_id=token_id,
        )
        try:
            resp = self.client.create_and_post_order(order)
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
            log_func(
                "SELL-ERROR",
                "Portfolio",
                token_id,
                f"Sell failed: {exc}",
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
        if execution_price <= 0:
            log_func(
                "EXECUTION",
                asset_type,
                execution_token_id,
                f"Invalid market price for execution: {execution_price}",
            )
            return False

        shares = round(bet_amount_usd / execution_price, 2)

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
