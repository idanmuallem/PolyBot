"""ClobVenue: the live Venue adapter over py_clob_client + the Data API.

A behaviour-preserving move of the executor's former live code paths:
orders are limit orders posted to the Polymarket CLOB at the caller's
price, cash is the CLOB collateral balance, and open positions come from
the Data API for the proxy wallet.

Build one with ClobVenue.from_config(); it raises ClobUnavailable when no
usable client can be built, so live mode fails at startup instead of
reporting fake trades. Tests inject a stub client (tests/unit/test_clob_venue.py).
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable, Dict, List, Optional, Sequence

import requests
from eth_utils import to_checksum_address

from core.models import Position
from trading.venue import FillReasons, LimitOrder, MarketRef, log_placement_failed, normalize_side

try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import OrderArgs, AssetType, BalanceAllowanceParams
    from py_clob_client.order_builder.constants import BUY, SELL
    CLOB_IMPORT_OK = True
except Exception as exc:  # pragma: no cover - depends on the install
    logging.error(f"Import Error: {exc}")
    ClobClient = None
    AssetType = None
    BalanceAllowanceParams = None
    BUY = "BUY"
    SELL = "SELL"
    CLOB_IMPORT_OK = False

    class OrderArgs:  # minimal stand-in so a stub client can still be driven
        def __init__(self, token_id, price, side, size):
            self.token_id, self.price, self.side, self.size = token_id, price, side, size

logger = logging.getLogger(__name__)

CLOB_HOST = "https://clob.polymarket.com"
POLYGON_CHAIN_ID = 137
DATA_API_POSITIONS_URL = "https://data-api.polymarket.com/positions?user={user}"


class ClobUnavailable(RuntimeError):
    """No usable Polymarket CLOB client: live trading cannot start."""


def _no_log(*_args, **_kwargs) -> None:
    return None


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


def _whole_cents_of_shares(usd: float, price: float) -> float:
    """Shares *usd* buys at *price*, floored to 2 decimals (never over budget)."""
    return math.floor((float(usd) / float(price)) * 100.0) / 100.0


def is_valid_order_response(response: Any) -> bool:
    if not response:
        return False
    if isinstance(response, dict):
        if response.get("error") or response.get("errors") or response.get("errorMsg"):
            return False
        return bool(_order_id(response))
    return True


def _order_id(response: Any) -> Optional[str]:
    if not isinstance(response, dict):
        return None
    for key in ("orderID", "orderId", "order_id", "id"):
        value = response.get(key)
        if value:
            return str(value)
    return None


def _usd(raw: Any) -> float:
    value = float(raw)
    return value / 1_000_000.0 if value > 1_000_000 else value


class ClobVenue:
    """Venue adapter for real trading on the Polymarket CLOB."""

    name = "live"
    fill_reasons = FillReasons(not_filled="insufficient_liquidity", all_filled="all_legs_filled")

    def __init__(self, client: Any, proxy_address: Optional[str], http_get: Callable = requests.get):
        self.client = client
        self.proxy_address = str(proxy_address or "").strip() or None
        self._http_get = http_get

    @classmethod
    def from_config(cls, config) -> "ClobVenue":
        """Build and authenticate a CLOB client from the trading config.
        Raises ClobUnavailable with the reason when that isn't possible."""
        if not config.private_key or not config.proxy_address:
            raise ClobUnavailable(
                "missing POLYMARKET_PRIVATE_KEY / POLYMARKET_PROXY_ADDRESS (or the wallet "
                "config's private_key / proxy_address)"
            )
        if not CLOB_IMPORT_OK or ClobClient is None:
            raise ClobUnavailable("py-clob-client is not installed or failed to import")
        try:
            client = ClobClient(
                host=CLOB_HOST,
                chain_id=POLYGON_CHAIN_ID,
                key=config.private_key,
                funder=to_checksum_address(config.proxy_address),
                signature_type=config.signature_type,
            )
        except Exception as exc:
            raise ClobUnavailable(f"ClobClient failed to build: {exc}") from exc
        try:
            client.set_api_creds(client.create_or_derive_api_creds())
        except Exception as exc:
            raise ClobUnavailable(f"could not derive CLOB API credentials: {exc}") from exc
        print(f"[AUTH] Live CLOB client ready for proxy {config.proxy_address}")
        return cls(client=client, proxy_address=config.proxy_address)

    # -- Venue port ---------------------------------------------------------

    def cash(self) -> float:
        """Available CLOB collateral (deployable cash); 0.0 if unreadable."""
        try:
            if hasattr(self.client, "get_collateral_balance"):
                return _usd(self.client.get_collateral_balance())

            if hasattr(self.client, "get_balance_allowance"):
                params = (
                    BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
                    if BalanceAllowanceParams is not None else None
                )
                resp = self.client.get_balance_allowance(params=params)
                if isinstance(resp, dict):
                    balance_raw = resp.get("balance", resp)
                    if isinstance(balance_raw, dict):
                        for key in ("balance", "amount", "available", "usdc", "USDC"):
                            if balance_raw.get(key) is not None:
                                return _usd(balance_raw[key])
                    return _usd(balance_raw)
        except Exception as exc:
            logging.warning(f"Live collateral balance fetch failed: {exc}")
        return 0.0

    def open_positions(self) -> List[Position]:
        """Open positions from the Polymarket Data API, marked to the API's
        current price (py-clob-client has no stable positions listing)."""
        if not self.proxy_address:
            return []

        url = DATA_API_POSITIONS_URL.format(user=self.proxy_address)
        try:
            response = self._http_get(url, timeout=10)
            response.raise_for_status()
        except requests.exceptions.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            if status_code != 404:
                logging.debug(f"Data API positions HTTP error: {exc}")
            return []
        except requests.exceptions.RequestException as exc:
            logging.debug(f"Data API positions request error: {exc}")
            return []

        positions: List[Position] = []
        try:
            raw_positions = response.json()
            if isinstance(raw_positions, dict):
                raw_positions = raw_positions.get("positions", [])
            if not isinstance(raw_positions, list):
                return []
            for raw in raw_positions:
                position = _position_from_data_api(raw)
                if position is not None:
                    positions.append(position)
        except Exception as exc:
            logging.warning(f"Could not parse open positions: {exc}")
        return positions

    def buy(
        self, market: MarketRef, side: str, usd: float,
        price: Optional[float] = None, log_func: Optional[Callable] = None,
    ) -> bool:
        """Post a limit BUY for *usd* worth of *side* at *price* (the side's
        per-share price). LIVE-TRADE / LIVE-TRADE-ERROR rows go to log_func."""
        log = log_func or _no_log
        execution_side = normalize_side(side)
        token_id = market.token_for(execution_side)
        if not price or float(price) <= 0:
            logger.warning(f"[LIVE] No price for {token_id} -- not placing a buy")
            return False
        shares = _whole_cents_of_shares(usd, price)

        order_resp = self._submit_order(token_id, float(price), BUY, shares)
        if not is_valid_order_response(order_resp):
            error_payload = {
                "error": "order_rejected_or_unconfirmed",
                "response": order_resp, "side": execution_side,
                "price": float(price), "shares": shares,
            }
            logging.error(f"LIVE order rejected: {error_payload}")
            log("LIVE-TRADE-ERROR", "Portfolio", token_id, error_payload)
            return False

        log("LIVE-TRADE", "Portfolio", token_id, {
            "success": True, "response": order_resp,
            "side": execution_side, "price": float(price),
        })
        return True

    def sell(
        self, token_id: str, shares: float,
        price: Optional[float] = None, log_func: Optional[Callable] = None,
    ) -> bool:
        """Post a limit SELL of *shares* at *price*. SELL / SELL-ERROR rows
        go to log_func."""
        log = log_func or _no_log
        if not price or float(price) <= 0:
            logger.warning(f"[LIVE] No price for {token_id} -- not placing a sell")
            return False

        order_resp = self._submit_order(str(token_id), float(price), SELL, float(shares))
        if not is_valid_order_response(order_resp):
            error_payload = {
                "error": "order_rejected_or_unconfirmed",
                "response": order_resp, "price": price, "shares": shares,
            }
            logging.error(f"SELL order rejected: {error_payload}")
            log("SELL-ERROR", "Portfolio", token_id, error_payload)
            return False

        log("SELL", "Portfolio", token_id, {
            "success": True, "response": order_resp, "price": price, "shares": shares,
        })
        return True

    def place_limit_buy(
        self, market: MarketRef, side: str, usd: float, limit_price: float,
        shares: Optional[float] = None, log_func: Optional[Callable] = None,
    ) -> Optional[LimitOrder]:
        """Post a limit BUY of *shares* (default: what *usd* buys) at
        *limit_price*. A rejection is logged as EXECUTION-ERROR with the
        CLOB's response."""
        token_id = market.token_for(side)
        if shares is None:
            shares = _whole_cents_of_shares(usd, limit_price)
        order_resp = self._submit_order(token_id, float(limit_price), BUY, float(shares))
        order_id = _order_id(order_resp) if is_valid_order_response(order_resp) else None
        if order_id is None:
            logger.warning(f"[LIVE] Limit buy for {token_id} not accepted: {order_resp}")
            log_placement_failed(log_func, token_id, "order_placement_failed", response=order_resp)
            return None
        return LimitOrder(id=order_id, token_id=token_id)

    def poll_limit_orders(self, orders: Sequence[LimitOrder]) -> Dict[Any, float]:
        """Shares matched so far per order. ClobClient.get_order is a
        synchronous REST call; it blocks the (sequential) engine loop for
        one round-trip per order per poll."""
        filled: Dict[Any, float] = {}
        for order in orders:
            try:
                status = self.client.get_order(order.id)
            except Exception as exc:
                logging.warning(f"get_order failed for {order.id}: {exc}")
                status = None
            filled[order.id] = (
                _pick_float(status, "size_matched", "matched_size", "sizeMatched", "filled_size", "filledSize")
                if isinstance(status, dict) else 0.0
            )
        return filled

    def cancel_limit_order(self, order: LimitOrder, log_func: Optional[Callable] = None) -> bool:
        """Cancel a resting order; CANCEL (with the CLOB's response) or
        CANCEL-ERROR rows go to log_func, keyed by order id."""
        log = log_func or _no_log
        try:
            resp = self.client.cancel(order.id)
        except Exception as exc:
            logger.warning(f"[LIVE] Cancel failed for order {order.id}: {exc}")
            log("CANCEL-ERROR", "Portfolio", str(order.id), {"reason": "cancel_failed", "error": str(exc)})
            return False
        log("CANCEL", "Portfolio", str(order.id), {"response": resp})
        return True

    def resolve_closed_markets(
        self,
        resolve_arbitrage: bool = True,
        log_func: Optional[Callable] = None,
        on_resolved: Optional[Callable[[dict], Any]] = None,
    ) -> int:
        """Live markets settle on-chain; there is nothing to resolve here."""
        return 0

    # -- internals ----------------------------------------------------------

    def _submit_order(self, token_id: str, price: float, side: str, size: float) -> Any:
        """Post one limit order; the CLOB's response, or None if posting raised."""
        try:
            price = round(float(price), 2)
            size = round(float(size), 2)
            print(f"[EXECUTION] Attempting {side} order: {size} shares at ${price}")
            return self.client.create_and_post_order(
                OrderArgs(token_id=token_id, price=price, side=side, size=size)
            )
        except Exception as exc:
            print(f"[LIVE-TRADE-ERROR] {token_id} - {exc}")
            return None


def _position_from_data_api(raw: Any) -> Optional[Position]:
    if not isinstance(raw, dict):
        return None
    token_id = str(raw.get("asset") or raw.get("token_id") or raw.get("tokenId") or "")
    if not token_id:
        return None

    shares = abs(float(_pick_float(
        raw, "size", "shares", "quantity", "balance", "positionSize", "numShares"
    )))
    initial_price = _pick_float(raw, "avgPrice", "avg_price", "entry_price", "initial_price", "price")
    current_price = _pick_float(raw, "currentPrice", "current_price", "markPrice", "mark_price")
    current_value = abs(_pick_float(
        raw, "currentValue", "current_value", "positionValue", "position_value", "value", "usdValue",
    ))

    if current_value <= 0.0 and shares > 0.0 and current_price > 0.0:
        current_value = shares * current_price
    if current_price <= 0.0 and shares > 0.0 and current_value > 0.0:
        current_price = current_value / shares
    if shares <= 0.0 and current_value > 0.0 and initial_price > 0.0:
        shares = current_value / initial_price
    if shares <= 0.0 and current_value <= 0.0:
        return None

    value = current_value if current_value > 0.0 else (
        shares * (current_price if current_price > 0.0 else initial_price)
    )
    if current_price <= 0.0:
        current_price = initial_price

    pnl_ratio = (current_price - initial_price) / initial_price if initial_price > 0 else 0.0

    return Position(
        market_id=str(
            raw.get("conditionId") or raw.get("condition_id") or raw.get("market_id") or token_id
        ),
        token_id=token_id,
        initial_price=initial_price,
        current_price=current_price,
        shares=shares,
        value=value,
        pnl_ratio=pnl_ratio,
        side=str(raw.get("outcome") or raw.get("side") or "UNKNOWN"),
        live_ev=float(pnl_ratio),
    )
