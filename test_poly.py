import json
import os
from decimal import Decimal
from unittest.mock import Mock

import requests
from dotenv import load_dotenv
from py_clob_client.client import ClobClient

try:
    from py_clob_client.clob_types import AssetType, BalanceAllowanceParams
except Exception:
    AssetType = None
    BalanceAllowanceParams = None

try:
    from web3 import Web3
except Exception:
    Web3 = None

try:
    from eth_account import Account
except Exception:
    Account = None


HOST = "https://clob.polymarket.com"
CHAIN_ID = 137
SIGNATURE_TYPE = 2
POLYGON_RPC = os.getenv("POLYGON_RPC_URL", "https://polygon.drpc.org")
USDC_E = "0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174"


def _to_decimal(raw_value) -> Decimal:
    value = Decimal(str(raw_value))
    if value > Decimal("1000000"):
        return value / Decimal("1000000")
    return value


def _format_usd(value) -> str:
    try:
        return f"${Decimal(str(value)):.2f}"
    except Exception:
        return f"{value}"


def _derive_signer_address(private_key: str, client: ClobClient) -> str:
    if Web3 is not None:
        return Web3().eth.account.from_key(private_key).address
    if Account is not None:
        return Account.from_key(private_key).address
    return client.get_address()


def _balance_via_web3(address: str) -> Decimal:
    if Web3 is None:
        raise RuntimeError("web3 is not installed")

    w3 = Web3(Web3.HTTPProvider(POLYGON_RPC))
    if not w3.is_connected():
        raise RuntimeError(f"Unable to connect to Polygon RPC: {POLYGON_RPC}")

    abi = [
        {
            "constant": True,
            "inputs": [{"name": "_owner", "type": "address"}],
            "name": "balanceOf",
            "outputs": [{"name": "balance", "type": "uint256"}],
            "stateMutability": "view",
            "type": "function",
        }
    ]
    contract = w3.eth.contract(address=Web3.to_checksum_address(USDC_E), abi=abi)
    raw_balance = contract.functions.balanceOf(Web3.to_checksum_address(address)).call()
    return Decimal(raw_balance) / Decimal("1000000")


def _collateral_balance(client: ClobClient) -> Decimal:
    if hasattr(client, "get_collateral_balance"):
        return _to_decimal(client.get_collateral_balance())

    if hasattr(client, "get_balance_allowance") and BalanceAllowanceParams is not None and AssetType is not None:
        params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
        resp = client.get_balance_allowance(params=params)
        if isinstance(resp, dict):
            raw = resp.get("balance", resp)
            if isinstance(raw, dict):
                for key in ("balance", "amount", "available", "usdc", "USDC"):
                    if key in raw:
                        return _to_decimal(raw[key])
            return _to_decimal(raw)

    return Decimal("0")


def _fetch_positions(proxy_address: str):
    url = f"https://data-api.polymarket.com/positions?user={proxy_address}"
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    data = response.json()
    if isinstance(data, dict):
        return data.get("positions", [])
    if isinstance(data, list):
        return data
    return []


def test_to_decimal_scales_micro_usdc_values() -> None:
    assert _to_decimal(1_500_000) == Decimal("1.5")


def test_to_decimal_keeps_normal_values() -> None:
    assert _to_decimal("9.22") == Decimal("9.22")


def test_fetch_positions_accepts_list_payload(monkeypatch) -> None:
    fake_response = Mock()
    fake_response.json.return_value = [{"asset": "abc", "size": 3}]
    fake_response.raise_for_status.return_value = None

    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: fake_response)

    result = _fetch_positions("0x123")
    assert isinstance(result, list)
    assert len(result) == 1


def test_fetch_positions_accepts_wrapped_positions_payload(monkeypatch) -> None:
    fake_response = Mock()
    fake_response.json.return_value = {"positions": [{"asset": "xyz", "size": 1}]}
    fake_response.raise_for_status.return_value = None

    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: fake_response)

    result = _fetch_positions("0x456")
    assert isinstance(result, list)
    assert len(result) == 1


def main() -> None:
    load_dotenv()

    private_key = (os.getenv("POLYMARKET_PRIVATE_KEY") or "").strip()
    proxy_address = (os.getenv("POLYMARKET_PROXY_ADDRESS") or "").strip()

    if not private_key:
        raise ValueError("Missing POLYMARKET_PRIVATE_KEY in .env")
    if not proxy_address:
        raise ValueError("Missing POLYMARKET_PROXY_ADDRESS in .env")

    client = ClobClient(
        host=HOST,
        chain_id=CHAIN_ID,
        key=private_key,
        funder=proxy_address,
        signature_type=SIGNATURE_TYPE,
    )
    client.set_api_creds(client.create_or_derive_api_creds())

    signer_address = _derive_signer_address(private_key, client)

    signer_balance = Decimal("0")
    proxy_balance = Decimal("0")
    collateral = Decimal("0")
    positions = []

    signer_error = None
    proxy_error = None
    collateral_error = None
    positions_error = None

    try:
        signer_balance = _balance_via_web3(signer_address)
    except Exception as exc:
        signer_error = str(exc)

    try:
        proxy_balance = _balance_via_web3(proxy_address)
    except Exception as exc:
        proxy_error = str(exc)

    try:
        collateral = _collateral_balance(client)
    except Exception as exc:
        collateral_error = str(exc)

    try:
        positions = _fetch_positions(proxy_address)
    except Exception as exc:
        positions_error = str(exc)

    has_positions = len(positions) > 0

    print("\n=== Polymarket Deep Scan Summary ===")
    print(f"Signer Address (EOA) [{signer_address}]: {_format_usd(signer_balance)}")
    print(f"Proxy Address (Funder) [{proxy_address}]: {_format_usd(proxy_balance)}")
    print(f"CLOB Exchange Collateral: {_format_usd(collateral)}")
    print(f"Active Positions Found: {'Yes' if has_positions else 'No'}")

    if has_positions:
        print(f"Positions Count: {len(positions)}")

    if signer_error:
        print(f"[WARN] Signer balance check failed: {signer_error}")
    if proxy_error:
        print(f"[WARN] Proxy balance check failed: {proxy_error}")
    if collateral_error:
        print(f"[WARN] Collateral check failed: {collateral_error}")
    if positions_error:
        print(f"[WARN] Positions API check failed: {positions_error}")

    # Optional raw snippet for quick debug visibility
    if has_positions:
        print("Sample Position:")
        print(json.dumps(positions[0], indent=2)[:1200])


if __name__ == "__main__":
    main()
