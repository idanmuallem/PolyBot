import os
from decimal import Decimal

from dotenv import load_dotenv

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import AssetType, BalanceAllowanceParams

try:
    from py_clob_client.exceptions import PolyApiException
except Exception:  # Backward compatibility across SDK versions
    class PolyApiException(Exception):
        pass


HOST = "https://clob.polymarket.com"
CHAIN_ID = 137
SIGNATURE_TYPE = 2


def _to_usd(raw_value) -> str:
    """Best-effort formatter for 6-decimal collateral balances."""
    try:
        value = Decimal(str(raw_value))
    except Exception:
        return str(raw_value)

    if value > 1_000_000:
        value = value / Decimal("1000000")
    return f"${value:,.6f}"


def main() -> None:
    load_dotenv()

    private_key = (os.getenv("POLYMARKET_PRIVATE_KEY") or "").strip()
    proxy_address = (os.getenv("POLYMARKET_PROXY_ADDRESS") or "").strip()

    if not private_key:
        raise ValueError("Missing POLYMARKET_PRIVATE_KEY in .env")
    if not proxy_address:
        raise ValueError("Missing POLYMARKET_PROXY_ADDRESS in .env")

    print("[DIAG] Initializing ClobClient...")
    print(f"[DIAG] Host: {HOST}")
    print(f"[DIAG] Chain ID: {CHAIN_ID}")
    print(f"[DIAG] Signature Type: {SIGNATURE_TYPE}")
    print(f"[DIAG] Proxy Funder: {proxy_address}")

    client = ClobClient(
        host=HOST,
        chain_id=CHAIN_ID,
        key=private_key,
        funder=proxy_address,
        signature_type=SIGNATURE_TYPE,
    )

    try:
        client.set_api_creds(client.create_or_derive_api_creds())

        address = client.get_address()
        if hasattr(client, "get_collateral_balance"):
            collateral_balance = client.get_collateral_balance()
        else:
            params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
            allowance = client.get_balance_allowance(params=params)
            if isinstance(allowance, dict):
                collateral_balance = allowance.get("balance", allowance)
            else:
                collateral_balance = allowance

        print("[DIAG] get_address() ->", address)
        print("[DIAG] get_collateral_balance() ->", collateral_balance)
        print("[DIAG] Interpreted collateral balance ->", _to_usd(collateral_balance))

    except PolyApiException as exc:
        print("[DIAG][PolyApiException] Signature/API failure:")
        print(str(exc))
    except Exception as exc:
        print("[DIAG][Unexpected Error]")
        print(type(exc).__name__, str(exc))


if __name__ == "__main__":
    main()
