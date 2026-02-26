"""Run a single hunt cycle across all default hunters and print findings.

This is a quick smoke-test script to validate hunters after merging venvs.
"""
from hunters import get_default_hunters


def main():
    hunters = get_default_hunters()
    print(f"Found {len(hunters)} hunters")
    for h in hunters:
        print(f"Running hunt: {h.__class__.__name__}")
        try:
            m = h.hunt()
            if m:
                print(f"  -> Found: {m.get('market_id')} | {m.get('asset_type')} | strike={m.get('strike_price')} | vol={m.get('volume')}")
            else:
                print("  -> No market found")
        except Exception as e:
            print(f"  -> Error during hunt: {e}")


if __name__ == '__main__':
    main()
