import json


def parse_payload(payload_value) -> dict:
    if isinstance(payload_value, dict):
        return payload_value
    text = str(payload_value or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        try:
            import ast
            return ast.literal_eval(text)
        except Exception:
            return {}
