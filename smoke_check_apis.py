import os
import requests

# load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
FRED_API_KEY = os.getenv("FRED_API_KEY")

FRED_SERIES_MAP = {"FedRate": "FEDFUNDS"}


def check_binance(symbol):
    try:
        r = requests.get(f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}", timeout=6)
        r.raise_for_status()
        j = r.json()
        return True, j.get("price")
    except Exception as e:
        return False, str(e)


def check_openweather(location="Miami"):
    if not OPENWEATHER_API_KEY:
        return False, "OPENWEATHER_API_KEY not set"
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&units=metric&appid={OPENWEATHER_API_KEY}"
        r = requests.get(url, timeout=6)
        r.raise_for_status()
        j = r.json()
        temp = j.get("main", {}).get("temp")
        return True, temp
    except Exception as e:
        return False, str(e)


def check_fred(indicator="FedRate"):
    if not FRED_API_KEY:
        return False, "FRED_API_KEY not set"
    series = FRED_SERIES_MAP.get(indicator, indicator)
    try:
        url = (f"https://api.stlouisfed.org/fred/series/observations?series_id={series}&api_key={FRED_API_KEY}" \
               f"&file_type=json&limit=1&sort_order=desc")
        r = requests.get(url, timeout=6)
        r.raise_for_status()
        j = r.json()
        obs = j.get("observations", [])
        if not obs:
            return False, "no observations"
        return True, obs[0].get("value")
    except Exception as e:
        return False, str(e)


if __name__ == "__main__":
    print("--- API Smoke Check ---")

    ok, res = check_binance("BTCUSDT")
    print(f"Binance BTCUSDT: {'OK' if ok else 'FAIL'} -> {res}")

    ok, res = check_binance("ETHUSDT")
    print(f"Binance ETHUSDT: {'OK' if ok else 'FAIL'} -> {res}")

    ok, res = check_openweather("Miami")
    print(f"OpenWeather (Miami): {'OK' if ok else 'FAIL'} -> {res}")

    ok, res = check_fred("FedRate")
    print(f"FRED (FedRate): {'OK' if ok else 'FAIL'} -> {res}")

    print('--- End ---')
