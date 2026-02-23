import requests

def get_live_clob_tokens():
    # Fetching the top active events from Polymarket
    url = "https://gamma-api.polymarket.com/events"
    params = {
        "active": "true",
        "closed": "false",
        "limit": 10
    }
    
    try:
        response = requests.get(url, params=params)
        events = response.json()
        
        print("\n--- 🎯 TOP ACTIVE TRADABLE TOKENS ---\n")
        
        for event in events:
            markets = event.get('markets', [])
            for m in markets:
                tokens = m.get('clobTokenIds')
                if tokens:
                    # Clean up the ID strings by removing any unwanted characters
                    yes_id = str(tokens[0]).replace('[', '').replace(']', '').replace('"', '').strip()
                    no_id = str(tokens[1]).replace('[', '').replace(']', '').replace('"', '').strip()
                    
                    print(f"Question: {m.get('question')}")
                    print(f"  ✅ YES ID: {yes_id}")
                    print(f"  ❌ NO ID:  {no_id}")
                    print("-" * 40)
                    
    except Exception as e:
        print(f"Error fetching data: {e}")

if __name__ == "__main__":
    get_live_clob_tokens()