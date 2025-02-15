#!/usr/bin/env python3
import requests
import json
import uuid
from datetime import datetime, timedelta

# Replace with your Terra API credentials and base URL.
API_KEY = '4actk-rayan-testing-1Zc12PJqrG'
BASE_URL = 'https://api.tryterra.co'  # Adjust if needed
DEV_ID = 'OwQmSDxQWwjinYp6GuxL8-bxiOuSZMDw'

# For widget session generation:
WIDGET_SESSION_URL = f"{BASE_URL}/v2/auth/generateWidgetSession"
PROVIDERS = "GARMIN,FITBIT,OURA,WITHINGS,SUUNTO"
LANGUAGE = "en"
REFERENCE_ID = "user123@email.com"
AUTH_SUCCESS_REDIRECT_URL = "https://myapp.com/success"
AUTH_FAILURE_REDIRECT_URL = "https://myapp.com/failure"

# The real user we will base our data on (as per your instructions)
REAL_USER_ID = "f7461933-2ecf-45cf-bcd6-eeac0c2f0c53"
REAL_REFERENCE_ID = "tony_stark"


# --- FUNCTIONS ---

def generate_widget_session():
    """
    Calls the Terra API endpoint to generate a widget session.
    """
    headers = {
        "x-api-key": API_KEY,
        "dev-id": DEV_ID,
        "Content-Type": "application/json"
    }
    payload = {
        "providers": PROVIDERS,
        "language": LANGUAGE,
        "reference_id": REFERENCE_ID,
        "auth_success_redirect_url": AUTH_SUCCESS_REDIRECT_URL,
        "auth_failure_redirect_url": AUTH_FAILURE_REDIRECT_URL
    }
    response = requests.post(WIDGET_SESSION_URL, headers=headers, json=payload)
    session_data = response.json()
    print("Widget session generated successfully:")
    print(json.dumps(session_data, indent=2))
    return session_data


def get_user_data(user_id, start_time, end_time):
    """
    Fetches the data for a given user over a specific time window.
    This simulates getting data from Terra after authentication.
    """
    # Example endpoint to fetch user data – adjust based on your integration
    url = f"{BASE_URL}/v2/users/{user_id}/data"
    params = {
        "start_time": start_time.isoformat() + "Z",
        "end_time": end_time.isoformat() + "Z"
    }
    headers = {"x-api-key": API_KEY}
    response = requests.get(url, params=params, headers=headers)
    if response.status_code == 200:
        print("Successfully fetched user data.")
        return response.json()
    else:
        print("Error fetching user data:", response.status_code, response.text)
        return None


def simulate_fake_member(base_data, user_name):
    """
    Clones the base user data and modifies it to simulate another team member.
    """
    new_user_id = str(uuid.uuid4())
    new_reference_id = user_name
    member = {
        "user": {
            "user_id": new_user_id,
            "reference_id": new_reference_id,
            "provider": base_data.get("user", {}).get("provider", "GARMIN"),
            "active": True,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "last_webhook_update": datetime.utcnow().isoformat() + "Z",
            "scopes": base_data.get("user", {}).get("scopes", "")
        },
        "data": {}
    }
    # Clone each data type (e.g., activity, sleep, heart_rate) and update its summary_id.
    for key, value in base_data.get("data", {}).items():
        new_value = value.copy()
        if "metadata" in new_value and "summary_id" in new_value["metadata"]:
            new_value["metadata"]["summary_id"] = f"fake-{uuid.uuid4()}"
        member["data"][key] = new_value
    return member


def main():
    # --- STEP 1: Generate Widget Session ---
    widget_session = generate_widget_session()
    if not widget_session:
        print("Cannot continue without a widget session.")
        return

    # In a real application the user would now use the provided widget URL to authenticate.
    # For this demo we assume the user authenticates and Terra processes the data.
    # --- STEP 2: Get User Data for the Past 30 Seconds ---
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(seconds=30)
    
    # Attempt to fetch data from the real user
    real_data = get_user_data(REAL_USER_ID, start_time, end_time)

    
    # --- STEP 3: Build Fake Team Data ---
    team = {
        "team": {
            "team_id": "team_fake_001",
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "members": []
        }
    }
    
    # Add the real user as the first team member.
    team["team"]["members"].append(real_data)
    
    # Define nine fake member names.
    fake_names = [
        "steve_rogers", "bruce_banner", "natasha_romanoff", "clint_barton",
        "thor_odinson", "peter_parker", "wanda_maximoff", "dr_strange", "captain_marvel"
    ]
    
    for name in fake_names:
        fake_member = simulate_fake_member(real_data, name)
        team["team"]["members"].append(fake_member)
    
    # --- STEP 4: Output the Team JSON ---
    print("\nFake Team JSON:")
    print(json.dumps(team, indent=2))


if __name__ == "__main__":
    main()