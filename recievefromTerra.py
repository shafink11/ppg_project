#!/usr/bin/env python3
import requests
import json
import uuid
from datetime import datetime, timedelta
import pinecone
import numpy as np
from sentence_transformers import SentenceTransformer

# -------------------------
# Terra API Configuration
# -------------------------
API_KEY = '4actk-rayan-testing-1Zc12PJqrG'
BASE_URL = 'https://api.tryterra.co'
DEV_ID = 'OwQmSDxQWwjinYp6GuxL8-bxiOuSZMDw'
WIDGET_SESSION_URL = f"{BASE_URL}/v2/auth/generateWidgetSession"
PROVIDERS = "GARMIN,FITBIT,OURA,WITHINGS,SUUNTO"
LANGUAGE = "en"
REFERENCE_ID = "user123@email.com"
AUTH_SUCCESS_REDIRECT_URL = "https://myapp.com/success"
AUTH_FAILURE_REDIRECT_URL = "https://myapp.com/failure"

# The real user we will base our data on.
REAL_USER_ID = "f7461933-2ecf-45cf-bcd6-eeac0c2f0c53"

# -------------------------
# Pinecone Configuration
# -------------------------
PINECONE_API_KEY = "YOUR_PINECONE_API_KEY"
PINECONE_ENVIRONMENT = "YOUR_PINECONE_ENVIRONMENT"
INDEX_NAME = "terra-rt-index"

# -------------------------
# Functions for Terra API
# -------------------------
def generate_widget_session():
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
    # Clone each data type and update its summary_id.
    for key, value in base_data.get("data", {}).items():
        new_value = value.copy()
        if "metadata" in new_value and "summary_id" in new_value["metadata"]:
            new_value["metadata"]["summary_id"] = f"fake-{uuid.uuid4()}"
        member["data"][key] = new_value
    return member

# -------------------------
# Utility: Convert a Member's Data to Text
# -------------------------
def member_to_text(member):
    user_info = member.get("user", {})
    data_info = member.get("data", {})
    name = user_info.get("reference_id", "unknown")
    activity = data_info.get("activity", {}).get("metadata", {}).get("name", "N/A")
    steps = sum(dp.get("steps", 0) for dp in data_info.get("activity", {}).get("data_points", []))
    sleep_quality = data_info.get("sleep", {}).get("metadata", {}).get("sleep_quality", "N/A")
    heart_rates = [dp.get("bpm", 0) for dp in data_info.get("heart_rate", {}).get("data_points", [])]
    avg_heart_rate = sum(heart_rates) / len(heart_rates) if heart_rates else "N/A"
    text = (
        f"User: {name}\n"
        f"Activity: {activity}\n"
        f"Total Steps: {steps}\n"
        f"Sleep Quality: {sleep_quality}\n"
        f"Average Heart Rate: {avg_heart_rate}\n"
    )
    return text

# -------------------------
# Update Pinecone with TerraRT Data
# -------------------------
def update_pinecone(team):
    # Initialize Pinecone.
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    
    # Prepare text records for each team member.
    team_members = team["team"]["members"]
    texts = [member_to_text(member) for member in team_members]
    
    # Load a small Sentence Transformer model for embeddings.
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(texts, convert_to_numpy=True)
    
    # Create the index if it doesn't exist.
    if INDEX_NAME not in pinecone.list_indexes():
        dimension = embeddings.shape[1]
        pinecone.create_index(INDEX_NAME, dimension=dimension)
    index = pinecone.Index(INDEX_NAME)
    
    # Upsert each member's embedding into Pinecone.
    vectors = []
    for i, text in enumerate(texts):
        vector = embeddings[i].tolist()
        vector_id = f"terra_member_{i}"
        vectors.append({"id": vector_id, "values": vector, "metadata": {"text": text}})
    index.upsert(vectors=vectors)
    print("Pinecone vector database updated with terraRT data.")

# -------------------------
# Main Function
# -------------------------
def main():
    # --- STEP 1: Generate Widget Session ---
    widget_session = generate_widget_session()
    if not widget_session:
        print("Cannot continue without a widget session.")
        return

    # --- STEP 2: Get User Data for the Past 30 Seconds ---
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(seconds=30)
    real_data = get_user_data(REAL_USER_ID, start_time, end_time)
    
    if not real_data:
        print("Real user data not available, aborting.")
        return

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
    
    # --- STEP 5: Update Pinecone Vector Database with terraRT Data ---
    update_pinecone(team)

if __name__ == "__main__":
    main()
