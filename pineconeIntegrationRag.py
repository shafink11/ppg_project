import json
import pinecone
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -------------------------
# 1. Data Preparation
# -------------------------
# Load JSON file
with open("retrieved.json", "r") as file:
    data = json.load(file)

# Extract user fitness data
users = data["team"]["members"]

def extract_data(users):
    records = []
    for user in users:
        user_id = user["user"]["user_id"]
        name = user["user"]["reference_id"]
        activity = user["data"]["activity"]["metadata"]["name"]
        steps = sum(dp["steps"] for dp in user["data"]["activity"]["data_points"])
        sleep_quality = user["data"]["sleep"]["metadata"]["sleep_quality"]
        heart_rates = [dp["bpm"] for dp in user["data"]["heart_rate"]["data_points"]]
        avg_heart_rate = sum(heart_rates) / len(heart_rates) if heart_rates else None

        record = {
            "user_id": user_id,
            "name": name,
            "activity": activity,
            "total_steps": steps,
            "sleep_quality": sleep_quality,
            "avg_heart_rate": avg_heart_rate,
        }
        records.append(record)
    
    return records

# Convert to DataFrame
fitness_df = pd.DataFrame(extract_data(users))

def df_to_text(df):
    texts = []
    for _, row in df.iterrows():
        text = (
            f"User: {row['name']}\n"
            f"Activity: {row['activity']}\n"
            f"Total Steps: {row['total_steps']}\n"
            f"Sleep Quality: {row['sleep_quality']}\n"
            f"Average Heart Rate: {row['avg_heart_rate']}\n"
        )
        texts.append(text)
    return texts

fitness_texts = df_to_text(fitness_df)

# -------------------------
# 2. Embedding Generation
# -------------------------
# Use a small Sentence Transformer for embeddings (all-MiniLM-L6-v2 is a good choice)
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
text_embeddings = embedding_model.encode(fitness_texts, convert_to_numpy=True)

# -------------------------
# 3. Pinecone Initialization & Upsert
# -------------------------
# Initialize Pinecone (replace with your actual API key and environment)
pinecone.init(api_key="YOUR_API_KEY", environment="YOUR_ENVIRONMENT")
index_name = "fitness-index"
dimension = text_embeddings.shape[1]

# Create index if it doesn't exist
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=dimension)

# Connect to the index
index = pinecone.Index(index_name)

# Prepare data for upsert: assign a unique ID to each record
ids = [f"record-{i}" for i in range(len(fitness_texts))]
vectors = [
    {"id": ids[i], "values": text_embeddings[i].tolist(), "metadata": {"text": fitness_texts[i]}}
    for i in range(len(fitness_texts))
]

# Upsert vectors into Pinecone
index.upsert(vectors=vectors)

# -------------------------
# 4. Retrieval Function Using Pinecone
# -------------------------
def retrieve_data(query):
    # Generate embedding for the query
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)[0].tolist()
    # Query Pinecone for top 3 similar records, including metadata
    result = index.query(queries=[query_embedding], top_k=3, include_metadata=True)
    matches = result["results"][0]["matches"]
    # Extract the text from the metadata of each match
    retrieved_texts = [match["metadata"]["text"] for match in matches]
    return retrieved_texts

# -------------------------
# 5. Local LLM Setup & Generation
# -------------------------
# Use Google Flan-T5 Small as the local LLM (instruction-tuned and lightweight)
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def generate_response(query):
    retrieved_texts = retrieve_data(query)
    context = "\n\n".join(retrieved_texts)
    input_text = (
        f"Based on the following fitness records:\n\n{context}\n\n"
        f"Answer the question: {query}"
    )
    
    # Tokenize and generate response
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_length=200)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response

# -------------------------
# 6. Example Queries
# -------------------------
query1 = "How active was Tony Stark last week?"
query2 = "How was Steve Rogers' sleep last night?"
query3 = "What was Bruce Banner's heart rate during his last run?"

print(generate_response(query1))
print(generate_response(query2))
print(generate_response(query3))
