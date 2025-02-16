import json
import pinecone
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

with open("retrieved.json", "r") as file:
    data = json.load(file)

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
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
text_embeddings = embedding_model.encode(fitness_texts, convert_to_numpy=True)

pinecone.init(api_key="YOUR_API_KEY", environment="YOUR_ENVIRONMENT")
index_name = "fitness-index"
dimension = text_embeddings.shape[1]

if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=dimension)

index = pinecone.Index(index_name)

ids = [f"record-{i}" for i in range(len(fitness_texts))]
vectors = [
    {"id": ids[i], "values": text_embeddings[i].tolist(), "metadata": {"text": fitness_texts[i]}}
    for i in range(len(fitness_texts))
]

index.upsert(vectors=vectors)

def retrieve_data(query):
    query_embedding = embedding_model.encode([query], convert_to_numpy=True)[0].tolist()
    result = index.query(queries=[query_embedding], top_k=3, include_metadata=True)
    matches = result["results"][0]["matches"]
    retrieved_texts = [match["metadata"]["text"] for match in matches]
    return retrieved_texts

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
    
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_length=200)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response