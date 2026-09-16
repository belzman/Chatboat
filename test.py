import os
import sys
import requests
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings

load_dotenv()
api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not api_key:
    print("❌ Error: HUGGINGFACEHUB_API_TOKEN not found in .env file.")
    sys.exit(1)

# Using a robust multilingual model that is guaranteed to support the Inference API
model_id = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

print(f"🔄 Initializing connection to {model_id}...")

embeddings = HuggingFaceEndpointEmbeddings(
    model=model_id,
    huggingfacehub_api_token=api_key
)

# Test text
test_text = "ጤና ይስጥልኝ" 

try:
    print(f"📡 Sending request for: '{test_text}'...")
    vector = embeddings.embed_query(test_text)
    print(f"✅ Success! Vector length: {len(vector)}")
    print(f"📊 Preview: {vector[:3]}...")
except Exception as e:
    print(f"❌ Connection failed: {e}")
    print("\n--- Diagnostic Check ---")
    print(f"Token length: {len(api_key) if api_key else 0}")
    print(f"Token starts with: {api_key[:4] if api_key else 'None'}")