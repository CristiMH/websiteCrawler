from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import os

load_dotenv()

embedder = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# pc.create_index(
#     name="uiprime",
#     dimension=512,
#     metric="cosine",
#     spec=ServerlessSpec(cloud="aws", region="us-east-1")
# )

index = pc.Index(os.getenv("PINECONE_INDEX"))

def scrape_site(url):
    resp = requests.get(url)
    soup = BeautifulSoup(resp.content, 'html.parser')
    text = soup.get_text(separator=' ')
    return text

def split_text(text, max_words=300):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = ' '.join(words[i:i+max_words])
        chunks.append(chunk)
    return chunks

def get_embedding(text):
    embedding = embedder.encode(text)
    return embedding.tolist()

site_text = scrape_site("https://uiprime.online")

chunks = split_text(site_text)

embeddings = [get_embedding(chunk) for chunk in chunks]

for i, embed in enumerate(embeddings):
    index.upsert([
        (f"chunk-{i}", embed, {"text": chunks[i]})
    ])
    
print("✅ Upload finished successfully!")