
import os
import numpy as np
import faiss
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# ─── Configuration ─────────────────────────────────────
# Load Gemini API Key from environment variable or directly
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or "AIzaSyBE_XmEyvLJKHm35uMkrH--9-YaKhnZAgQ"
genai.configure(api_key=GEMINI_API_KEY)

# ─── Load FAISS Index and Text Data ─────────────────────
index = faiss.read_index("financial_advisor_index.faiss")
text_chunks = np.load("financial_text_chunks.npy", allow_pickle=True)

# ─── Load Embedding Model ────────────────────────────────
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ─── RAG Chatbot Response Function ───────────────────────
def generate_chat_response(query, top_k=3):
    if not isinstance(query, str) or not query.strip():
        return "Invalid query. Please ask a proper question."

    # Step 1: Encode query into embedding
    query_embedding = embedding_model.encode([query])

    # Step 2: Search FAISS index
    _, top_indices = index.search(np.array(query_embedding).astype("float32"), top_k)
    retrieved_contexts = [text_chunks[idx] for idx in top_indices[0]]
    context_text = "\n\n".join(retrieved_contexts)

    # Step 3: Create prompt and generate response
    prompt = f"""You are a financial advisor chatbot. Use the following customer financial profiles to answer the question:

{context_text}

User Question: {query}

Answer:"""

    model = genai.GenerativeModel("models/gemini-1.5-flash-latest")
    response = model.generate_content(prompt)

    return response.text
