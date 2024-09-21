from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import faiss
import numpy as np
import torch
import chromadb
from chromadb.utils import embedding_functions
from fastapi import FastAPI, Query
import uvicorn
import groq
from typing import List, Dict
groq_api_key = "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"  # Replace with your actual Groq API key
groq_client = groq.Groq(api_key=groq_api_key)
app = FastAPI()
# Step 1: Load pre-trained models
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
tokenizer = AutoTokenizer.from_pretrained("gpt2")
lm_model = AutoModelForCausalLM.from_pretrained("gpt2")

# Step 2: Initialize ChromaDB and retrieve documents
persist_directory = "./chroma_db"
client = chromadb.PersistentClient(path=persist_directory)
collection = client.get_or_create_collection("documents")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Retrieve documents and their embeddings from ChromaDB
results = collection.get(
    include=['documents', 'embeddings']
)

documents = results['documents']

# Step 3: Process embeddings
if 'embeddings' in results and results['embeddings']:
    doc_embeddings = np.array(results['embeddings'])
    if len(doc_embeddings.shape) == 1:
        doc_embeddings = doc_embeddings.reshape(1, -1)
    elif len(doc_embeddings.shape) > 2:
        doc_embeddings = doc_embeddings.reshape(doc_embeddings.shape[0], -1)
    
    dimension = doc_embeddings.shape[1]
else:
    print("No embeddings found in the ChromaDB results")
    # Use the dimension of the sentence transformer model
    dimension = sentence_model.get_sentence_embedding_dimension()
    doc_embeddings = np.random.rand(1, dimension)

# Step 4: Create FAISS index
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings.astype('float32'))

# Step 5: Implement multi-head attention for aspect-aware retrieval
class MultiHeadRetrieval(torch.nn.Module):
    def __init__(self, input_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.attention = torch.nn.MultiheadAttention(input_dim, num_heads)
    
    def forward(self, query_embedding):
        # Ensure query_embedding is 3D: (seq_len, batch_size, input_dim)
        if query_embedding.dim() == 1:
            query_embedding = query_embedding.unsqueeze(0).unsqueeze(1)
        elif query_embedding.dim() == 2:
            query_embedding = query_embedding.unsqueeze(1)
        
        # Now apply attention
        attn_output, _ = self.attention(query_embedding, query_embedding, query_embedding)
        return attn_output.mean(dim=0)

multi_head_retrieval = MultiHeadRetrieval(dimension, num_heads=4)

# Step 6: RAG function
# In your rag function:
def rag(query, top_k=3):
    query_embedding = sentence_model.encode([query])[0]
    
    # Check if the collection is empty
    if collection.count() == 0:
        return "No documents found in the database. Please upload some documents first."

    # Query ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k
    )

    # Check if any results were returned
    if not results['documents']:
        return "No relevant documents found for the given query."

    # Process the results
    retrieved_docs = results['documents'][0]
    
    # Prepare input for language model
    context = " ".join(retrieved_docs)
    input_text = f"Query: {query}\nContext: {context}\nAnswer:"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    
    # Generate answer
    output = lm_model.generate(input_ids, max_length=150, num_return_sequences=1, no_repeat_ngram_size=2)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return answer
def get_multi_head_embeddings(text):
    # Generate base embedding
    base_embedding = sentence_model.encode(text)
    
    # Apply multi-head attention
    with torch.no_grad():
        multi_head_emb = multi_head_retrieval(torch.tensor(base_embedding))
    
    # Convert to list of embeddings
    return [emb.cpu().numpy().tolist() for emb in multi_head_emb]
# In your main code:
@app.post("/query")
async def query_documents(
    query: str = Query(..., description="The query string to search for"),
    n_results: int = Query(3, description="Number of results to return")
):
    # Generate embedding for the query
    query_embedding = get_multi_head_embeddings(query)  # Convert to list of floats
    
    # Query ChromaDB
    results = collection.query(
        query_embeddings=query_embedding,  # Wrap in a list as ChromaDB expects a list of embeddings
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    # Format the results
    formatted_results = []
    for doc, metadata, distance in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
        formatted_results.append({
            "content": doc[:200] + "..." if len(doc) > 200 else doc,
            "metadata": metadata,
            "similarity_score": 1 - distance
        })

    return {
        "query": query,
        "results": formatted_results
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005)