from fastapi import FastAPI, Query
from sentence_transformers import SentenceTransformer
import chromadb
import uvicorn
import groq
from typing import List, Dict
from flask import jsonify
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
groq_api_key = "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"  # Replace with your actual Groq API key
groq_client = groq.Groq(api_key=groq_api_key)
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Add your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Initialize Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')
# Connect to existing ChromaDB
persist_directory = "./chroma_db"
client = chromadb.PersistentClient(path=persist_directory)
collection = client.get_collection("documents")
groq_response=""
@app.post("/query")
async def query_documents(
    query: str = Query(..., description="The query string to search for"),
    n_results: int = Query(1, description="Number of results to return")
):
    global groq_response
    # Generate embedding for the query
    query_embedding = model.encode(query).tolist()
    # Query ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    # Format the results
    formatted_results = []
    for doc, metadata, distance in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
        formatted_results.append({
            "content": doc[:200] + "..." if len(doc) > 200 else doc,  # Show first 200 characters
            "metadata": metadata,
            "similarity_score": 1 - distance  # Convert distance to similarity score
        })
    groq_response=generate_groq_response(query, formatted_results)
    print(groq_response)

    return {
        "query": query,
        "results": formatted_results,
        "groq_response": groq_response  
    }
def generate_groq_response(query: str, results: List[Dict]):
    # Prepare the prompt for Groq
    prompt = f"Based on the following query and search results, provide a concise and informative answer:\n\nQuery: {query}\n\nSearch Results:\n"
    
    low_similarity_warning = False
    for i, result in enumerate(results, 1):
        prompt += f"{i}. Content: {result['content']}\n   Metadata: {result['metadata']}\n   Similarity Score: {result['similarity_score']:.2f}\n"
        if result['similarity_score'] < 0.2:
            low_similarity_warning = True

    if low_similarity_warning:
        prompt += "\nNOTE: There is not a very relevant result for this query, but this is the best match we have. Please proceed with caution.\n"

    prompt += "\nPlease synthesize an answer based on these results and remember to include the most relevant url for the answer you are giving along with a short summary of the most relevant answer based on results received."
    # Call the Groq API
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant that provides concise and accurate information based on given search results."},
            {"role": "user", "content": prompt}
        ],
        model="mixtral-8x7b-32768",  # You can change this to other available models
        max_tokens=150,  # Adjust as needed
        temperature=0.5  # Adjust for desired creativity/randomness
    )
    
    return chat_completion.choices[0].message.content.strip()
@app.get("/getresponse")
async def get_llm_response():
    return JSONResponse(content={"response": groq_response})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)  # Note: Using port 8001 to avoid conflict with your existing API