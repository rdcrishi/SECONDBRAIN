"""
Embeddings Generator using Google Gemini
"""
import google.generativeai as genai
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

def generate_embedding(text: str) -> List[float]:
    """
    Generate embedding for a single text
    
    Args:
        text: Input text
        
    Returns:
        Embedding vector as list of floats
    """
    try:
        result = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return []

def generate_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for multiple texts
    
    Args:
        texts: List of input texts
        
    Returns:
        List of embedding vectors
    """
    embeddings = []
    for text in texts:
        emb = generate_embedding(text)
        embeddings.append(emb)
    return embeddings

def generate_query_embedding(query: str) -> List[float]:
    """
    Generate embedding optimized for queries
    
    Args:
        query: Query text
        
    Returns:
        Embedding vector
    """
    try:
        result = genai.embed_content(
            model="models/embedding-001",
            content=query,
            task_type="retrieval_query"
        )
        return result['embedding']
    except Exception as e:
        print(f"Error generating query embedding: {e}")
        return []
