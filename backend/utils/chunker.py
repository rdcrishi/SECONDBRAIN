"""
Text Chunker - Split text into chunks for embedding
"""
from typing import List, Dict

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping chunks
    
    Args:
        text: Input text to chunk
        chunk_size: Maximum characters per chunk
        overlap: Characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    if not text or len(text) == 0:
        return []
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # If not at the end, try to break at sentence boundary
        if end < len(text):
            # Look for sentence endings within next 100 chars
            for i in range(end, min(end + 100, len(text))):
                if text[i] in '.!?\n':
                    end = i + 1
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position with overlap
        start = end - overlap if end < len(text) else len(text)
    
    return chunks

def chunk_with_metadata(pages_text: List[Dict], chunk_size: int = 1000, overlap: int = 100) -> List[Dict]:
    """
    Chunk text while preserving page metadata
    
    Args:
        pages_text: List of dicts with 'page' and 'text' keys
        chunk_size: Maximum characters per chunk
        overlap: Characters to overlap between chunks
        
    Returns:
        List of dicts with chunk text and metadata
    """
    chunks_with_meta = []
    
    for page_data in pages_text:
        page_num = page_data['page']
        text = page_data['text']
        
        # Chunk this page's text
        page_chunks = chunk_text(text, chunk_size, overlap)
        
        # Add metadata to each chunk
        for i, chunk in enumerate(page_chunks):
            chunks_with_meta.append({
                'text': chunk,
                'page': page_num,
                'chunk_index': i,
                'char_count': len(chunk)
            })
    
    return chunks_with_meta
