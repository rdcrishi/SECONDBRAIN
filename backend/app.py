"""
Main Flask Application for NexusMind RAG Backend with Ollama
"""
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import uuid
from datetime import datetime
import numpy as np
from openai import OpenAI

# Import utilities
from utils.pdf_processor import extract_text_from_pdf
from utils.chunker import chunk_with_metadata
from utils.audio_processor import transcribe_audio
from utils.ocr_processor import extract_text_from_image

# Add FFmpeg to PATH for transcription support
FFMPEG_PATH = r"C:\Users\Himank Suiwala\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin"
if os.path.exists(FFMPEG_PATH) and FFMPEG_PATH not in os.environ["PATH"]:
    os.environ["PATH"] += os.pathsep + FFMPEG_PATH

# Initialize Flask app
app = Flask(__name__, static_folder='..', static_url_path='')
CORS(app)  # Enable CORS for frontend

# Configuration
UPLOAD_FOLDER = 'uploads'
OLLAMA_BASE_URL = 'http://localhost:11434'

# Initialize OpenAI client for Ollama
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # required but ignored by Ollama
)

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# In-memory vector store
vector_store = {
    'embeddings': [],  # List of numpy arrays
    'texts': [],       # Chunk texts
    'metadata': []     # Metadata (file_id, file_name, page, etc.)
}

print("✅ Flask app initialized successfully!")
print(f"📁 Upload folder: {UPLOAD_FOLDER}")
print(f"🤖 Ollama URL: {OLLAMA_BASE_URL}")

# ===== OLLAMA HELPER FUNCTIONS =====

def generate_embedding(text):
    """Generate embedding using Ollama nomic-embed-text model via OpenAI client"""
    try:
        # Note: OpenAI client doesn't have embeddings endpoint for Ollama yet
        # Fall back to direct API call for embeddings
        import requests
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={
                "model": "nomic-embed-text",
                "prompt": text
            }
        )
        response.raise_for_status()
        data = response.json()
        return np.array(data['embedding'])
    except Exception as e:
        print(f"❌ Embedding error: {e}")
        return None

def generate_chat_response(prompt):
    """Generate chat response using Ollama llama3.2 model via OpenAI client"""
    try:
        response = client.chat.completions.create(
            model="llama3.2",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"❌ Chat generation error: {e}")
        return None

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def search_similar_chunks(query_embedding, top_k=5):
    """Search for most similar chunks to query"""
    if len(vector_store['embeddings']) == 0:
        return []
    
    similarities = []
    for i, emb in enumerate(vector_store['embeddings']):
        sim = cosine_similarity(query_embedding, emb)
        similarities.append((i, sim))
    
    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Get top K
    top_indices = [idx for idx, sim in similarities[:top_k]]
    
    results = []
    for idx in top_indices:
        results.append({

            'text': vector_store['texts'][idx],
            'metadata': vector_store['metadata'][idx],
            'similarity': similarities[idx][1]
        })
    
    return results

# ===== STATIC FILE ROUTES =====

@app.route('/')
def index():
    """Serve homepage"""
    return send_from_directory('..', 'nexusmind-homepage.html')

@app.route('/login')
def login():
    """Serve login page"""
    return send_from_directory('..', 'nexusmind-login.html')

@app.route('/dashboard')
def dashboard():
    """Serve dashboard page"""
    return send_from_directory('..', 'nexusmind-dashboard.html')

# ===== API ROUTES =====

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    # Check if Ollama is running
    try:
        import requests
        requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        ollama_status = "running"
    except:
        ollama_status = "not running"
    
    return jsonify({
        'status': 'healthy',
        'message': 'NexusMind RAG Backend is running!',
        'ollama': ollama_status,
        'documents': len(set(m['file_id'] for m in vector_store['metadata'])),
        'chunks': len(vector_store['texts']),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload and process a PDF or audio file"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        user_id = request.form.get('userId', 'default_user')
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Detect file type
        file_extension = file.filename.split('.')[-1].lower()
        is_pdf = file_extension == 'pdf'
        is_audio = file_extension in ['mp3', 'wav', 'm4a', 'ogg', 'flac', 'aac']
        is_image = file_extension in ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp']
        
        if not (is_pdf or is_audio or is_image):
            return jsonify({'error': 'Unsupported file type. Allowed: PDF, Audio, Images'}), 400
        
        # Generate unique file ID
        file_id = str(uuid.uuid4())
        saved_filename = f"{file_id}.{file_extension}"
        
        # Create user folder
        user_folder = os.path.join(UPLOAD_FOLDER, user_id)
        os.makedirs(user_folder, exist_ok=True)
        
        # Save file
        file_path = os.path.join(user_folder, saved_filename)
        file.save(file_path)
        
        # Process based on file type
        audio_duration = None
        if is_pdf:
            print(f"📄 Processing PDF: {file.filename}")
            
            # Extract text from PDF
            pdf_data = extract_text_from_pdf(file_path)
            
            if not pdf_data['success']:
                return jsonify({'error': f"Failed to extract PDF: {pdf_data['error']}"}), 500
            
            print(f"✅ Extracted {pdf_data['num_pages']} pages, {pdf_data['char_count']} characters")
            
            # Chunk the text with metadata
            chunks = chunk_with_metadata(pdf_data['pages'], chunk_size=1000, overlap=100)
            print(f"✂️  Created {len(chunks)} chunks")
            
            file_type = 'pdf'
            num_pages = pdf_data['num_pages']
            
        elif is_image:
            print(f"🖼️  Processing Image: {file.filename}")
            
            # Extract text from image
            ocr_data = extract_text_from_image(file_path)
            
            if not ocr_data['success']:
                return jsonify({'error': f"Failed to process image: {ocr_data.get('error')}"}), 500
                
            print(f"✅ OCR Extracted {len(ocr_data['text'])} characters (Conf: {ocr_data['confidence']:.2f})")
            
            # Create chunks (treat as 1 page)
            pages = [{'page': 1, 'text': ocr_data['text']}]
            chunks = chunk_with_metadata(pages, chunk_size=1000, overlap=100)
            print(f"✂️  Created {len(chunks)} chunks")
            
            file_type = 'image'
            num_pages = 1

        else:  # Audio file
            print(f"🎵 Processing Audio: {file.filename}")
            
            # Transcribe audio
            audio_data = transcribe_audio(file_path)
            
            if not audio_data['success']:
                return jsonify({'error': f"Failed to transcribe audio: {audio_data['error']}"}), 500
            
            print(f"✅ Transcribed {audio_data['duration']:.1f} seconds, {len(audio_data['text'])} characters")
            print(f"🌍 Detected language: {audio_data['language']}")
            
            # Create chunks from transcript
            # Treat entire transcript as one "page" for chunking
            pages = [{'page': 1, 'text': audio_data['text']}]
            chunks = chunk_with_metadata(pages, chunk_size=1000, overlap=100)
            print(f"✂️  Created {len(chunks)} chunks")
            
            file_type = 'audio'
            num_pages = 1  # Audio files have 1 "page" (the transcript)
            audio_duration = audio_data.get('duration')
        
        # Generate embeddings and store (same for both types)
        for i, chunk_data in enumerate(chunks):
            # Generate embedding
            embedding = generate_embedding(chunk_data['text'])
            
            if embedding is None:
                print(f"⚠️  Warning: Failed to generate embedding for chunk {i}")
                continue
            
            # Store in vector store
            vector_store['embeddings'].append(embedding)
            vector_store['texts'].append(chunk_data['text'])
            vector_store['metadata'].append({
                'file_id': file_id,
                'file_name': file.filename,
                'file_type': file_type,
                'user_id': user_id,
                'page': chunk_data['page'],
                'chunk_index': i,
                'total_chunks': len(chunks),
                'duration': audio_duration,
                'uploaded_at': datetime.now().isoformat()
            })
        
        print(f"💾 Stored {len(chunks)} chunks in vector store")
        
        return jsonify({
            'success': True,
            'file_id': file_id,
            'file_name': file.filename,
            'file_type': file_type,
            'num_pages': num_pages,
            'chunks': len(chunks),
            'num_pages': num_pages,
            'chunks': len(chunks),
            'message': f'{file_type.upper()} file processed successfully!'
        })
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/query', methods=['POST'])
def query_knowledge_base():
    """Query the knowledge base with AI"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        user_id = data.get('userId', 'default_user')
        top_k = data.get('topK', 5)
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        print(f"🔍 Query: {query}")
        
        # Check if any documents exist
        if len(vector_store['texts']) == 0:
            return jsonify({
                'answer': "I couldn't find any documents in the knowledge base. Please upload some documents (PDF or Audio) first!",
                'sources': []
            })
        
        # Generate query embedding
        query_embedding = generate_embedding(query)
        
        if query_embedding is None:
            return jsonify({'error': 'Failed to generate query embedding'}), 500
        
        # Search for similar chunks
        results = search_similar_chunks(query_embedding, top_k)
        
        # Filter by user_id
        results = [r for r in results if r['metadata']['user_id'] == user_id]
        
        if len(results) == 0:
            return jsonify({
                'answer': "I couldn't find any relevant information in your uploaded documents.",
                'sources': []
            })
        
        # Build context from retrieved chunks
        context_parts = []
        sources = []
        
        for i, result in enumerate(results):
            context_parts.append(f"[Source {i+1}]: {result['text']}")
            sources.append({
                'file_name': result['metadata']['file_name'],
                'page': result['metadata']['page'],
                'snippet': result['text'][:200] + '...' if len(result['text']) > 200 else result['text']
            })
        
        context = "\n\n".join(context_parts)
        
        # Generate answer with Ollama
        prompt = f"""You are a helpful AI assistant for a student's Second Brain knowledge base.

Answer the following question based ONLY on the provided context from the student's uploaded documents.
If the answer cannot be found in the context, say so clearly.

Context from documents:
{context}

Question: {query}

Provide a clear, helpful answer that matches the tone of the student's question—whether it's professional, layman, or casual—while remaining grounded in the provided context. Cite which sources you used."""

        answer = generate_chat_response(prompt)
        
        if answer is None:
            return jsonify({'error': 'Failed to generate AI response'}), 500
        
        print(f"✅ Generated answer using {len(sources)} sources")
        
        return jsonify({
            'answer': answer,
            'sources': sources
        })
    
    except Exception as e:
        print(f"❌ Query error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/query-stream', methods=['POST'])
def query_knowledge_base_stream():
    """Query the knowledge base with AI - streaming version for stop capability"""
    try:
        from flask import Response, stream_with_context
        import json as json_module
        
        data = request.get_json()
        query = data.get('query', '')
        user_id = data.get('userId', 'default_user')
        top_k = data.get('topK', 5)
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        print(f"🔍 Streaming Query: {query}")
        
        # Check if any documents exist
        if len(vector_store['texts']) == 0:
            def generate_error():
                yield f"data: {json_module.dumps({'error': 'No documents found'})}\n\n"
                yield "data: [DONE]\n\n"
            return Response(stream_with_context(generate_error()), content_type='text/event-stream')
        
        # Generate query embedding
        query_embedding = generate_embedding(query)
        
        if query_embedding is None:
            def generate_error():
                yield f"data: {json_module.dumps({'error': 'Failed to generate query embedding'})}\n\n"
                yield "data: [DONE]\n\n"
            return Response(stream_with_context(generate_error()), content_type='text/event-stream')
        
        # Search for similar chunks
        results = search_similar_chunks(query_embedding, top_k)
        
        # Filter by user_id
        results = [r for r in results if r['metadata']['user_id'] == user_id]
        
        if len(results) == 0:
            def generate_error():
                yield f"data: {json_module.dumps({'error': 'No relevant information found'})}\n\n"
                yield "data: [DONE]\n\n"
            return Response(stream_with_context(generate_error()), content_type='text/event-stream')
        
        # Build context from retrieved chunks
        context_parts = []
        sources = []
        
        for i, result in enumerate(results):
            context_parts.append(f"[Source {i+1}]: {result['text']}")
            sources.append({
                'file_name': result['metadata']['file_name'],
                'page': result['metadata']['page'],
                'snippet': result['text'][:200] + '...' if len(result['text']) > 200 else result['text']
            })
        
        context = "\n\n".join(context_parts)
        
        # Generate answer with Ollama - STREAMING
        prompt = f"""You are a helpful AI assistant for a student's Second Brain knowledge base.

Answer the following question based ONLY on the provided context from the student's uploaded documents.
If the answer cannot be found in the context, say so clearly.

Context from documents:
{context}

Question: {query}

Provide a clear, helpful answer that matches the tone of the student's question—whether it's professional, layman, or casual—while remaining grounded in the provided context. Cite which sources you used."""

        def generate_stream():
            try:
                # Stream response from Ollama
                response = client.chat.completions.create(
                    model="llama3.2",
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    stream=True  # Enable streaming
                )
                
                for chunk in response:
                    if chunk.choices[0].delta.content:
                        token = chunk.choices[0].delta.content
                        yield f"data: {json_module.dumps({'token': token})}\n\n"
                
                # Send sources at the end
                yield f"data: {json_module.dumps({'sources': sources})}\n\n"
                yield "data: [DONE]\n\n"
                
                print(f"✅ Streamed answer using {len(sources)} sources")
                
            except Exception as e:
                print(f"❌ Streaming error: {str(e)}")
                yield f"data: {json_module.dumps({'error': str(e)})}\n\n"
                yield "data: [DONE]\n\n"
        
        return Response(stream_with_context(generate_stream()), content_type='text/event-stream')
    
    except Exception as e:
        print(f"❌ Query error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/files/<user_id>', methods=['GET'])
def get_user_files(user_id):
    """Get list of files for a user"""
    try:
        # Get unique file IDs for this user
        user_files = {}
        for metadata in vector_store['metadata']:
            if metadata['user_id'] == user_id:
                file_id = metadata['file_id']
                if file_id not in user_files:
                    user_files[file_id] = {
                        'id': file_id,
                        'name': metadata['file_name'],
                        'type': metadata.get('file_type', 'pdf'),
                        'chunks': metadata['total_chunks'],
                        'duration': metadata.get('duration'),
                        'uploaded_at': metadata['uploaded_at']
                    }
        
        files_list = list(user_files.values())
        
        return jsonify({
            'files': files_list,
            'count': len(files_list)
        })
    
    except Exception as e:
        print(f"❌ Error getting files: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/file/<file_id>', methods=['DELETE'])
def delete_file(file_id):
    """Delete a file and its embeddings"""
    try:
        # Find indices of chunks belonging to this file
        indices_to_delete = []
        for i, metadata in enumerate(vector_store['metadata']):
            if metadata['file_id'] == file_id:
                indices_to_delete.append(i)
        
        if len(indices_to_delete) == 0:
            return jsonify({'error': 'File not found'}), 404
        
        # Delete from vector store (in reverse order to maintain indices)
        for idx in reversed(indices_to_delete):
            del vector_store['embeddings'][idx]
            del vector_store['texts'][idx]
            del vector_store['metadata'][idx]
        
        print(f"🗑️  Deleted {len(indices_to_delete)} chunks for file {file_id}")
        
        return jsonify({
            'success': True,
            'message': f'Deleted {len(indices_to_delete)} chunks'
        })
    
    except Exception as e:
        print(f"❌ Delete error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = 5000
    print(f"\n🚀 Starting NexusMind RAG Backend on port {port}...")
    print(f"📊 Visit http://localhost:{port}/api/health to check status\n")
    app.run(debug=True, port=port, host='0.0.0.0')
