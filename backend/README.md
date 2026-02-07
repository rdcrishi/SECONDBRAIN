# NexusMind RAG Backend

Complete RAG (Retrieval Augmented Generation) backend for the NexusMind Second Brain application.

## Features
- ✅ PDF text extraction
- ✅ Intelligent text chunking with overlap
- ✅ Google Gemini embeddings
- ✅ ChromaDB vector storage
- ✅ AI-powered question answering
- ✅ Source citation

## Setup

### 1. Create Virtual Environment
```bash
cd backend
python -m venv venv
```

### 2. Activate Virtual Environment
**Windows:**
```bash
venv\Scripts\activate
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
Edit `.env` file and add your Google Gemini API key:
```
GOOGLE_API_KEY=your_actual_api_key_here
```

Get your free API key at: https://makersuite.google.com/app/apikey

### 5. Run the Server
```bash
python app.py
```

Server will start on http://localhost:5000

## API Endpoints

### Health Check
```
GET /api/health
```

### Upload PDF
```
POST /api/upload
Content-Type: multipart/form-data

Body:
- file: PDF file
- userId: User ID (optional)
```

### Query Knowledge Base
```
POST /api/query
Content-Type: application/json

{
  "query": "What is the main topic?",
  "userId": "user-123",
  "topK": 5
}
```

### Get User Files
```
GET /api/files/{userId}
```

### Delete File
```
DELETE /api/file/{fileId}
```

## Testing

### Test Upload
```bash
curl -X POST http://localhost:5000/api/upload \
  -F "file=@test.pdf" \
  -F "userId=test-user"
```

### Test Query
```bash
curl -X POST http://localhost:5000/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Summarize the document", "userId": "test-user"}'
```

## Troubleshooting

### ChromaDB Issues
Delete the `chroma_db` folder and restart the server.

### Embedding Errors
Check that your GOOGLE_API_KEY is valid and has quota remaining.

### Memory Issues
Reduce chunk size in `chunker.py` if processing large PDFs.
