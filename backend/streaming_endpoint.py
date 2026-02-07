"""
Streaming query endpoint for real-time AI responses with stop capability
"""
from flask import Response, stream_with_context
import json

@app.route('/api/query-stream', methods=['POST'])
def query_knowledge_base_stream():
    """Query the knowledge base with AI - streaming version"""
    try:
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
                yield f"data: {json.dumps({'error': 'No documents found'})}\n\n"
                yield "data: [DONE]\n\n"
            return Response(stream_with_context(generate_error()), content_type='text/event-stream')
        
        # Generate query embedding
        query_embedding = generate_embedding(query)
        
        if query_embedding is None:
            def generate_error():
                yield f"data: {json.dumps({'error': 'Failed to generate query embedding'})}\n\n"
                yield "data: [DONE]\n\n"
            return Response(stream_with_context(generate_error()), content_type='text/event-stream')
        
        # Search for similar chunks
        results = search_similar_chunks(query_embedding, top_k)
        
        # Filter by user_id
        results = [r for r in results if r['metadata']['user_id'] == user_id]
        
        if len(results) == 0:
            def generate_error():
                yield f"data: {json.dumps({'error': 'No relevant information found'})}\n\n"
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

Provide a clear, concise answer based on the context above. Cite which sources you used."""

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
                        yield f"data: {json.dumps({'token': token})}\n\n"
                
                # Send sources at the end
                yield f"data: {json.dumps({'sources': sources})}\n\n"
                yield "data: [DONE]\n\n"
                
                print(f"✅ Streamed answer using {len(sources)} sources")
                
            except Exception as e:
                print(f"❌ Streaming error: {str(e)}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                yield "data: [DONE]\n\n"
        
        return Response(stream_with_context(generate_stream()), content_type='text/event-stream')
    
    except Exception as e:
        print(f"❌ Query error: {str(e)}")
        return jsonify({'error': str(e)}), 500
