import wikipedia
from youtubesearchpython import VideosSearch
import json

class GapAgent:
    def __init__(self, llm_client):
        self.client = llm_client

    def analyze_gaps(self, text, topic=None):
        """Analyze text to find missing key concepts/subtopics"""
        
        prompt = f"""You are a "Gap Detection Agent" for an educational system.
        Your goal is to identify IMPLICIT GAPS in the provided notes.
        
        Analyzed Content:
        {text[:8000]}
        
        Task:
        1. Identify the core subject/topic of the notes.
        2. Compare this against a standard comprehensive curriculum for this topic.
        3. Identify 2-3 CRITICAL subtopics or concepts that are MISSING or under-explained.
        4. Return a STRICT JSON list of these missing concepts.
        
        Output Format (JSON ONLY):
        {{
            "detected_topic": "The main topic identified",
            "missing_concepts": [
                {{
                    "concept": "Name of missing concept",
                    "reason": "Why this is critical to the topic"
                }},
                ...
            ]
        }}
        """
        
        try:
            # Call LLM (using the client passed from app.py)
            response = self.client.chat.completions.create(
                model="llama3.2",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            content = response.choices[0].message.content
            
            # Parse JSON
            cleaned_json = content.replace('```json', '').replace('```', '').strip()
            return json.loads(cleaned_json)
            
        except Exception as e:
            print(f"❌ Gap Analysis Error: {e}")
            return None

    def search_content(self, concept):
        """Search Wikipedia and YouTube for a concept"""
        results = {
            'concept': concept,
            'wiki': None,
            'youtube': []
        }
        
        # 1. Wikipedia Search
        try:
            # Search for best match
            search_res = wikipedia.search(concept, results=1)
            if search_res:
                page = wikipedia.page(search_res[0], auto_suggest=False)
                results['wiki'] = {
                    'title': page.title,
                    'summary': page.summary[:1000], # First 1000 chars
                    'url': page.url
                }
        except Exception as e:
            print(f"⚠️ Wiki Search Error for {concept}: {e}")

        # 2. YouTube Search
        try:
            videos = VideosSearch(concept + " education explanation", limit=2)
            for v in videos.result()['result']:
                results['youtube'].append({
                    'title': v['title'],
                    'link': v['link'],
                    'thumbnail': v['thumbnails'][0]['url'],
                    'duration': v['duration']
                })
        except Exception as e:
            print(f"⚠️ YouTube Search Error for {concept}: {e}")
            
        return results
