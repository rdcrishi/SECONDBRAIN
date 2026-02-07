"""
Audio Processor Utility for NexusMind
Transcribes audio files using OpenAI Whisper
"""
import whisper
import os
from pathlib import Path

def transcribe_audio(audio_path):
    """
    Transcribe audio file to text using Whisper
    
    Args:
        audio_path (str): Path to audio file
        
    Returns:
        dict: {
            'success': bool,
            'text': str,
            'duration': float,
            'language': str,
            'error': str (if failed)
        }
    """
    try:
        # Check if file exists
        if not os.path.exists(audio_path):
            return {
                'success': False,
                'error': f'Audio file not found: {audio_path}'
            }
        
        # Get file info
        file_size = os.path.getsize(audio_path)
        file_ext = Path(audio_path).suffix.lower()
        
        # Validate audio format
        supported_formats = ['.mp3', '.wav', '.m4a', '.ogg', '.flac', '.aac']
        if file_ext not in supported_formats:
            return {
                'success': False,
                'error': f'Unsupported audio format: {file_ext}. Supported: {", ".join(supported_formats)}'
            }
        
        print(f"🎵 Loading Whisper model...")
        # Load Whisper model (base is a good balance)
        # Models: tiny, base, small, medium, large
        model = whisper.load_model("base")
        
        print(f"🎤 Transcribing audio file: {Path(audio_path).name}")
        print(f"📊 File size: {file_size / 1024 / 1024:.2f} MB")
        
        # Transcribe
        result = model.transcribe(audio_path)
        
        # Extract information
        transcribed_text = result['text'].strip()
        detected_language = result.get('language', 'unknown')
        
        # Get segments for duration calculation
        segments = result.get('segments', [])
        duration = segments[-1]['end'] if segments else 0
        
        print(f"✅ Transcription complete!")
        print(f"📝 Text length: {len(transcribed_text)} characters")
        print(f"⏱️ Duration: {duration:.1f} seconds")
        print(f"🌍 Language: {detected_language}")
        
        return {
            'success': True,
            'text': transcribed_text,
            'duration': duration,
            'language': detected_language,
            'segments': segments,
            'file_size': file_size
        }
        
    except Exception as e:
        print(f"❌ Transcription error: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

def format_transcript_with_timestamps(segments):
    """
    Format transcript with timestamps
    
    Args:
        segments (list): Whisper segments with timestamps
        
    Returns:
        str: Formatted transcript with timestamps
    """
    formatted = []
    
    for segment in segments:
        start_time = segment['start']
        end_time = segment['end']
        text = segment['text'].strip()
        
        # Format time as MM:SS
        start_min = int(start_time // 60)
        start_sec = int(start_time % 60)
        
        timestamp = f"[{start_min:02d}:{start_sec:02d}]"
        formatted.append(f"{timestamp} {text}")
    
    return "\n".join(formatted)
