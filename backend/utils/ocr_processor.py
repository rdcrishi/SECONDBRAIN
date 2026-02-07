"""
OCR Processor Utility for NexusMind
Extracts text from images and scanned documents using EasyOCR
"""
import easyocr
import fitz  # PyMuPDF
from PIL import Image
import io
import os
import numpy as np

# Initialize reader globally to avoid reloading model
# 'en' for English. Add more languages if needed.
try:
    print("🧠 Loading EasyOCR model (this may take a moment)...")
    reader = easyocr.Reader(['en'], gpu=False) # Set gpu=True if CUDA is available
    print("✅ EasyOCR model loaded")
except Exception as e:
    print(f"❌ Failed to load EasyOCR: {e}")
    reader = None

def extract_text_from_image(image_path):
    """
    Extract text from an image file
    
    Args:
        image_path (str): Path to image file
        
    Returns:
        dict: {
            'success': bool,
            'text': str,
            'confidence': float
        }
    """
    if reader is None:
        return {'success': False, 'error': 'OCR model not initialized'}
    
    try:
        print(f"👁️ OCR Processing: {os.path.basename(image_path)}")
        
        # Read text from image
        result = reader.readtext(image_path)
        
        # Join detected text
        text_parts = []
        total_conf = 0
        count = 0
        
        for (bbox, text, prob) in result:
            text_parts.append(text)
            total_conf += prob
            count += 1
            
        full_text = ' '.join(text_parts)
        avg_conf = total_conf / count if count > 0 else 0
        
        print(f"✅ OCR Complete: Found {len(full_text)} chars (Conf: {avg_conf:.2f})")
        
        return {
            'success': True,
            'text': full_text,
            'confidence': avg_conf
        }
        
    except Exception as e:
        print(f"❌ OCR Error: {e}")
        return {'success': False, 'error': str(e)}

def extract_text_from_scanned_pdf(pdf_path):
    """
    Extract text from scanned PDF by converting pages to images
    
    Args:
        pdf_path (str): Path to PDF file
        
    Returns:
        dict: {
            'success': bool,
            'pages': list of dicts {'page': int, 'text': str},
            'full_text': str
        }
    """
    if reader is None:
        return {'success': False, 'error': 'OCR model not initialized'}
    
    try:
        print(f"📄 Processing Scanned PDF: {os.path.basename(pdf_path)}")
        doc = fitz.open(pdf_path)
        pages_data = []
        
        for page_num, page in enumerate(doc):
            print(f"  - OCR Page {page_num + 1}/{len(doc)}")
            
            # Render page to image (pixmap)
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2)) # 2x zoom for better quality
            
            # Convert to PIL Image
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))
            
            # Convert PIL image to numpy array for EasyOCR
            img_np = np.array(image)
            
            # Perform OCR
            result = reader.readtext(img_np)
            
            page_text = ' '.join([text for (_, text, _) in result])
            
            if page_text.strip():
                pages_data.append({
                    'page': page_num + 1,
                    'text': page_text
                })
        
        full_text = ' '.join([p['text'] for p in pages_data])
        
        print(f"✅ PDF OCR Complete: extracted {len(full_text)} chars")
        
        return {
            'success': True,
            'pages': pages_data,
            'text': full_text,
            'num_pages': len(doc),
            'char_count': len(full_text)
        }
        
    except Exception as e:
        print(f"❌ PDF OCR Error: {e}")
        return {'success': False, 'error': str(e)}
