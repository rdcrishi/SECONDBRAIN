"""
PDF Processor - Extract text from PDF files
"""
import PyPDF2
from typing import List, Dict

def extract_text_from_pdf(pdf_path: str) -> Dict[str, any]:
    """
    Extract text from PDF file with page numbers
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Dictionary with text content and metadata
    """
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)
            
            # Extract text from each page
            pages_text = []
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text = page.extract_text() or ""
                pages_text.append({
                    'page': page_num + 1,
                    'text': text.strip()
                })
            
            # Combine all text
            full_text = ' '.join([p['text'] for p in pages_text])
            
            return {
                'success': True,
                'text': full_text,
                'pages': pages_text,
                'num_pages': num_pages,
                'char_count': len(full_text)
            }
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def get_page_text(pdf_path: str, page_num: int) -> str:
    """Get text from specific page"""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            if page_num < 1 or page_num > len(pdf_reader.pages):
                return ""
            page = pdf_reader.pages[page_num - 1]
            return page.extract_text().strip()
    except:
        return ""
