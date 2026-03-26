"""Utilities for extracting text and metadata from PDFs."""

import fitz  # PyMuPDF
from typing import List, Dict


class PDFTextExtractor:
    """Read PDF text with PyMuPDF."""
    
    @staticmethod
    def extract_text(pdf_path: str) -> str:
        """Extract full text from a PDF file."""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text += page.get_text()
            
            doc.close()
            return text.strip()
        
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    @staticmethod
    def extract_text_by_page(pdf_path: str) -> List[Dict[str, any]]:
        """Extract text per page with page numbers."""
        try:
            doc = fitz.open(pdf_path)
            pages_data = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text().strip()
                
                if text:
                    pages_data.append({
                        'page_number': page_num + 1,
                        'text': text
                    })
            
            doc.close()
            return pages_data
        
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    @staticmethod
    def get_pdf_info(pdf_path: str) -> Dict[str, any]:
        """Return basic PDF metadata and page count."""
        try:
            doc = fitz.open(pdf_path)
            metadata = doc.metadata
            
            info = {
                'num_pages': len(doc),
                'title': metadata.get('title', 'N/A'),
                'author': metadata.get('author', 'N/A'),
                'subject': metadata.get('subject', 'N/A'),
                'creator': metadata.get('creator', 'N/A')
            }
            
            doc.close()
            return info
        
        except Exception as e:
            raise Exception(f"Error reading PDF info: {str(e)}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        
        extractor = PDFTextExtractor()
        
        print("PDF Information:")
        info = extractor.get_pdf_info(pdf_path)
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        print("\nExtracting text...")
        text = extractor.extract_text(pdf_path)
        print(f"\nExtracted {len(text)} characters")
        print("\nFirst 500 characters:")
        print(text[:500])
    else:
        print("Usage: python pdf_extractor.py <pdf_file_path>")
