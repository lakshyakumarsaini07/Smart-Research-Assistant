"""
Document loading and text extraction utilities
"""
import os
import tempfile
import logging
import subprocess
import sys
from typing import List, Dict, Any, Optional, Tuple

import fitz  # PyMuPDF
import pdfplumber
import pypdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_tesseract_installation() -> Tuple[bool, str]:
    """
    Check if Tesseract OCR is installed on the system
    
    Returns:
        Tuple of (is_installed, installation_path or error_message)
    """
    try:
        # For Windows
        if sys.platform.startswith('win'):
            try:
                # Try to get the Tesseract path from where command
                result = subprocess.run(['where', 'tesseract'], capture_output=True, text=True, check=True)
                path = result.stdout.strip().split('\n')[0]
                return True, path
            except (subprocess.SubprocessError, FileNotFoundError, IndexError):
                # Check common installation paths
                common_paths = [
                    r'C:\Program Files\Tesseract-OCR\tesseract.exe',
                    r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
                ]
                for path in common_paths:
                    if os.path.exists(path):
                        return True, path
                return False, "Tesseract not found. Please install from https://github.com/UB-Mannheim/tesseract/wiki"
        # For Linux/Mac
        else:
            try:
                result = subprocess.run(['which', 'tesseract'], capture_output=True, text=True, check=True)
                path = result.stdout.strip()
                return True, path
            except (subprocess.SubprocessError, FileNotFoundError):
                return False, "Tesseract not found. Please install using your package manager (e.g., apt-get install tesseract-ocr)"
    except Exception as e:
        return False, f"Error checking Tesseract installation: {str(e)}"

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from a PDF file using PyMuPDF (faster but sometimes less accurate)
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Extracted text as a string
    """
    text = ""
    try:
        logger.info(f"Attempting to extract text with PyMuPDF from {file_path}")
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()
        
        if text.strip():
            logger.info("Text successfully extracted with PyMuPDF")
            return text
        else:
            logger.warning("PyMuPDF extracted empty text, trying pdfplumber next")
            # PyMuPDF returned empty text, try pdfplumber
            return extract_text_from_pdf_with_pdfplumber(file_path)
    except Exception as e:
        logger.warning(f"Error extracting text with PyMuPDF: {e}")
        # Fallback to pdfplumber if PyMuPDF fails
        return extract_text_from_pdf_with_pdfplumber(file_path)

def extract_text_from_pdf_with_pdfplumber(file_path: str) -> str:
    """
    Extract text from a PDF file using pdfplumber (slower but more accurate for complex layouts)
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Extracted text as a string
    """
    text = ""
    try:
        logger.info(f"Attempting to extract text with pdfplumber from {file_path}")
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text += page_text
        
        if text.strip():
            logger.info("Text successfully extracted with pdfplumber")
            return text
        else:
            logger.warning("pdfplumber extracted empty text, trying PyPDF next")
            # pdfplumber returned empty text, try PyPDF
            return extract_text_from_pdf_with_pypdf(file_path)
    except Exception as e:
        logger.warning(f"Error extracting text with pdfplumber: {e}")
        # Fallback to PyPDF if pdfplumber fails
        return extract_text_from_pdf_with_pypdf(file_path)

def extract_text_from_pdf_with_pypdf(file_path: str) -> str:
    """
    Extract text from a PDF file using PyPDF (alternative PDF extraction)
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Extracted text as a string
    """
    text = ""
    try:
        logger.info(f"Attempting to extract text with PyPDF from {file_path}")
        with open(file_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text() or ""
                text += page_text
        
        if text.strip():
            logger.info("Text successfully extracted with PyPDF")
        else:
            logger.warning("PyPDF extracted empty text, will try OCR next if available")
        
        return text
    except Exception as e:
        logger.warning(f"Error extracting text with PyPDF: {e}")
        return ""

def extract_text_with_ocr(file_path: str) -> str:
    """
    Extract text from a PDF file using OCR (Optical Character Recognition)
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Extracted text as a string
    """
    text = ""
    try:
        # Check Tesseract installation
        tesseract_installed, tesseract_info = check_tesseract_installation()
        if not tesseract_installed:
            logger.error(f"Tesseract OCR is not installed: {tesseract_info}")
            return f"OCR FAILED: {tesseract_info}"
        
        # Import optional OCR dependencies
        try:
            import pytesseract
            from PIL import Image
            from pdf2image import convert_from_path
        except ImportError as e:
            logger.error(f"Missing OCR dependencies: {e}")
            return f"OCR FAILED: Missing Python dependencies - {e}"
        
        # Set tesseract path if found in non-standard location
        if os.path.isfile(tesseract_info):
            pytesseract.pytesseract.tesseract_cmd = tesseract_info
        
        logger.info("Converting PDF to images for OCR processing")
        
        # Try with different DPI settings
        for dpi in [300, 200, 150]:  # Try with different DPI settings
            try:
                logger.info(f"Attempting OCR with DPI={dpi}")
                images = convert_from_path(file_path, dpi=dpi)
                page_texts = []
                
                for i, image in enumerate(images):
                    logger.info(f"Processing page {i+1} with OCR")
                    # Try multiple languages and enhance image if needed
                    page_text = pytesseract.image_to_string(image, lang='eng')
                    if not page_text.strip() and hasattr(image, 'convert'):
                        # Try preprocessing the image
                        bw_image = image.convert('L')  # Convert to grayscale
                        page_text = pytesseract.image_to_string(bw_image, lang='eng')
                    
                    page_texts.append(page_text)
                
                text = "\n\n".join(page_texts)
                
                if text.strip():
                    logger.info(f"OCR extraction successful with DPI={dpi}")
                    break
                else:
                    logger.warning(f"OCR with DPI={dpi} returned empty text, trying other settings")
            except Exception as e:
                logger.warning(f"OCR with DPI={dpi} failed: {e}")
        
        return text
    except Exception as e:
        logger.error(f"OCR extraction failed: {e}")
        return f"OCR FAILED: {str(e)}"

def extract_text_from_txt(file_path: str) -> str:
    """
    Extract text from a TXT file
    
    Args:
        file_path: Path to the TXT file
        
    Returns:
        Extracted text as a string
    """
    try:
        logger.info(f"Attempting to extract text from TXT file {file_path}")
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            logger.info("Text successfully extracted from TXT file with UTF-8 encoding")
            return text
    except UnicodeDecodeError:
        # Try another encoding if utf-8 fails
        try:
            logger.warning("UTF-8 decoding failed, trying latin-1 encoding")
            with open(file_path, 'r', encoding='latin-1') as file:
                text = file.read()
                logger.info("Text successfully extracted from TXT file with latin-1 encoding")
                return text
        except Exception as e:
            logger.error(f"Error extracting text from TXT file with latin-1 encoding: {e}")
            return ""
    except Exception as e:
        logger.error(f"Error extracting text from TXT file: {e}")
        return ""

def get_pdf_info(file_path: str) -> Dict[str, Any]:
    """
    Get information about a PDF file to help diagnose extraction issues
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Dictionary with PDF information
    """
    info = {"path": file_path, "size_bytes": os.path.getsize(file_path)}
    
    try:
        # Try with PyMuPDF
        with fitz.open(file_path) as doc:
            info.update({
                "page_count": len(doc),
                "metadata": doc.metadata,
                "is_encrypted": doc.is_encrypted,
                "permissions": doc.permissions if hasattr(doc, "permissions") else None
            })
    except Exception as e:
        logger.warning(f"Could not get PDF info with PyMuPDF: {e}")
        
        # Fallback to PyPDF
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                info.update({
                    "page_count": len(pdf_reader.pages),
                    "metadata": pdf_reader.metadata,
                    "is_encrypted": pdf_reader.is_encrypted
                })
        except Exception as e2:
            logger.warning(f"Could not get PDF info with PyPDF either: {e2}")
            info["error"] = f"Failed to get PDF info: {e2}"
    
    return info

def process_document(file_path: str, file_extension: str) -> str:
    """
    Process a document and extract its text content
    
    Args:
        file_path: Path to the document
        file_extension: File extension (e.g., .pdf, .txt)
        
    Returns:
        Extracted text as a string
    """
    logger.info(f"Processing document: {file_path} with extension {file_extension}")
    
    # Verify file exists
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Verify file is not empty
    if os.path.getsize(file_path) == 0:
        logger.error("The uploaded file is empty")
        raise ValueError("The uploaded file is empty")
    
    # Process file based on extension
    if file_extension.lower() == '.pdf':
        # Log PDF information to help diagnose issues
        pdf_info = get_pdf_info(file_path)
        logger.info(f"PDF information: {pdf_info}")
        
        # Try standard extraction methods first
        text = extract_text_from_pdf(file_path)
        
        # If all methods failed, try OCR
        if not text or text.strip() == "":
            logger.warning("All PDF extraction methods failed, trying OCR")
            text = extract_text_with_ocr(file_path)
            
            # Check if OCR failed due to missing dependencies
            if text.startswith("OCR FAILED:"):
                error_msg = text[11:]  # Remove "OCR FAILED: " prefix
                logger.error(f"OCR failed: {error_msg}")
                
                # Provide helpful error message
                if "Missing Python dependencies" in text:
                    raise ValueError(
                        "The document appears to be image-based (scanned) but OCR dependencies are missing. "
                        "Please install required packages: pip install pytesseract pdf2image Pillow"
                    )
                elif "Tesseract OCR is not installed" in text:
                    raise ValueError(
                        "The document appears to be image-based (scanned) but Tesseract OCR is not installed. "
                        "Please install Tesseract OCR: https://github.com/UB-Mannheim/tesseract/wiki (Windows) "
                        "or use your package manager (Linux/Mac)"
                    )
                else:
                    raise ValueError(f"OCR processing failed: {error_msg}")
            
    elif file_extension.lower() == '.txt':
        text = extract_text_from_txt(file_path)
    else:
        logger.error(f"Unsupported file format: {file_extension}")
        raise ValueError(f"Unsupported file format: {file_extension}")
    
    # Verify that text was extracted
    if not text or text.strip() == "":
        logger.error("No text could be extracted from the document")
        
        # Provide a more detailed error message
        if file_extension.lower() == '.pdf':
            pdf_info = get_pdf_info(file_path)
            is_encrypted = pdf_info.get("is_encrypted", "Unknown")
            page_count = pdf_info.get("page_count", "Unknown")
            
            error_msg = (
                "No text could be extracted from the document. "
                f"PDF info: {page_count} pages, Encrypted: {is_encrypted}. "
                "This could be due to: "
                "1) The PDF contains only images (scanned document) - install Tesseract OCR, "
                "2) The PDF is encrypted or password-protected, "
                "3) The PDF uses unusual fonts or encoding, or "
                "4) The PDF has content protection enabled."
            )
        else:
            error_msg = "No text could be extracted from the document."
        
        raise ValueError(error_msg)
    
    logger.info(f"Successfully extracted {len(text)} characters from document")
    return text

def save_uploaded_file(uploaded_file: Any) -> str:
    """
    Save an uploaded file to a temporary location
    
    Args:
        uploaded_file: The uploaded file object from Streamlit
        
    Returns:
        Path to the saved file
    """
    # Create a temporary file with the same extension as the uploaded file
    file_extension = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
        tmp.write(uploaded_file.getvalue())
        return tmp.name

def create_document_chunks(text: str, chunk_size: int, chunk_overlap: int) -> List[Document]:
    """
    Split the document text into chunks for processing
    
    Args:
        text: The document text
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of document chunks
    """
    if not text or not isinstance(text, str):
        raise ValueError("Invalid document text provided")
    
    # Create a text splitter with the specified parameters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    
    # Split the text into chunks
    chunks = text_splitter.create_documents([text])
    
    # Verify that we have at least one chunk
    if not chunks:
        raise ValueError("Failed to create document chunks - text may be too short")
    
    return chunks
