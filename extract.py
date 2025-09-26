import io
from pdf2image import convert_from_path
from pdfminer.high_level import extract_text
import pytesseract
from PIL import Image
import spacy
import re
import os

# Configure Tesseract path for Windows
if os.name == 'nt':  # Windows
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF using both text extraction and OCR"""
    try:
        text = ""
        
        # First, try to extract text directly from PDF (for searchable PDFs)
        try:
            direct_text = extract_text(pdf_path)
            if direct_text.strip():
                text = direct_text
            else:
                # If no direct text, use OCR on PDF pages
                pages = convert_from_path(pdf_path, dpi=300)
                ocr_text = ""
                for page in pages:
                    # Convert PIL image to text using OCR
                    page_text = extract_text_from_pil_image(page)
                    ocr_text += page_text + "\n"
                text = ocr_text
        except Exception as e:
            # Fallback to OCR if direct extraction fails
            pages = convert_from_path(pdf_path, dpi=300)
            ocr_text = ""
            for page in pages:
                page_text = extract_text_from_pil_image(page)
                ocr_text += page_text + "\n"
            text = ocr_text
        
        return text.strip()
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")
        return ""

def extract_text_from_pil_image(image):
    """Extract text from a PIL Image object using OCR"""
    try:
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image if too small (helps with OCR accuracy)
        width, height = image.size
        if width < 1000 or height < 1000:
            scale_factor = max(1000/width, 1000/height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            image = image.resize((new_width, new_height), Image.LANCZOS)
        
        # Configure Tesseract for better text extraction
        custom_config = '--oem 3 --psm 6'
        
        # Extract text with improved configuration
        text = pytesseract.image_to_string(image, config=custom_config)
        
        # Post-process text to fix common OCR errors
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII characters
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.strip()
        
        return text
    except Exception as e:
        print(f"Error processing PIL image: {e}")
        return ""

def extract_text_from_image(image_path):
    try:
        # Check if file is an image by extension
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif'}
        file_ext = os.path.splitext(image_path)[1].lower()
        if file_ext not in valid_extensions:
            return ""
        
        # Open and preprocess image for better OCR
        image = Image.open(image_path)
        
        # Use the common PIL image processing function
        return extract_text_from_pil_image(image)
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return ""

def extract_text_from_document(file_path):
    """
    Universal text extraction function that handles multiple file formats:
    - Images: JPG, PNG, BMP, TIFF, GIF
    - PDFs: Both searchable and scanned PDFs
    - Unknown: Try both PDF and image extraction
    """
    try:
        file_ext = os.path.splitext(file_path)[1].lower()
        
        # Handle PDF files
        if file_ext == '.pdf':
            return extract_text_from_pdf(file_path)
        
        # Handle image files
        elif file_ext in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif'}:
            return extract_text_from_image(file_path)
        
        # Handle unknown or .tmp files - try both approaches
        elif file_ext in {'.tmp', ''}:
            # First try as PDF
            try:
                pdf_text = extract_text_from_pdf(file_path)
                if pdf_text.strip():
                    return pdf_text
            except:
                pass
            
            # If PDF fails, try as image
            try:
                # Override the extension check for unknown files
                image = Image.open(file_path)
                return extract_text_from_pil_image(image)
            except:
                pass
            
            print(f"Could not process file as PDF or image: {file_path}")
            return ""
        
        # Unsupported format
        else:
            print(f"Unsupported file format: {file_ext}")
            return ""
            
    except Exception as e:
        print(f"Error processing document {file_path}: {e}")
        return ""

def clean_text(text, remove_stopwords=False):
    """Clean and normalize text for better processing."""
    
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text) 
    text = text.strip()
    
    # Convert to lowercase for consistency
    text = text.lower()
    
    # Fix common OCR character errors
    text = re.sub(r'\b[o]\b', '0', text)  # isolated 'o' to '0'
    text = re.sub(r'\b[l]\b', '1', text)  # isolated 'l' to '1'
    
    # Replace 'i' with '1' if between digits or adjacent to a digit
    text = re.sub(r'(?<=\d)\s*i\s*(?=\d)', '1', text)   # between digits
    text = re.sub(r'(?<=\d)\s*i\b', '1', text)          # after a digit
    text = re.sub(r'\bi\s*(?=\d)', '1', text)           # before a digit
    
    # Remove excessive repeated characters (OCR artifacts)
    text = re.sub(r'(.)\1{3,}', r'\1', text)  # Allow up to 2 repeats, remove more
    
    # Fix common email/document OCR issues
    text = re.sub(r'\b(frorn|frem)\b', 'from', text)    # Fix "from"
    text = re.sub(r'\b(subjecf|subjeci)\b', 'subject', text)  # Fix "subject"
    text = re.sub(r'\b(ernail|emaii)\b', 'email', text)  # Fix "email"
    text = re.sub(r'\b(dafe|dale)\b', 'date', text)     # Fix "date"
    
    # Preserve important punctuation for email detection
    text = re.sub(r'([a-zA-Z0-9])@([a-zA-Z0-9])', r'\1@\2', text)  # Ensure @ is preserved
    text = re.sub(r'([a-zA-Z0-9])\.([a-zA-Z]{2,})', r'\1.\2', text)  # Preserve domain extensions
    
    if remove_stopwords:
        doc = nlp(text)
        tokens = [token.text for token in doc if not token.is_stop and len(token.text) > 1]
        text = ' '.join(tokens)

    return text

def extract_entities(text):
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append((ent.text, ent.label_))
    return entities

def extract_custom_fields(text):
    data = {}

    inv_match = re.search(r'(?:Invoice Number[:\s]*)([A-Z0-9\-]+)', text, re.IGNORECASE)
    if inv_match:
        data["invoice_number"] = inv_match.group(1)

    date_match = re.search(r'(\d{1,4}[-/]\d{1,2}[-/]\d{2,4})', text)
    if date_match:
        data["date"] = date_match.group(1)

    total_match = re.search(r'(?:Total(?: Due)?:?\s*)(\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', text, re.IGNORECASE)
    if total_match:
        data["total"] = total_match.group(1)

    return data

def extract_structured_data(text):
    entities = extract_entities(text)
    custom_fields = extract_custom_fields(text)

    structured_data = {
        "invoice_number": custom_fields.get("invoice_number"),
        "date": custom_fields.get("date") or (entities.get("DATE")[0] if "DATE" in entities else None),
        "total": custom_fields.get("total") or (entities.get("MONEY")[0] if "MONEY" in entities else None),
        "persons": entities.get("PERSON", []),
        "organizations": entities.get("ORG", [])
    }
    return structured_data

def extract_text_for_classification(file_path):
    """
    Extract and prepare text from any supported document format for classification.
    Supports: PDF, JPG, PNG, BMP, TIFF, GIF
    """
    # Extract raw text from document (image or PDF)
    raw_text = extract_text_from_document(file_path)
    
    if not raw_text.strip():
        return {
            "raw_text": "",
            "cleaned_text": "",
            "classification_features": {},
            "success": False,
            "error": "No text extracted from document"
        }
    
    # Clean text for classification (remove stopwords to focus on distinguishing features)
    cleaned_text = clean_text(raw_text, remove_stopwords=True)
    
    # Extract classification-specific features
    classification_features = extract_classification_features(raw_text, cleaned_text)
    
    return {
        "raw_text": raw_text,
        "cleaned_text": cleaned_text,
        "classification_features": classification_features,
        "success": True,
        "file_path": file_path
    }

def extract_classification_features(raw_text, cleaned_text):
    """
    Extract features specifically useful for document classification.
    """
    features = {}
    
    # Text statistics
    features["word_count"] = len(cleaned_text.split())
    features["char_count"] = len(cleaned_text)
    features["line_count"] = len(raw_text.split('\n'))
    
    # Invoice-specific features
    features["has_invoice_number"] = bool(re.search(r'(invoice(?:\s+(?:number|no\.?|#))?[:\s]*)([A-Z0-9\-]+)', cleaned_text, re.IGNORECASE))
    features["has_seller_info"] = bool(re.search(r'seller|vendor|supplier|company|corporation', cleaned_text, re.IGNORECASE))
    features["has_items_list"] = bool(re.search(r'items?\s+descriptions?|quantity|prices?|amount|unit cost', cleaned_text, re.IGNORECASE))
    features["has_total_amount"] = bool(re.search(r'(total(?: due)?:?\s*)(\$?\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', cleaned_text, re.IGNORECASE))
    features["has_client_info"] = bool(re.search(r'client|customer|bill to|sold to', cleaned_text, re.IGNORECASE))
    features["has_date_info"] = bool(re.search(r'\b(date|due date|invoice date)[:\s]*\d{1,4}[-/]\d{1,2}[-/]\d{2,4}', cleaned_text, re.IGNORECASE))
    features["has_payment_terms"] = bool(re.search(r'payment terms|net \d+|due upon receipt', cleaned_text, re.IGNORECASE))
    features["has_tax_info"] = bool(re.search(r'tax|vat|gst|sales tax', cleaned_text, re.IGNORECASE))

    # Financial statement-specific features
    features["has_financial_terms"] = bool(re.search(r'balance|income statement|cash flow|equity|liabilities|assets|revenue|expenses', cleaned_text, re.IGNORECASE))
    features["has_transaction"] = bool(re.search(r'transactions?|debit|credit|account number|balance', cleaned_text, re.IGNORECASE))
    features["has_statement_period"] = bool(re.search(r'for the (month|quarter|year) ending|period ending|statement period', cleaned_text, re.IGNORECASE))
    features["has_statement_title"] = bool(re.search(r'financial statement|balance sheet|income statement|profit and loss|p&l|statements?', cleaned_text, re.IGNORECASE))
    features["has_accounting_terms"] = bool(re.search(r'receivables|payables|depreciation|amortization|goodwill|retained earnings', cleaned_text, re.IGNORECASE))
    features["has_financial_ratios"] = bool(re.search(r'ratio|percentage|percent|%|margin', cleaned_text, re.IGNORECASE))

    # Resume-specific features
    features["has_education"] = bool(re.search(r'education|degree|bachelor|master|phd|university|college|school|graduated', cleaned_text, re.IGNORECASE))
    features["has_experience"] = bool(re.search(r'experience|worked at|responsibilities|projects?|employment|career|position', cleaned_text, re.IGNORECASE))
    features["has_skills"] = bool(re.search(r'skills?|proficiencies?|technologies?|tools?|programming|languages?', cleaned_text, re.IGNORECASE))
    features["has_certifications"] = bool(re.search(r'certifications?|licensed|certified|awards?|achievements?', cleaned_text, re.IGNORECASE))
    features["has_contact_info"] = bool(re.search(r'phone|email|contact|address|linkedin|portfolio', cleaned_text, re.IGNORECASE))
    features["has_objective"] = bool(re.search(r'objective|summary|profile|about me|career goal', cleaned_text, re.IGNORECASE))
    features["has_job_titles"] = bool(re.search(r'manager|developer|engineer|analyst|director|specialist|coordinator', cleaned_text, re.IGNORECASE))
    
    # Email-specific features (enhanced)
    features["has_email_address"] = bool(re.search(r'\w+@\w+\.\w+', cleaned_text))
    features["has_subject"] = bool(re.search(r'subject:', cleaned_text, re.IGNORECASE))
    features["has_email_headers"] = bool(re.search(r'from:.*to:|sent:|received:', cleaned_text, re.IGNORECASE))
    features["has_greeting"] = bool(re.search(r'dear \w+|hello|hi|greetings', cleaned_text, re.IGNORECASE))
    features["has_signature"] = bool(re.search(r'sincerely|regards|best wishes|thank you|thanks', cleaned_text, re.IGNORECASE))
    features["has_reply_forward"] = bool(re.search(r'reply|forward|fwd|re:|fw:', cleaned_text, re.IGNORECASE))
    features["has_cc_bcc"] = bool(re.search(r'\b(cc|bcc):', cleaned_text, re.IGNORECASE))

    
    # Document structure features
    features["has_tables"] = bool(re.search(r'(\w+\s+){3,}\$?\d+', cleaned_text))  # Likely table structure
    features["has_addresses"] = bool(re.search(r'\d+\s+\w+\s+(street|st|avenue|ave|road|rd)', cleaned_text, re.IGNORECASE))
    features["has_phone"] = bool(re.search(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', cleaned_text))
    
    # Count specific keywords for each document type
    email_keywords = ['subject', 'from', 'to', 'sent', 'received', 'reply', 'message', 'dear', 'regards']
    invoice_keywords = ['invoice', 'bill', 'payment', 'total', 'amount', 'due', 'customer', 'vendor']
    financial_keywords = ['balance', 'assets', 'liabilities', 'equity', 'revenue', 'expenses', 'profit', 'loss']
    resume_keywords = ['experience', 'education', 'skills', 'university', 'degree', 'worked', 'position', 'responsibilities']
    
    features["email_keyword_count"] = sum(1 for keyword in email_keywords if keyword in cleaned_text)
    features["invoice_keyword_count"] = sum(1 for keyword in invoice_keywords if keyword in cleaned_text)
    features["financial_keyword_count"] = sum(1 for keyword in financial_keywords if keyword in cleaned_text)
    features["resume_keyword_count"] = sum(1 for keyword in resume_keywords if keyword in cleaned_text)
    
    # Entity-based features
    entities = extract_entities(raw_text)
    entity_dict = {}
    for entity, label in entities:
        if label not in entity_dict:
            entity_dict[label] = []
        entity_dict[label].append(entity)
    
    features["money_entities"] = len(entity_dict.get("MONEY", []))
    features["person_entities"] = len(entity_dict.get("PERSON", []))
    features["org_entities"] = len(entity_dict.get("ORG", []))
    features["date_entities"] = len(entity_dict.get("DATE", []))
    
    return features
