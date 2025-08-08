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

def extract_text_from_image(image_path):
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return ""

def clean_text(text, remove_stopwords=False):

    text = re.sub(r'\s+', ' ', text) 
    text = text.strip()

    text = text.lower()

    text = re.sub(r'\b[o]\b', '0', text)
    text = re.sub(r'\b[l]\b', '1', text)

    # Replace 'i' with '1' if between digits or adjacent to a digit, remove spaces
    text = re.sub(r'(?<=\d)\s*i\s*(?=\d)', '1', text)   # between digits
    text = re.sub(r'(?<=\d)\s*i\b', '1', text)          # after a digit
    text = re.sub(r'\bi\s*(?=\d)', '1', text)           # before a digit


    text = re.sub(r'(.)\1{2,}', r'\1', text)

    if remove_stopwords:
        doc = nlp(text)
        tokens = [token.text for token in doc if not token.is_stop]
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

def extract_text_for_classification(image_path):
    # Extract raw text from image
    raw_text = extract_text_from_image(image_path)
    
    if not raw_text.strip():
        return {
            "raw_text": "",
            "cleaned_text": "",
            "classification_features": {},
            "success": False,
            "error": "No text extracted from image"
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
        "file_path": image_path
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
    features["has_invoice_number"] = bool(re.search(r'invoice\s*#?\s*\d+', cleaned_text, re.IGNORECASE))
    features["has_amount"] = bool(re.search(r'\$\d+\.?\d*', cleaned_text))
    features["has_total"] = bool(re.search(r'total.*\$\d+', cleaned_text, re.IGNORECASE))
    features["has_due_date"] = bool(re.search(r'due.*date', cleaned_text, re.IGNORECASE))
    features["has_billing"] = bool(re.search(r'billing|payment|remittance', cleaned_text, re.IGNORECASE))
    
    # Email-specific features
    features["has_email_address"] = bool(re.search(r'\w+@\w+\.\w+', cleaned_text))
    features["has_subject"] = bool(re.search(r'subject:', cleaned_text, re.IGNORECASE))
    features["has_email_headers"] = bool(re.search(r'from:.*to:', cleaned_text, re.IGNORECASE))
    features["has_greeting"] = bool(re.search(r'dear \w+|hello|hi', cleaned_text, re.IGNORECASE))
    features["has_signature"] = bool(re.search(r'sincerely|regards|best wishes', cleaned_text, re.IGNORECASE))
    
    # Document structure features
    features["has_tables"] = bool(re.search(r'(\w+\s+){3,}\$?\d+', cleaned_text))  # Likely table structure
    features["has_addresses"] = bool(re.search(r'\d+\s+\w+\s+(street|st|avenue|ave|road|rd)', cleaned_text, re.IGNORECASE))
    features["has_phone"] = bool(re.search(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', cleaned_text))
    
    # Count specific keywords
    invoice_keywords = ['invoice', 'bill', 'receipt', 'payment', 'due', 'amount', 'total', 'tax', 'subtotal']
    email_keywords = ['subject', 'from', 'to', 'sent', 'received', 'reply', 'message', 'dear', 'regards']
    
    features["invoice_keyword_count"] = sum(1 for keyword in invoice_keywords if keyword in cleaned_text)
    features["email_keyword_count"] = sum(1 for keyword in email_keywords if keyword in cleaned_text)
    
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
