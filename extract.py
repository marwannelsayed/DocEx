import io
from pdf2image import convert_from_path
from pdfminer.high_level import extract_text
import pytesseract
from PIL import Image
import spacy
import re

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
    text = extract_text_from_image(image_path)
    cleaned_text = clean_text(text, remove_stopwords=True)
    structured_data = extract_structured_data(cleaned_text)
    return structured_data
