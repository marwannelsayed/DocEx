import io
from pdf2image import convert_from_path
from pdfminer.high_level import extract_text
import pytesseract
from PIL import Image

def has_text(image_path):
    text = extract_text(image_path)
    return bool(text.strip())

def extract_text_native(pdf_path):
    return extract_text(pdf_path)

def extract_text_scanned(pdf_path):
    pages = convert_from_path(pdf_path)
    text = ""
    for page in pages:
        text += pytesseract.image_to_string(page)
        text += "\n"
    return text

def extract_text(pdf_path):
    if has_text(pdf_path):
        return extract_text_native(pdf_path)
    else:
        return extract_text_scanned(pdf_path)