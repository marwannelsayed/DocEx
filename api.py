from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import tempfile
from classifier import DocumentClassifierTrainer
from extract import extract_text_for_classification
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="DocEx Email Classification API",
    description="API for classifying documents as email or not email using OCR and machine learning",
    version="1.0.0"
)

# Configure CORS for your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for your specific frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the classifier globally
classifier = None

@app.on_event("startup")
async def startup_event():
    """Initialize the classifier on startup"""
    global classifier
    try:
        classifier = DocumentClassifierTrainer()
        # Test if model exists
        test_result = classifier.predict("test")
        logger.info("Email classifier loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load classifier: {e}")
        logger.error("Make sure you have trained the model by running: python train_classifier.py")

class TextClassificationRequest(BaseModel):
    text: str

class ClassificationResponse(BaseModel):
    success: bool
    predicted_class: str
    confidence: float
    probabilities: dict
    message: str = None

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "DocEx Email Classification API",
        "status": "running",
        "classifier_loaded": classifier is not None
    }

@app.post("/classify/text", response_model=ClassificationResponse)
async def classify_text(request: TextClassificationRequest):
    """
    Classify text as email or not email
    """
    if not classifier:
        raise HTTPException(status_code=503, detail="Classifier not loaded. Please train the model first.")
    
    try:
        result = classifier.predict(request.text)
        
        return ClassificationResponse(
            success=True,
            predicted_class=result['predicted_class'],
            confidence=result['confidence'],
            probabilities=result['probabilities'],
            message="Classification successful"
        )
    except Exception as e:
        logger.error(f"Error classifying text: {e}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@app.post("/classify/document", response_model=ClassificationResponse)
async def classify_document(file: UploadFile = File(...)):
    """
    Classify uploaded document as email or not email.
    Supports: PDF, JPG, PNG, BMP, TIFF, GIF
    """
    if not classifier:
        raise HTTPException(status_code=503, detail="Classifier not loaded. Please train the model first.")
    
    # Check file type - now supporting PDFs and images
    allowed_types = [
        'image/jpeg', 'image/png', 'image/bmp', 'image/tiff', 'image/gif',
        'application/pdf'
    ]
    
    # Also check by file extension for more reliability
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.pdf'}
    file_ext = os.path.splitext(file.filename or '')[1].lower()
    
    # If no extension detected, try to determine from content type
    if not file_ext and file.content_type:
        content_type_map = {
            'image/jpeg': '.jpg',
            'image/png': '.png',
            'image/bmp': '.bmp',
            'image/tiff': '.tiff',
            'image/gif': '.gif',
            'application/pdf': '.pdf'
        }
        file_ext = content_type_map.get(file.content_type, '.tmp')
    
    if file.content_type not in allowed_types and file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Supported formats: PDF, JPG, PNG, BMP, TIFF, GIF. "
                   f"Received: {file.content_type} ({file_ext})"
        )
    
    try:
        # Save uploaded file temporarily with appropriate extension
        # Use detected file extension, defaulting to .pdf if still unknown
        if not file_ext or file_ext == '.tmp':
            file_ext = '.pdf'  # Default to PDF for unknown files
            
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # Extract text from document (PDF or image)
            extraction_result = extract_text_for_classification(tmp_file_path)
            
            if not extraction_result["success"]:
                raise HTTPException(status_code=400, detail=f"Failed to extract text from document: {extraction_result.get('error', 'Unknown error')}")
            
            if not extraction_result["cleaned_text"].strip():
                raise HTTPException(status_code=400, detail="No text found in the document")
            
            # Classify the extracted text
            classification_result = classifier.predict(extraction_result["cleaned_text"])
            
            # Determine document type for response message
            doc_type = "PDF" if file_ext.lower() == '.pdf' else "image"
            
            return ClassificationResponse(
                success=True,
                predicted_class=classification_result['predicted_class'],
                confidence=classification_result['confidence'],
                probabilities=classification_result['probabilities'],
                message=f"{doc_type} document processed successfully. Extracted {len(extraction_result['cleaned_text'])} characters."
            )
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error classifying document: {e}")
        raise HTTPException(status_code=500, detail=f"Document classification failed: {str(e)}")

@app.get("/model/info")
async def model_info():
    """
    Get information about the loaded model
    """
    if not classifier:
        return {"model_loaded": False, "message": "No model loaded"}
    
    try:
        # Check if model files exist
        model_exists = os.path.exists(classifier.model_save_path)
        vectorizer_exists = os.path.exists(classifier.vectorizer_save_path)
        
        return {
            "model_loaded": True,
            "model_file_exists": model_exists,
            "vectorizer_file_exists": vectorizer_exists,
            "model_path": classifier.model_save_path,
            "vectorizer_path": classifier.vectorizer_save_path,
            "classification_type": "binary",
            "classes": ["email", "not email"]
        }
    except Exception as e:
        return {"model_loaded": False, "error": str(e)}

if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
