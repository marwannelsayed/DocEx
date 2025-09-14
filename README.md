# DocEx - Document Classification System

A PyTorch-based machine learning system for classifying scanned documents as either **emails** or **not emails** using OCR and neural networks.

## 🚀 Features

- **Multi-Format Support**: Processes PDFs, images (PNG, JPG, TIFF, BMP, GIF) and scanned documents
- **OCR Text Extraction**: Advanced text extraction using Tesseract OCR with preprocessing
- **PDF Processing**: Handles both searchable PDFs (direct text) and scanned PDFs (OCR)
- **Binary Classification**: PyTorch-based deep learning model for email detection
- **Feature Engineering**: 20+ engineered features optimized for email detection
- **Data Processing**: Automated dataset creation from Kaggle datasets with balanced sampling
- **Interactive Testing**: Command-line interface for testing classifications
- **REST API**: FastAPI service for easy integration with document management systems

## 📋 Requirements

### System Dependencies
- **Tesseract OCR**: For optical character recognition
- **Python 3.8+**: Programming language
- **spaCy English Model**: For natural language processing

### Python Packages
```
torch
scikit-learn
pandas
numpy
pytesseract
Pillow
spacy
kagglehub
pdfminer.six
pdf2image
reportlab  # For testing PDF creation
fastapi    # For API service
uvicorn    # For API server
python-multipart  # For file uploads
```

## 🛠️ Installation

### 1. Install Tesseract OCR

**macOS:**
```bash
brew install tesseract
```

**Windows:**
```bash
winget install --id UB-Mannheim.TesseractOCR
```

Or download from: https://github.com/UB-Mannheim/tesseract/wiki

### 2. Install Python Dependencies
```bash
pip install torch scikit-learn pandas numpy pytesseract Pillow spacy kagglehub pdfminer.six pdf2image fastapi uvicorn python-multipart reportlab
```

### 3. Install spaCy English Model
```bash
python -m spacy download en_core_web_sm
```

### 4. Configure Kaggle API (Required for dataset download)
1. Create a Kaggle account at https://www.kaggle.com/
2. Go to Account → API → Create New API Token
3. Download `kaggle.json` and place it in your project directory
4. **Important**: Never commit `kaggle.json` to version control (it's in `.gitignore`)

### 5. Verify Installation
```bash
python test_dependencies.py
```

## 📁 Project Structure

```
DocEx/
├── app.py                    # Dataset creation from Kaggle
├── extract.py               # OCR and feature extraction
├── classifier.py            # PyTorch neural network
├── train_classifier.py      # Training pipeline
├── test_classifier.py       # Testing and inference
├── diagnose_model.py        # Model debugging
├── test_dependencies.py     # Installation verification
├── training_dataset.csv     # Generated training data
├── document_classifier.pth  # Trained model
├── vectorizer.pkl           # Text preprocessing
└── README.md
```

## 🚀 Quick Start

### 1. Create Training Dataset
```bash
python app.py
```
This downloads email and non-email document images from Kaggle and creates `training_dataset.csv` with balanced sampling (all emails + 100 files per non-email folder).

### 2. Train the Classifier
```bash
python train_classifier.py
```
This trains a PyTorch neural network for binary email classification and saves the model.

### 3. Test Classifications
```bash
python test_classifier.py
```
Interactive testing with sample texts and your own images.

## 🔧 Usage Examples

### Classify Text
```python
from classifier import DocumentClassifierTrainer

trainer = DocumentClassifierTrainer()
result = trainer.predict("Subject: Meeting Tomorrow From: john@company.com")

print(f"Class: {result['predicted_class']}")  # "email" or "not email"
print(f"Confidence: {result['confidence']:.3f}")
```

### Classify PDF Document
```python
from extract import extract_text_for_classification
from classifier import DocumentClassifierTrainer

# Extract text from PDF
result = extract_text_for_classification("document.pdf")
text = result["cleaned_text"]

# Classify
trainer = DocumentClassifierTrainer()
prediction = trainer.predict(text)
print(f"Document type: {prediction['predicted_class']}")
```

### Classify Image Document
```python
from extract import extract_text_for_classification
from classifier import DocumentClassifierTrainer

# Extract text from image
result = extract_text_for_classification("document.jpg")
text = result["cleaned_text"]

# Classify
trainer = DocumentClassifierTrainer()
prediction = trainer.predict(text)
print(f"Document type: {prediction['predicted_class']}")
```

## 🧠 Model Architecture

The classifier uses a 4-layer neural network for binary classification:

```
Input (5000 TF-IDF features)
    ↓
Linear(5000 → 256) → ReLU → Dropout(30%)
    ↓
Linear(256 → 128) → ReLU → Dropout(30%)
    ↓
Linear(128 → 64) → ReLU → Dropout(30%)
    ↓
Linear(64 → 1) → Sigmoid [Email=1, Not Email=0]
```

## 📊 Features Extracted for Email Detection

### Email-Specific Features
- **Email addresses**: Detects `@` symbols and email patterns
- **Subject lines**: Looks for "Subject:" headers
- **Email headers**: Detects "From:", "To:", "Sent:", "Received:"
- **Greetings**: Identifies "Dear", "Hello", "Hi" patterns
- **Signatures**: Detects "Regards", "Sincerely", "Best wishes"
- **Email keyword count**: Counts email-related terms

### Document Structure Features
- **Text statistics**: Word count, character count, line count
- **Named entities**: Person, organization, date, money entities
- **Document formatting**: Tables, addresses, phone numbers
- **Non-email indicators**: Invoice numbers, billing terms, amounts

### Advanced Text Processing
- **PDF text extraction**: Direct text extraction for searchable PDFs
- **PDF OCR processing**: OCR for scanned PDF pages using pdf2image
- **Image preprocessing**: Auto-resize, RGB conversion for better OCR
- **Error correction**: Fixes common OCR mistakes
- **Text cleaning**: Removes artifacts, normalizes whitespace

## 🔍 Troubleshooting

### Common Issues

**1. Tesseract not found**
```
Error: tesseract is not installed or it's not in your PATH
```
**Solution**: Install Tesseract OCR and ensure it's in your PATH, or the code will automatically set the path to `C:\Program Files\Tesseract-OCR\tesseract.exe`.

**2. spaCy model missing**
```
OSError: Can't find model 'en_core_web_sm'
```
**Solution**: 
```bash
python -m spacy download en_core_web_sm
```

## 📈 Performance Tips

### Improving Email Detection Accuracy
1. **Balanced Dataset**: The app automatically processes all emails + 100 files per non-email folder
2. **Multi-format Support**: Use high-quality PDFs, clear images with proper preprocessing
3. **Better OCR**: For scanned documents, use high-resolution images or PDFs
4. **Feature Engineering**: Focus on email-specific patterns in `extract_classification_features()`
5. **Hyperparameter Tuning**: Adjust learning rate, epochs, batch size for binary classification

### Training Parameters
```python
trainer.train(
    csv_path="training_dataset.csv",
    epochs=50,           # Increase for better convergence
    batch_size=32,       # Adjust based on memory
    learning_rate=0.001  # Lower for fine-tuning
)
```

## 📄 File Descriptions

| File | Purpose |
|------|---------|
| `app.py` | Downloads Kaggle datasets and creates balanced training CSV |
| `extract.py` | Advanced text extraction from PDFs and images with email-focused feature engineering |
| `classifier.py` | PyTorch binary classification neural network |
| `train_classifier.py` | Complete training pipeline for email detection |
| `test_classifier.py` | Interactive testing interface for email classification |
| `api.py` | FastAPI service for document classification with multi-format support |
| `test_pdf_support.py` | Test script for PDF processing functionality |

## 🌐 API Integration with DocRepo

DocEx provides a FastAPI service that can be integrated with document management systems like [DocRepo](https://github.com/marwannelsayed/DocRepo.git).

### 🚀 Quick API Setup

1. **Install API dependencies:**
```bash
pip install fastapi uvicorn python-multipart pdfminer.six pdf2image
```

2. **Start the API server:**
```bash
python api.py
```

3. **Test the API with multiple formats:**
```bash
python test_pdf_support.py
```

### 📡 API Endpoints

- **`GET /`** - Health check and API status
- **`POST /classify/text`** - Classify text as email or not email
- **`POST /classify/document`** - Classify uploaded documents (PDF, JPG, PNG, BMP, TIFF, GIF)
- **`GET /model/info`** - Get model information and status

### 🔗 Integration with DocRepo

#### Frontend Integration (JavaScript)
```javascript
// Classify uploaded document in DocRepo (supports PDF and images)
async function classifyDocument(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch('http://localhost:8000/classify/document', {
        method: 'POST',
        body: formData
    });
    
    const result = await response.json();
    
    // Use classification result in DocRepo
    if (result.success) {
        console.log(`Document classified as: ${result.predicted_class}`);
        console.log(`Confidence: ${result.confidence}`);
        console.log(`Message: ${result.message}`); // Shows if PDF or image was processed
        
        // Add email tag to document metadata
        if (result.predicted_class === 'email') {
            addDocumentTag(documentId, 'email');
        }
    }
    
    return result;
}
```

#### Backend Integration (Python/FastAPI)
```python
import requests

def integrate_classification_with_docrepo(document_path, document_id):
    """
    Integrate email classification with DocRepo document processing
    Supports PDF, JPG, PNG, BMP, TIFF, GIF formats
    """
    
    # Classify the document using DocEx API
    with open(document_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(
            'http://localhost:8000/classify/document',
            files=files
        )
    
    if response.status_code == 200:
        result = response.json()
        
        # Auto-tag documents in DocRepo based on classification
        if result['predicted_class'] == 'email' and result['confidence'] > 0.8:
            # Add email tag with high confidence
            add_document_tag(document_id, 'email', confidence=result['confidence'])
            
        # Log processing details
        print(f"Processed: {result['message']}")
        return result
    
    return None
```

### 🏗️ Integration Architecture

```
DocRepo Frontend
       ↓ (Upload PDF/Image Document)
DocRepo Backend 
       ↓ (API Call)
DocEx Classification API (localhost:8000)
       ↓ (PDF Text Extraction OR OCR + ML Classification)
Email/Not Email Result
       ↓ (Return Classification + Processing Info)
DocRepo Backend
       ↓ (Auto-tag Document)
DocRepo Database
```

### 📊 Classification Response Format

```json
{
    "success": true,
    "predicted_class": "email",
    "confidence": 0.95,
    "probabilities": {
        "email": 0.95,
        "not email": 0.05
    },
    "message": "Classification successful"
}
```

### 🔧 DocRepo Integration Benefits

1. **Multi-Format Support** - Automatically processes PDFs and images upon upload
2. **Automatic Email Detection** - Documents are classified regardless of format
3. **Smart Tagging** - Auto-tag emails for better organization
4. **Enhanced Search** - Filter documents by type (email/not email)
5. **Metadata Enrichment** - Add classification confidence scores and processing info
6. **Workflow Automation** - Route emails to specific folders or users
7. **PDF Intelligence** - Handle both searchable and scanned PDF documents

### 📝 Complete Integration Steps

1. **Set up DocEx API** (this repository)
2. **Clone DocRepo** from https://github.com/marwannelsayed/DocRepo.git
3. **Modify DocRepo upload handler** to call DocEx classification API
4. **Update DocRepo database schema** to include classification tags
5. **Add frontend UI** to display classification results
6. **Implement filtering/search** by document classification

For detailed integration instructions, see `API_INTEGRATION.md`.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 🙏 Acknowledgments

- **Kaggle**: For providing the document datasets for email detection
- **Tesseract OCR**: For optical character recognition
- **spaCy**: For natural language processing
- **PyTorch**: For deep learning framework

**DocEx - AI-powered email detection and document classification system**
