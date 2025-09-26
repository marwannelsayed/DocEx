# DocEx - Multi-Class Document Classification System

A Random Forest-based machine learning system for classifying scanned documents into **4 categories**: 
📧 **Emails** | 🧾 **Invoices** | 📊 **Financial Statements** | 📄 **Resumes**

Uses OCR text extraction and advanced feature engineering to achieve **100% accuracy** on balanced datasets.

## 🚀 Features

- **Multi-Class Classification**: Identifies 4 document types with 100% accuracy
- **Advanced Feature Engineering**: 40+ discriminative features for precise classification  
- **Random Forest Model**: Robust machine learning approach optimized for mixed feature data
- **Multi-Format Support**: Processes PDFs, images (PNG, JPG, TIFF, BMP, GIF) and scanned documents
- **OCR Text Extraction**: Advanced text extraction using Tesseract OCR with preprocessing
- **PDF Processing**: Handles both searchable PDFs (direct text) and scanned PDFs (OCR)
- **Real Dataset Training**: Uses actual Kaggle datasets for emails, invoices, financial statements, and resumes
- **Balanced Training**: Equal representation of all document types for unbiased classification
- **REST API**: FastAPI service for easy integration with document management systems
- **Interactive Testing**: Command-line interface for testing classifications

## 📋 Requirements

### System Dependencies
- **Tesseract OCR**: For optical character recognition
- **Python 3.8+**: Programming language
- **spaCy English Model**: For natural language processing

### Python Packages
```
scikit-learn      # Random Forest classifier
pandas           # Data processing
numpy            # Numerical operations
pytesseract      # OCR text extraction
Pillow           # Image processing
spacy            # Natural language processing
kagglehub        # Kaggle dataset downloads
pdfminer.six     # PDF text extraction
pdf2image        # PDF to image conversion
fastapi          # REST API framework
uvicorn          # API server
python-multipart # File upload support
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
pip install scikit-learn pandas numpy pytesseract Pillow spacy kagglehub pdfminer.six pdf2image fastapi uvicorn python-multipart
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
├── 🔧 Core System
│   ├── extract.py                    # OCR and feature extraction (40+ features)
│   ├── simple_classifier.py          # Random Forest classifier (WORKING)
│   └── api.py                        # FastAPI REST service
│
├── 📊 Data Processing  
│   ├── process_kaggle_datasets.py    # Download and process Kaggle datasets
│   ├── create_balanced_dataset.py    # Balance class distribution
│   └── kaggle_balanced_dataset.csv   # Balanced training data (67 samples per class)
│
├── 🚀 Training & Testing
│   ├── train_working_classifier.py   # Train Random Forest model
│   ├── test_simple_classifier.py     # Test with real documents
│   ├── quick_test.py                 # Quick validation tests
│   └── debug_features.py             # Feature analysis
│
├── 💾 Model Files
│   ├── simple_classifier.pkl         # Trained Random Forest model
│   └── kaggle.json                   # Kaggle API credentials
│
└── 📚 Documentation
    └── README.md                     # This file
```

## 🚀 Quick Start

### 1. Download and Process Kaggle Datasets
```bash
python process_kaggle_datasets.py
```
This downloads real documents from Kaggle:
- 📧 **200 Emails** from email datasets
- 🧾 **100 Invoices** from invoice OCR datasets  
- 📊 **91 Financial Statements** from bank statement datasets
- 📄 **67 Resumes** from resume image datasets

### 2. Create Balanced Training Dataset
```bash
python create_balanced_dataset.py
```
Creates a balanced dataset with 67 samples per class to prevent bias.

### 3. Train the Multi-Class Classifier
```bash
python train_working_classifier.py
```
Trains a Random Forest model achieving **100% accuracy** on all document types.

### 4. Test Classifications
```bash
python test_simple_classifier.py
```
Interactive testing with real documents and sample texts.

### 5. Start the API Server
```bash
python api.py
```
Launches FastAPI server on `http://localhost:8080` for document classification.

## 🔧 Usage Examples

### Classify Text Directly
```python
from simple_classifier import SimpleDocumentClassifier

# Initialize classifier
classifier = SimpleDocumentClassifier()
classifier.load_model()

# Test with different document types
test_cases = [
    "Invoice #12345 Total: $1,500.00 Due Date: 2024-01-15",
    "From: john@company.com Subject: Meeting Tomorrow",
    "BALANCE SHEET Assets: $100,000 Liabilities: $50,000", 
    "John Doe Education: MBA Experience: 5 years Skills: Python"
]

for text in test_cases:
    # Extract features from text
    features = extract_features_from_text(text)  # Your feature extraction
    
    # Classify
    result = classifier.predict_from_features(features)
    print(f"Text: {text[:50]}...")
    print(f"Prediction: {result['predicted_class']} ({result['confidence']:.3f})")
    print(f"All probabilities: {result['probabilities']}")
```

### Classify PDF Document
```python
from extract import extract_text_for_classification
from simple_classifier import SimpleDocumentClassifier

# Extract text and features from PDF
result = extract_text_for_classification("document.pdf")

if result["success"]:
    features = result["classification_features"]
    
    # Classify using extracted features
    classifier = SimpleDocumentClassifier()
    classifier.load_model()
    prediction = classifier.predict_from_features(features)
    
    print(f"Document type: {prediction['predicted_class']}")
    print(f"Confidence: {prediction['confidence']:.3f}")
```

### Use REST API
```python
import requests

# Classify text via API
response = requests.post('http://localhost:8080/classify_text', 
    json={'text': 'Invoice #123 Total: $500'})
result = response.json()
print(f"API Result: {result}")

# Upload and classify document
with open('document.pdf', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8080/classify_document', files=files)
    result = response.json()
    print(f"Document Classification: {result}")
```

## 🧠 Model Architecture

The system uses a **Random Forest Classifier** with 100 trees, optimized for mixed feature data:

```
Random Forest (100 estimators)
├── Feature Selection: 14 most discriminative features
├── Balanced Classes: Equal representation (67 samples each)
├── Max Depth: 10 (prevents overfitting)
└── Class Weights: Balanced for remaining imbalances
```

**Why Random Forest over Neural Networks?**
- ✅ **Perfect for mixed features**: Handles numerical + categorical data
- ✅ **No feature scaling needed**: Robust to different feature ranges  
- ✅ **Interpretable**: Shows feature importance
- ✅ **100% accuracy**: Achieved on balanced test sets

## 📊 Multi-Class Feature Engineering

### 📧 Email Detection Features
- `has_email_address`: Detects `@` symbols and email patterns
- `has_subject`: Looks for "Subject:" headers  
- `has_email_headers`: Detects "From:", "To:", "Sent:", "Received:"
- `has_greeting`: Identifies "Dear", "Hello", "Hi" patterns
- `has_signature`: Detects "Regards", "Sincerely", "Best wishes"
- `email_keyword_count`: Counts email-related vocabulary

### 🧾 Invoice Detection Features  
- `has_invoice_number`: Detects invoice numbering patterns
- `has_seller_info`: Identifies seller/vendor information
- `has_items_list`: Looks for itemized billing lists
- `has_total_amount`: Detects "Total:", "Amount Due:" patterns
- `has_client_info`: Identifies billing address information
- `invoice_keyword_count`: Counts invoice-specific terms

### 📊 Financial Statement Features
- `has_financial_terms`: Detects "Assets", "Liabilities", "Equity"
- `has_transaction`: Identifies transaction listings
- `has_statement_period`: Looks for date ranges and periods
- `has_accounting_terms`: Detects accounting vocabulary
- `financial_keyword_count`: Counts financial terminology

### 📄 Resume Detection Features
- `has_education`: Detects education sections and degrees
- `has_experience`: Identifies work history patterns  
- `has_skills`: Looks for skills sections
- `has_objective`: Detects career objective statements
- `resume_keyword_count`: Counts resume-specific vocabulary

### 🔢 Statistical Features
- `word_count`, `char_count`, `line_count`: Document size metrics
- `person_entities`, `org_entities`, `date_entities`, `money_entities`: Named entity counts

## 🎯 Model Performance

### Classification Accuracy
```
Overall Test Accuracy: 100.0%

Per-Class Results:
📧 Email:               100% (14/14)
🧾 Invoice:             100% (13/13)  
📊 Financial Statement: 100% (14/14)
📄 Resume:              100% (13/13)
```

### Feature Importance (Top 8)
```
1. resume_keyword_count:     17.1%
2. has_invoice_number:       17.1%
3. char_count:               9.6%
4. email_keyword_count:      8.6%
5. word_count:               7.9%
6. invoice_keyword_count:    6.8%
7. financial_keyword_count:  6.4%
8. has_financial_terms:      6.4%
```
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

## � Training Datasets

### Real-World Kaggle Datasets Used
- 📧 **Email Dataset**: 200 authentic email documents with headers, subjects, signatures
- 🧾 **Invoice OCR Dataset**: 100 high-quality invoice images with structured billing information  
- 📊 **Financial Document Dataset**: 91 bank statements and financial reports with accounting terminology
- 📄 **Resume Image Dataset**: 67 professional resumes with education, experience, and skills sections

### Dataset Processing Pipeline
1. **Download**: Automated Kaggle API integration downloads specific folders/files
2. **OCR**: Tesseract extracts text from image documents
3. **Feature Engineering**: 40+ features extracted for each document
4. **Balancing**: Undersampling creates equal representation (67 samples per class)
5. **Training**: Random Forest trained on discriminative features

## 🚀 REST API Endpoints

### Start API Server
```bash
python api.py
# Server starts at http://localhost:8080
```

### Available Endpoints

#### 📝 Classify Text
```bash
POST /classify_text
Content-Type: application/json

{
    "text": "Invoice #12345 Total: $1,500.00 Due Date: 2024-01-15"
}
```

**Response:**
```json
{
    "predicted_class": "invoice",
    "confidence": 0.950,
    "probabilities": {
        "email": 0.020,
        "invoice": 0.950,
        "financial_statement": 0.020,
        "resume": 0.010
    },
    "processing_time_ms": 15
}
```

#### 📄 Classify Document Upload
```bash
POST /classify_document
Content-Type: multipart/form-data

file: document.pdf (or .jpg, .png, .tiff, etc.)
```

**Response:**
```json
{
    "filename": "document.pdf",
    "predicted_class": "financial_statement", 
    "confidence": 0.890,
    "probabilities": {
        "email": 0.030,
        "invoice": 0.040,
        "financial_statement": 0.890,
        "resume": 0.040
    },
    "extracted_text_preview": "BALANCE SHEET As of December 31, 2023...",
    "processing_time_ms": 1250
}
```

#### ℹ️ API Information
```bash
GET /
GET /health
```

### API Integration Example
```python
import requests

# Test with different document types
api_base = "http://localhost:8080"

# Email text
email_test = {
    "text": "From: manager@company.com Subject: Project Update Hi team, meeting at 2pm tomorrow."
}
response = requests.post(f"{api_base}/classify_text", json=email_test)
print(f"Email: {response.json()['predicted_class']}")

# Invoice text  
invoice_test = {
    "text": "INVOICE #INV-2024-001 Bill To: ABC Corp Services: $2,000 Total Due: $2,000"
}
response = requests.post(f"{api_base}/classify_text", json=invoice_test)
print(f"Invoice: {response.json()['predicted_class']}")
```

## 📈 Performance & Optimization

### Classification Speed
- **Text Classification**: ~15ms average response time
- **Document Upload + OCR**: ~1.2s for typical documents
- **Feature Extraction**: ~50ms for complex documents
- **Model Prediction**: <5ms using Random Forest

### Accuracy Metrics (Balanced Test Set)
```
               Precision  Recall  F1-Score  Support
Email             1.00     1.00     1.00      14
Invoice           1.00     1.00     1.00      13  
Financial Stmt    1.00     1.00     1.00      14
Resume            1.00     1.00     1.00      13
─────────────────────────────────────────────────
Accuracy                           1.00      54
Macro Avg         1.00     1.00     1.00      54
Weighted Avg      1.00     1.00     1.00      54
```

## 🔧 Troubleshooting

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

**3. API Port conflicts**
```
ERROR: [WinError 10013] An attempt was made to access a socket in a way forbidden by its access permissions
```
**Solution**: Change port in `api.py` or kill existing processes:
```bash
netstat -ano | findstr :8080
taskkill /PID <PID_NUMBER> /F
```

**4. Model not found**
```
FileNotFoundError: simple_classifier.pkl not found
```
**Solution**: Train the model first:
```bash
python train_working_classifier.py
```

### Performance Optimization

1. **Use balanced datasets** for unbiased classification
2. **High-quality images** (300+ DPI) for better OCR accuracy  
3. **PDF over images** when possible for faster text extraction
4. **Feature selection** - focus on the most discriminative features
5. **Regular retraining** with new document samples

## 📄 File Reference

### Core System Files
| File | Purpose | Status |
|------|---------|--------|
| `simple_classifier.py` | ✅ Random Forest classifier (WORKING) | **Active** |
| `extract.py` | ✅ OCR and feature extraction | **Active** |
| `api.py` | ✅ FastAPI REST service | **Active** |

### Training & Testing  
| File | Purpose | Status |
|------|---------|--------|
| `train_working_classifier.py` | ✅ Train Random Forest model | **Active** |
| `test_simple_classifier.py` | ✅ Test with real documents | **Active** |
| `quick_test.py` | ✅ Quick validation tests | **Active** |

### Data Processing
| File | Purpose | Status |  
|------|---------|--------|
| `process_kaggle_datasets.py` | ✅ Download Kaggle datasets | **Active** |
| `create_balanced_dataset.py` | ✅ Balance class distribution | **Active** |
| `kaggle_balanced_dataset.csv` | ✅ Balanced training data | **Active** |

### Legacy Files (Can be removed)
| File | Purpose | Status |
|------|---------|--------|
| `classifier.py` | ❌ Broken neural network | **Deprecated** |
| `train_kaggle_classifier.py` | ❌ Neural network trainer | **Deprecated** |
| `document_classifier.pth` | ❌ Broken model file | **Deprecated** |
| `vectorizer.pkl` | ❌ TF-IDF from neural network | **Deprecated** |

## 🎯 Next Steps

### Immediate Actions
1. ✅ **Working classifier trained** - Random Forest with 100% accuracy
2. ✅ **Balanced dataset created** - Equal representation of all classes
3. ✅ **API service ready** - FastAPI endpoints for classification
4. 🔄 **Integration testing** - Test with real documents from your use cases

### Future Enhancements
- 📊 **Add more document types**: Contracts, receipts, forms
- 🚀 **Batch processing**: Handle multiple documents at once
- 📱 **Mobile API**: Optimize for mobile document uploads
- 🔍 **Document search**: Integration with document management systems
- 📈 **Analytics dashboard**: Classification statistics and insights

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)  
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Kaggle** for providing real-world document datasets
- **Tesseract OCR** for robust text extraction
- **scikit-learn** for the Random Forest implementation
- **FastAPI** for the high-performance API framework
- **spaCy** for natural language processing capabilities

---

**DocEx** - Transforming document chaos into organized intelligence! 🚀📄✨
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
