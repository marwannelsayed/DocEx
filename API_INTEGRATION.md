# DocEx API Integration Guide

## FastAPI Email Classification Service

This API provides email classification capabilities for your DocRepo document management system.

### 🚀 Quick Start

1. **Install API dependencies:**
```bash
pip install -r requirements_api.txt
```

2. **Make sure your model is trained:**
```bash
python train_classifier.py
```

3. **Start the API server:**
```bash
python api.py
```

4. **Test the API:**
```bash
python test_api.py
```

### 📡 API Endpoints

#### Health Check
```
GET /
```
Returns API status and classifier loading state.

#### Classify Text
```
POST /classify/text
Content-Type: application/json

{
    "text": "Subject: Meeting Tomorrow From: john@company.com..."
}
```

#### Classify Document Image
```
POST /classify/document
Content-Type: multipart/form-data

file: <image_file>
```

#### Model Information
```
GET /model/info
```
Returns information about the loaded classification model.

### 🔗 Integration with DocRepo

To integrate with your DocRepo system, add these API calls:

#### 1. JavaScript/Frontend Integration

```javascript
// Classify uploaded document
async function classifyDocument(file) {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch('http://localhost:8000/classify/document', {
        method: 'POST',
        body: formData
    });
    
    const result = await response.json();
    return result;
}

// Classify text content
async function classifyText(text) {
    const response = await fetch('http://localhost:8000/classify/text', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: text })
    });
    
    const result = await response.json();
    return result;
}
```

#### 2. Python Backend Integration

```python
import requests

def classify_document_for_docrepo(file_path):
    """Classify a document file"""
    with open(file_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(
            'http://localhost:8000/classify/document',
            files=files
        )
    return response.json()

def classify_text_for_docrepo(text):
    """Classify text content"""
    response = requests.post(
        'http://localhost:8000/classify/text',
        json={'text': text}
    )
    return response.json()
```

### 🏗️ Architecture

```
DocRepo Frontend
       ↓
DocRepo Backend 
       ↓ (API calls)
DocEx Classification API
       ↓
PyTorch Model + OCR
```

### 📊 Response Format

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

### 🔧 Configuration

The API runs on `http://localhost:8000` by default. You can modify the host and port in `api.py`:

```python
uvicorn.run(
    "api:app",
    host="0.0.0.0",  # Change host
    port=8000,       # Change port
    reload=True
)
```

### 🛡️ Production Deployment

For production deployment with DocRepo:

1. **Use environment variables for configuration**
2. **Configure CORS for your specific frontend URL**
3. **Add authentication if needed**
4. **Use a production ASGI server like Gunicorn**
5. **Set up proper logging and monitoring**

### 📝 Integration Steps for DocRepo

1. **Clone and set up DocEx API** (this repo)
2. **Modify your DocRepo upload handler** to call the classification API
3. **Add email classification tags** to your document metadata
4. **Update your frontend** to display classification results
5. **Add filtering/search** by document type (email/not email)

### 🔍 Testing

Use the interactive API documentation at `http://localhost:8000/docs` to test all endpoints.
