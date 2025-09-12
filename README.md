# DocEx - Document Classification System

A PyTorch-based machine learning system for classifying scanned documents as either **emails** or **not emails** using OCR and neural networks.

## 🚀 Features

- **OCR Text Extraction**: Extracts text from images using Tesseract OCR with advanced preprocessing
- **Binary Classification**: PyTorch-based deep learning model for email detection
- **Feature Engineering**: 20+ engineered features optimized for email detection
- **Data Processing**: Automated dataset creation from Kaggle datasets with balanced sampling
- **Interactive Testing**: Command-line interface for testing classifications

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
pip install torch scikit-learn pandas numpy pytesseract Pillow spacy kagglehub pdfminer.six pdf2image
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

### Classify Image
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

### Advanced OCR Features
- **Image preprocessing**: Auto-resize, RGB conversion
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
2. **Better OCR**: Use high-quality, clear images with proper preprocessing
3. **Feature Engineering**: Focus on email-specific patterns in `extract_classification_features()`
4. **Hyperparameter Tuning**: Adjust learning rate, epochs, batch size for binary classification

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
| `extract.py` | Advanced OCR text extraction and email-focused feature engineering |
| `classifier.py` | PyTorch binary classification neural network |
| `train_classifier.py` | Complete training pipeline for email detection |
| `test_classifier.py` | Interactive testing interface for email classification |

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
