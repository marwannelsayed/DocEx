from classifier import DocumentClassifierTrainer
from extract import extract_text_for_classification
import os

def classify_image(image_path):
    """Classify a single image file."""
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return None
    
    # Extract text from image
    text = extract_text_for_classification(image_path)
    
    if not text.strip():
        print("No text extracted from image")
        return None
    
    # Load trained classifier
    trainer = DocumentClassifierTrainer()
    
    try:
        result = trainer.predict(text)
        return result
    except FileNotFoundError:
        print("Trained model not found. Please run train_classifier.py first.")
        return None
    except Exception as e:
        print(f"Error during classification: {e}")
        return None

def main():
    print("=== Document Classification Inference ===\n")
    
    # Test with sample text
    sample_texts = [
        "Invoice #12345 Total: $150.00 Due Date: 2024-01-15",
        "Subject: Meeting Tomorrow From: john@company.com To: team@company.com"
    ]
    
    trainer = DocumentClassifierTrainer()
    
    for i, text in enumerate(sample_texts, 1):
        print(f"Sample {i}: {text}")
        try:
            result = trainer.predict(text)
            print(f"Prediction: {result['predicted_class']} (confidence: {result['confidence']:.3f})")
            print(f"Probabilities: Email: {result['probabilities']['email']:.3f}, Not Email: {result['probabilities']['not email']:.3f}")
        except FileNotFoundError:
            print("Trained model not found. Please run train_classifier.py first.")
            break
        except Exception as e:
            print(f"Error: {e}")
        print("-" * 50)
    
    # Interactive mode
    print("\nInteractive Mode (type 'quit' to exit):")
    while True:
        user_input = input("\nEnter image path or text to classify: ").strip()
        if user_input.lower() == 'quit':
            break
        if os.path.exists(user_input):
            # It's a file path
            result = classify_image(user_input)
            if result:
                print(f"File: {user_input}")
                print(f"Prediction: {result['predicted_class']} (confidence: {result['confidence']:.3f})")
                print(f"Probabilities: Email: {result['probabilities']['email']:.3f}, Not Email: {result['probabilities']['not email']:.3f}")
        else:
            # It's text
            try:
                result = trainer.predict(user_input)
                print(f"Text: {user_input[:100]}...")
                print(f"Prediction: {result['predicted_class']} (confidence: {result['confidence']:.3f})")
                print(f"Probabilities: Email: {result['probabilities']['email']:.3f}, Not Email: {result['probabilities']['not email']:.3f}")
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    main()
