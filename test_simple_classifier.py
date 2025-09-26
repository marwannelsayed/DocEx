"""
Test the working simple classifier with real text input
"""

from simple_classifier import SimpleDocumentClassifier
import pandas as pd

def test_with_real_text():
    """Test the simple classifier with actual text from the dataset"""
    
    print("🧪 Testing Simple Classifier with Real Text")
    print("=" * 60)
    
    # Load the dataset to get feature data
    df = pd.read_csv("kaggle_balanced_dataset.csv")
    classifier = SimpleDocumentClassifier()
    
    if not classifier.load_model():
        print("❌ No trained model found. Please run simple_classifier.py first.")
        return
    
    class_names = {0: 'Email', 1: 'Invoice', 2: 'Financial Statement', 3: 'Resume'}
    
    # Test with a few samples from each class
    for class_id in range(4):
        class_df = df[df['label'] == class_id]
        sample = class_df.iloc[0]  # First sample
        
        print(f"\n📄 Testing {class_names[class_id]}:")
        print("-" * 40)
        print(f"Text: {sample['text'][:100]}...")
        
        # Extract features for this sample
        features = {}
        for col in classifier.feature_cols:
            if col in sample:
                features[col] = sample[col]
        
        # Predict
        result = classifier.predict_from_features(features)
        predicted = result['predicted_class'].replace('_', ' ').title()
        confidence = result['confidence']
        
        status = "✅" if result['predicted_label'] == class_id else "❌"
        
        print(f"{status} Expected: {class_names[class_id]} | Predicted: {predicted} ({confidence:.3f})")
        
        # Show key features that influenced the decision
        print("Key features:")
        key_features = ['has_invoice_number', 'has_email_address', 'has_education', 'has_financial_terms']
        for feature in key_features:
            if feature in features and features[feature] > 0:
                print(f"   ✅ {feature}: {features[feature]}")
        
        # Show keyword counts
        keyword_features = [f for f in features.keys() if 'keyword_count' in f and features[f] > 0]
        if keyword_features:
            print("Keyword counts:")
            for feature in keyword_features:
                print(f"   📊 {feature}: {features[feature]}")

def test_problematic_case():
    """Test the specific case that was failing"""
    
    print("\n🔍 Testing Previously Problematic Case")
    print("=" * 50)
    
    # Load the dataset to find an invoice sample
    df = pd.read_csv("kaggle_balanced_dataset.csv")
    invoice_samples = df[df['label'] == 1]
    
    classifier = SimpleDocumentClassifier()
    classifier.load_model()
    
    for i in range(min(3, len(invoice_samples))):
        sample = invoice_samples.iloc[i]
        text = sample['text']
        
        print(f"\n📄 Invoice Sample {i+1}:")
        print(f"Text: {text[:80]}...")
        
        # Extract features
        features = {}
        for col in classifier.feature_cols:
            if col in sample:
                features[col] = sample[col]
        
        # Predict
        result = classifier.predict_from_features(features)
        
        print(f"✅ Predicted: {result['predicted_class'].replace('_', ' ').title()} ({result['confidence']:.3f})")
        
        # Check if it has the word "invoice"
        has_invoice_word = "invoice" in text.lower()
        has_invoice_feature = features.get('has_invoice_number', 0) > 0
        invoice_keywords = features.get('invoice_keyword_count', 0)
        
        print(f"   Word 'invoice' in text: {'✅' if has_invoice_word else '❌'}")
        print(f"   has_invoice_number: {'✅' if has_invoice_feature else '❌'}")
        print(f"   invoice_keyword_count: {invoice_keywords}")

if __name__ == "__main__":
    test_with_real_text()
    test_problematic_case()