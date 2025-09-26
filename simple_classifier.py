"""
Simple Rule-Based + ML Classifier
================================

Uses the discriminative features we found to create a more reliable classifier.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os

class SimpleDocumentClassifier:
    def __init__(self):
        self.model = None
        self.feature_cols = None
        self.class_names = {0: 'email', 1: 'invoice', 2: 'financial_statement', 3: 'resume'}
        self.model_path = "simple_classifier.pkl"
    
    def extract_key_features(self, data):
        """Extract the most discriminative features"""
        
        # Key discriminative features based on our analysis
        key_features = [
            'has_invoice_number', 'has_email_address', 'has_education', 'has_financial_terms',
            'email_keyword_count', 'invoice_keyword_count', 'financial_keyword_count', 'resume_keyword_count',
            'money_entities', 'person_entities', 'org_entities', 'date_entities',
            'word_count', 'char_count'
        ]
        
        # Only use features that exist in the data
        available_features = [f for f in key_features if f in data.columns]
        print(f"Using {len(available_features)} discriminative features: {available_features}")
        
        return data[available_features]
    
    def train(self, dataset_file="kaggle_balanced_dataset.csv"):
        """Train the simple classifier"""
        
        print("🚀 Training Simple Document Classifier")
        print("=" * 50)
        
        # Load data
        df = pd.read_csv(dataset_file)
        
        # Extract features and labels
        X = self.extract_key_features(df)
        y = df['label']
        
        self.feature_cols = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        
        # Train Random Forest (better for mixed features)
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'  # Handle any remaining imbalance
        )
        
        self.model.fit(X_train, y_train)
        
        # Test performance
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\\n✅ Training completed!")
        print(f"📊 Test Accuracy: {accuracy:.3f}")
        
        # Show classification report
        class_labels = [self.class_names[i] for i in range(4)]
        print(f"\\n📈 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=class_labels, zero_division=0))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\\n🔝 Top Feature Importances:")
        for _, row in feature_importance.head(8).iterrows():
            print(f"   {row['feature']}: {row['importance']:.3f}")
        
        # Save model
        with open(self.model_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_cols': self.feature_cols,
                'class_names': self.class_names
            }, f)
        
        print(f"\\n💾 Model saved as: {self.model_path}")
        
        return self.model
    
    def load_model(self):
        """Load trained model"""
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
                self.model = data['model']
                self.feature_cols = data['feature_cols']
                self.class_names = data['class_names']
            return True
        return False
    
    def predict_from_features(self, features_dict):
        """Predict from feature dictionary"""
        if self.model is None:
            if not self.load_model():
                raise ValueError("No trained model found. Please train first.")
        
        # Create feature vector
        feature_vector = np.zeros(len(self.feature_cols))
        for i, feature in enumerate(self.feature_cols):
            feature_vector[i] = features_dict.get(feature, 0)
        
        # Predict
        prediction = self.model.predict([feature_vector])[0]
        probabilities = self.model.predict_proba([feature_vector])[0]
        
        # Create probability dict
        prob_dict = {self.class_names[i]: prob for i, prob in enumerate(probabilities)}
        
        return {
            'predicted_label': int(prediction),
            'predicted_class': self.class_names[prediction],
            'confidence': max(probabilities),
            'probabilities': prob_dict
        }

def main():
    """Train and test the simple classifier"""
    
    classifier = SimpleDocumentClassifier()
    classifier.train()
    
    # Quick test
    print(f"\\n🧪 Quick Test:")
    print("-" * 30)
    
    # Test with some feature combinations that should be distinctive
    test_cases = [
        ({'has_invoice_number': 1, 'invoice_keyword_count': 2, 'money_entities': 1}, 'invoice'),
        ({'has_email_address': 1, 'email_keyword_count': 3, 'person_entities': 2}, 'email'),
        ({'has_financial_terms': 1, 'financial_keyword_count': 2, 'money_entities': 2}, 'financial_statement'),
        ({'has_education': 1, 'has_experience': 1, 'resume_keyword_count': 3}, 'resume')
    ]
    
    for features, expected in test_cases:
        result = classifier.predict_from_features(features)
        predicted = result['predicted_class']
        confidence = result['confidence']
        
        status = "✅" if predicted == expected else "❌"
        print(f"{status} Expected: {expected} | Predicted: {predicted} ({confidence:.3f})")

if __name__ == "__main__":
    main()