import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np
import pickle
import os

class DocumentDataset(Dataset):
    def __init__(self, texts, labels, vectorizer=None, is_training=True):
        self.texts = texts
        self.labels = labels
        
        if is_training:
            # Fit vectorizer on training data
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2),  # Include bigrams
                min_df=2,
                max_df=0.8
            )
            self.features = self.vectorizer.fit_transform(texts).toarray()
        else:
            # Use existing vectorizer for test data
            self.vectorizer = vectorizer
            self.features = self.vectorizer.transform(texts).toarray()
            
        self.features = torch.FloatTensor(self.features)
        self.labels = torch.LongTensor([1 if label == 'invoice' else 0 for label in labels])
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class DocumentClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=256, dropout_rate=0.3):
        super(DocumentClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 2)  # 2 classes: email (0), invoice (1)
        )
    
    def forward(self, x):
        return self.classifier(x)

class DocumentClassifierTrainer:
    def __init__(self, model_save_path="document_classifier.pth", vectorizer_save_path="vectorizer.pkl"):
        self.model_save_path = model_save_path
        self.vectorizer_save_path = vectorizer_save_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
    def prepare_data(self, csv_path):
        df = pd.read_csv(csv_path)
        
        # Get texts and labels
        texts = df['text'].astype(str).tolist()
        labels = df['label'].tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Training distribution: {pd.Series(y_train).value_counts()}")
        
        return X_train, X_test, y_train, y_test
    
    def train(self, csv_path, epochs=50, batch_size=32, learning_rate=0.001):
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(csv_path)
        
        # Create datasets
        train_dataset = DocumentDataset(X_train, y_train, is_training=True)
        test_dataset = DocumentDataset(X_test, y_test, train_dataset.vectorizer, is_training=False)
        
        # Save vectorizer
        with open(self.vectorizer_save_path, 'wb') as f:
            pickle.dump(train_dataset.vectorizer, f)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        input_size = train_dataset.features.shape[1]
        model = DocumentClassifier(input_size).to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.7)
        
        # Training loop
        best_accuracy = 0
        train_losses = []
        test_accuracies = []
        
        for epoch in range(epochs):
            # Training
            model.train()
            total_loss = 0
            for batch_features, batch_labels in train_loader:
                batch_features, batch_labels = batch_features.to(self.device), batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            train_losses.append(avg_loss)
            
            # Evaluation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_features, batch_labels in test_loader:
                    batch_features, batch_labels = batch_features.to(self.device), batch_labels.to(self.device)
                    outputs = model(batch_features)
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()
            
            accuracy = correct / total
            test_accuracies.append(accuracy)
            
            # Save best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'input_size': input_size,
                    'accuracy': accuracy,
                    'epoch': epoch
                }, self.model_save_path)
            
            scheduler.step()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}')
        
        print(f"Training completed! Best accuracy: {best_accuracy:.4f}")
        
        # Final evaluation
        self.evaluate(X_test, y_test)
        
        return model, train_losses, test_accuracies
    
    def evaluate(self, X_test, y_test):
        # Load best model
        checkpoint = torch.load(self.model_save_path, map_location=self.device)
        model = DocumentClassifier(checkpoint['input_size']).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Load vectorizer
        with open(self.vectorizer_save_path, 'rb') as f:
            vectorizer = pickle.load(f)
        
        # Prepare test data
        test_dataset = DocumentDataset(X_test, y_test, vectorizer, is_training=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Get predictions
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch_features, batch_labels in test_loader:
                batch_features = batch_features.to(self.device)
                outputs = model(batch_features)
                _, predicted = torch.max(outputs.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(batch_labels.numpy())
        
        # Convert back to string labels
        label_map = {0: 'email', 1: 'invoice'}
        pred_labels = [label_map[p] for p in all_predictions]
        true_labels = [label_map[l] for l in all_labels]
        
        # Print results
        accuracy = accuracy_score(true_labels, pred_labels)
        print(f"\nFinal Test Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(true_labels, pred_labels))
        print("\nConfusion Matrix:")
        print(confusion_matrix(true_labels, pred_labels))
        
        return accuracy, pred_labels, true_labels
    
    def predict(self, text):
        # Load model and vectorizer
        checkpoint = torch.load(self.model_save_path, map_location=self.device)
        model = DocumentClassifier(checkpoint['input_size']).to(self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        with open(self.vectorizer_save_path, 'rb') as f:
            vectorizer = pickle.load(f)
        
        # Preprocess text
        features = vectorizer.transform([text]).toarray()
        features = torch.FloatTensor(features).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = model(features)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
        
        label_map = {0: 'email', 1: 'invoice'}
        predicted_class = label_map[predicted.item()]
        confidence = probabilities[0][predicted.item()].item()
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': {
                'email': probabilities[0][0].item(),
                'invoice': probabilities[0][1].item()
            }
        }

if __name__ == "__main__":
    # Example usage
    trainer = DocumentClassifierTrainer()
    
    # Train the model (assuming you have training_dataset.csv)
    if os.path.exists("training_dataset.csv"):
        print("Training document classifier...")
        model, losses, accuracies = trainer.train("training_dataset.csv", epochs=50)
        print("Training completed!")
    else:
        print("Please run app.py first to create training_dataset.csv")
