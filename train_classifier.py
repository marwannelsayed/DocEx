#!/usr/bin/env python3

import subprocess
import sys
import os

def main():
    print("=== Document Classification Training Pipeline ===\n")

    # Step 1: Create training dataset
    print("Step 1: Checking for training dataset...")
    
    if os.path.exists("training_dataset.csv"):
        print("✓ Training dataset already exists (training_dataset.csv)")
        print("Skipping dataset creation step.")
    else:
        print("Training dataset not found. Creating new dataset...")
        try:
            subprocess.run([sys.executable, 'app.py'], check=True)
            print("✓ Training dataset created successfully")
        except subprocess.CalledProcessError:
            print("✗ Failed to create training dataset")
            return
        except FileNotFoundError:
            print("✗ app.py not found")
            return
        
        # Check if dataset was created
        if not os.path.exists("training_dataset.csv"):
            print("✗ Training dataset file not found")
            return
    
    print("\n" + "="*50)

    # Step 2: Train classifier
    print("Step 2: Training PyTorch classifier...")
    try:
        from classifier import DocumentClassifierTrainer
        
        trainer = DocumentClassifierTrainer()
        model, losses, accuracies = trainer.train("training_dataset.csv", epochs=50)
        
        print("✓ Model training completed successfully")
        print(f"✓ Model saved as: {trainer.model_save_path}")
        print(f"✓ Vectorizer saved as: {trainer.vectorizer_save_path}")
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        print("Make sure all required packages are installed")
        return
    except Exception as e:
        print(f"✗ Training failed: {e}")
        return
    
    print("\n" + "="*50)
    print("Training pipeline completed successfully!")
    print("\nYou can now use the trained model to classify documents:")
    print("```python")
    print("from classifier import DocumentClassifierTrainer")
    print("trainer = DocumentClassifierTrainer()")
    print("result = trainer.predict('your text here')")
    print("print(result)")
    print("```")

if __name__ == "__main__":
    main()
