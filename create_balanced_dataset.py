"""
Create Balanced Multi-Class Dataset
==================================

Fixes the class imbalance issue by undersampling the email class to match other classes.
This should fix the "everything is email" prediction problem.
"""

import pandas as pd
import numpy as np
from sklearn.utils import resample

def create_balanced_dataset():
    """Create a balanced dataset by undersampling the majority class (emails)"""
    
    print("🚀 Creating Balanced Multi-Class Dataset")
    print("=" * 50)
    
    # Load original Kaggle dataset
    df = pd.read_csv("kaggle_multiclass_dataset.csv")
    
    print("📊 Original dataset distribution:")
    original_counts = df['label'].value_counts().sort_index()
    class_names = {0: 'Email', 1: 'Invoice', 2: 'Financial Statement', 3: 'Resume'}
    
    for label, count in original_counts.items():
        print(f"   {class_names[label]} (class {label}): {count} samples")
    
    # Find the minimum class size (excluding the majority class)
    min_class_size = min([count for label, count in original_counts.items() if label != 0])
    print(f"\n🎯 Target size per class: {min_class_size} samples")
    
    # Balance the dataset
    balanced_dfs = []
    
    for class_id in [0, 1, 2, 3]:
        class_df = df[df['label'] == class_id]
        
        if len(class_df) > min_class_size:
            # Undersample
            balanced_class_df = resample(
                class_df,
                n_samples=min_class_size,
                random_state=42,
                replace=False
            )
            print(f"   📉 {class_names[class_id]}: {len(class_df)} → {len(balanced_class_df)} (undersampled)")
        else:
            balanced_class_df = class_df
            print(f"   ✅ {class_names[class_id]}: {len(class_df)} (unchanged)")
            
        balanced_dfs.append(balanced_class_df)
    
    # Combine balanced classes
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    
    # Shuffle the balanced dataset
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\n📈 Balanced dataset distribution:")
    balanced_counts = balanced_df['label'].value_counts().sort_index()
    for label, count in balanced_counts.items():
        print(f"   {class_names[label]} (class {label}): {count} samples")
    
    print(f"\n💾 Total samples: {len(balanced_df)} (was {len(df)})")
    
    # Save balanced dataset
    output_file = "kaggle_balanced_dataset.csv"
    balanced_df.to_csv(output_file, index=False)
    print(f"✅ Balanced dataset saved as: {output_file}")
    
    # Show some samples from each class to verify
    print(f"\n🔍 Sample texts from each class:")
    for class_id in [0, 1, 2, 3]:
        sample = balanced_df[balanced_df['label'] == class_id].iloc[0]
        text_preview = sample['text'][:80] + "..." if len(sample['text']) > 80 else sample['text']
        print(f"   {class_names[class_id]}: {text_preview}")

if __name__ == "__main__":
    create_balanced_dataset()