import kagglehub
import os
import pandas as pd
from extract import extract_text_for_classification

# Download datasets
path1 = kagglehub.dataset_download("suvroo/scanned-images-dataset-for-ocr-and-vlm-finetuning")
print("Path to dataset files (emails):", path1)

path2 = kagglehub.dataset_download("osamahosamabdellatif/high-quality-invoice-images-for-ocr")
print("Path to dataset files (invoices):", path2)

data = []

# Process emails
print("Processing emails...")
email_count = 0
for root, dirs, files in os.walk(path1):
    if os.path.basename(root) == "Email":  
        for file in files:
            file_path = os.path.join(root, file)
            try:
                # Extract structured data for classification
                result = extract_text_for_classification(file_path)
                if result["success"] and result["cleaned_text"].strip():
                    data.append({
                        "file": file,
                        "text": result["cleaned_text"],
                        "raw_text": result["raw_text"],
                        "label": "email",
                        **result["classification_features"]  # Add all features as columns
                    })
                    email_count += 1
                    if email_count % 10 == 0:
                        print(f"Processed {email_count} emails...")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

# Process invoices
print("Processing invoices...")
invoice_count = 0
folder_path2 = os.path.join(path2, "batch_1", "batch_1", "batch1_1")
for file in os.listdir(folder_path2):
    file_path = os.path.join(folder_path2, file)
    if os.path.isfile(file_path):
        try:
            # Extract structured data for classification
            result = extract_text_for_classification(file_path)
            if result["success"] and result["cleaned_text"].strip():
                data.append({
                    "file": file,
                    "text": result["cleaned_text"],
                    "raw_text": result["raw_text"],
                    "label": "invoice",
                    **result["classification_features"]  # Add all features as columns
                })
                invoice_count += 1
                if invoice_count % 10 == 0:
                    print(f"Processed {invoice_count} invoices...")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

# Save training dataset with features
df = pd.DataFrame(data)
df.to_csv("training_dataset.csv", index=False)

print(f"\nDataset created with {len(df)} samples:")
print(f"- Emails: {len(df[df['label'] == 'email'])}")
print(f"- Invoices: {len(df[df['label'] == 'invoice'])}")
print("Saved as training_dataset.csv")

# Show feature statistics
print(f"\nDataset contains {len(df.columns)} features:")
feature_cols = [col for col in df.columns if col not in ['file', 'text', 'raw_text', 'label']]
print("Features:", feature_cols)

# Show feature distributions by class
print("\nFeature distributions:")
for feature in ['has_invoice_number', 'has_amount', 'has_email_address', 'has_subject']:
    if feature in df.columns:
        print(f"{feature}:")
        print(df.groupby('label')[feature].mean())
        print()

# Show sample texts
print("Sample email text:")
email_sample = df[df['label'] == 'email'].iloc[0]['text']
print(email_sample[:200] + "..." if len(email_sample) > 200 else email_sample)

print("\nSample invoice text:")
invoice_sample = df[df['label'] == 'invoice'].iloc[0]['text']
print(invoice_sample[:200] + "..." if len(invoice_sample) > 200 else invoice_sample)
