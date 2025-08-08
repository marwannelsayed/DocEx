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
        print(f"Found Email folder: {root}")
        print(f"Files in folder: {len(files)}")
        for file in files:
            file_path = os.path.join(root, file)
            try:
                # Extract structured data for classification
                result = extract_text_for_classification(file_path)
                if result["success"] and result["cleaned_text"].strip():
                    data.append({
                        "file": file,
                        "text": result["cleaned_text"],
                        "label": "email",
                        **result["classification_features"]  # Add all features as columns
                    })
                    email_count += 1
                    if email_count % 10 == 0:
                        print(f"Processed {email_count} emails...")
                else:
                    print(f"Failed to extract text from {file}: {result.get('error', 'Unknown error')}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                import traceback
                traceback.print_exc()

print(f"Total emails processed: {email_count}")

# Process invoices
print("Processing invoices...")
invoice_count = 0
folder_path2 = os.path.join(path2, "batch_1", "batch_1", "batch1_1")
print(f"Looking for invoices in: {folder_path2}")

if not os.path.exists(folder_path2):
    print(f"Invoice folder not found: {folder_path2}")
    print("Available paths:")
    for root, dirs, files in os.walk(path2):
        print(f"  {root}")
else:
    files_list = os.listdir(folder_path2)
    print(f"Files in invoice folder: {len(files_list)}")
    
    for file in files_list:
        file_path = os.path.join(folder_path2, file)
        if os.path.isfile(file_path):
            try:
                # Extract structured data for classification
                result = extract_text_for_classification(file_path)
                if result["success"] and result["cleaned_text"].strip():
                    data.append({
                        "file": file,
                        "text": result["cleaned_text"],
                        "label": "invoice",
                        **result["classification_features"]  # Add all features as columns
                    })
                    invoice_count += 1
                    if invoice_count % 10 == 0:
                        print(f"Processed {invoice_count} invoices...")
                else:
                    print(f"Failed to extract text from {file}: {result.get('error', 'Unknown error')}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                import traceback
                traceback.print_exc()

print(f"Total invoices processed: {invoice_count}")

# Save training dataset with features
df = pd.DataFrame(data)

# Check if we have any data
if len(data) == 0:
    print("ERROR: No data was processed successfully!")
    print("Possible issues:")
    print("1. spaCy model 'en_core_web_sm' not installed - run: python -m spacy download en_core_web_sm")
    print("2. Image files not found in expected directories")
    print("3. OCR issues with pytesseract")
    print("Please check the error messages above and fix the issues.")
    exit(1)

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
if len(df[df['label'] == 'email']) > 0:
    print("Sample email text:")
    email_sample = df[df['label'] == 'email'].iloc[0]['text']
    print(email_sample[:200] + "..." if len(email_sample) > 200 else email_sample)

if len(df[df['label'] == 'invoice']) > 0:
    print("\nSample invoice text:")
    invoice_sample = df[df['label'] == 'invoice'].iloc[0]['text']
    print(invoice_sample[:200] + "..." if len(invoice_sample) > 200 else invoice_sample)
