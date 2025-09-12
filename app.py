import kagglehub
import os
import pandas as pd
from extract import extract_text_for_classification

# Download datasets
path = kagglehub.dataset_download("suvroo/scanned-images-dataset-for-ocr-and-vlm-finetuning")
print("Path to dataset files", path)

data = []

# Process emails
print("Processing emails...")
email_count = 0
for root, dirs, files in os.walk(path):
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

# Process non-email documents (label as "not email")
print("Processing non-email documents...")
total_non_email_count = 0
MAX_FILES_PER_FOLDER = 100

for root, dirs, files in os.walk(path):
    folder_name = os.path.basename(root)
    # Process any folder that's not Email and not the root dataset folder
    if folder_name != "Email" and folder_name != "dataset" and folder_name != os.path.basename(path) and files:
        print(f"Found non-email folder: {folder_name} ({root})")
        print(f"Files in folder: {len(files)}")
        
        folder_file_count = 0
        for file in files:
            # Stop processing this folder if we've reached the limit for this folder
            if folder_file_count >= MAX_FILES_PER_FOLDER:
                print(f"Reached limit of {MAX_FILES_PER_FOLDER} files for folder {folder_name}. Moving to next folder.")
                break
                
            file_path = os.path.join(root, file)
            try:
                # Extract structured data for classification
                result = extract_text_for_classification(file_path)
                if result["success"] and result["cleaned_text"].strip():
                    data.append({
                        "file": file,
                        "text": result["cleaned_text"],
                        "label": "not email",
                        **result["classification_features"]  # Add all features as columns
                    })
                    folder_file_count += 1
                    total_non_email_count += 1
                    if total_non_email_count % 10 == 0:
                        print(f"Processed {total_non_email_count} total non-email documents...")
                else:
                    print(f"Failed to extract text from {file}: {result.get('error', 'Unknown error')}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"Processed {folder_file_count} files from folder {folder_name}")

print(f"Total non-email documents processed: {total_non_email_count}")

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
print(f"- Not emails: {len(df[df['label'] == 'not email'])}")
print("Saved as training_dataset.csv")

# Show feature statistics
print(f"\nDataset contains {len(df.columns)} features:")
feature_cols = [col for col in df.columns if col not in ['file', 'text', 'raw_text', 'label']]
print("Features:", feature_cols)

# Show feature distributions by class
print("\nFeature distributions:")
for feature in ['has_email_address', 'has_subject']:
    if feature in df.columns:
        print(f"{feature}:")
        print(df.groupby('label')[feature].mean())
        print()

# Show sample texts
if len(df[df['label'] == 'email']) > 0:
    print("Sample email text:")
    email_sample = df[df['label'] == 'email'].iloc[0]['text']
    print(email_sample[:200] + "..." if len(email_sample) > 200 else email_sample)
    print()

if len(df[df['label'] == 'not email']) > 0:
    print("Sample non-email text:")
    non_email_sample = df[df['label'] == 'not email'].iloc[0]['text']
    print(non_email_sample[:200] + "..." if len(non_email_sample) > 200 else non_email_sample)
