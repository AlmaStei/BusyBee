# script to classify images using the TreeOfLifeClassifier and organize them by family
# store results in a csv file with target families from gbif_families.txt
import os
import csv
import re
import time
from datetime import timedelta
from bioclip import TreeOfLifeClassifier, Rank
from tqdm import tqdm
import pandas as pd

# Start timing
start_time = time.time()

# === PATH CONFIGURATION ===
# Input folder containing images to classify
# CHANGE THIS to the path where your images are located
main_folder = "C:/Users/Almas/Desktop/Alma/seppi-cam38"

# Output folder where results CSV will be saved

# You can keep this as is or change to your preferred location
output_folder = "C:/Users/Almas/bioclip_test/BioClip_testrun"
os.makedirs(output_folder, exist_ok=True)

# CSV filename - customize this for each camera or batch
# CHANGE THIS to a descriptive name for your dataset
csv_filename = "classifications_cam38.csv"

# Path to the YOLO classification results
yolo_results_path = "C:/Users/Almas/YOLOv5/yolov5-master/runs/predict-cls/seppi-cam38/results/classification_results.csv"

# Full path where the output CSV will be saved
csv_path = os.path.join(output_folder, csv_filename)

# Path to the text file containing target family names
# This file should contain one family name per line
target_families_path = "C:/Users/Almas/bioclip_test/MadHornet/gbif_families.txt"

# Step 1: Initialize the classifier
classifier = TreeOfLifeClassifier()

# Get valid families from BioClip
label_data = classifier.get_label_data()
valid_families = set(label_data[Rank.FAMILY.get_label()])

# Load target families
with open(target_families_path, 'r') as f:
    target_families = [line.strip() for line in f.readlines() if line.strip()]
    excluded_families = [f for f in target_families if f not in valid_families]
    valid_target_families = [f for f in target_families if f in valid_families]
    
    print(f"\nFamily validation report:")
    print(f"Total families in input list: {len(target_families)}")
    print(f"Valid families: {len(valid_target_families)}")
    print(f"Excluded families: {len(excluded_families)}")
    
    if excluded_families:
        print("\nThe following families were excluded:")
        for family in sorted(excluded_families):
            print(f"- {family}")
    
    print(f"\nProceeding with {len(valid_target_families)} valid families")

    try:
        family_filter = classifier.create_taxa_filter(Rank.FAMILY, valid_target_families)
        print(f"Successfully created filter for {len(valid_target_families)} families")
    except ValueError as e:
        print(f"Error creating filter: {e}")
        print("Proceeding without filter...")
        family_filter = None

# Check if CSV already exists and load processed images
processed_images = set()
if os.path.exists(csv_path):
    try:
        previous_data = pd.read_csv(csv_path)
        processed_images = set(previous_data['Image_Path'])
        print(f"Found existing CSV with {len(processed_images)} processed images")
        
        # Backup the existing file just in case
        import shutil
        backup_path = csv_path + '.backup'
        shutil.copy2(csv_path, backup_path)
        print(f"Created backup of existing CSV: {backup_path}")
    except Exception as e:
        print(f"Error reading existing CSV: {e}")
        print("Will start processing from beginning")

# Load YOLO classification results
yolo_results = {}
if os.path.exists(yolo_results_path):
    try:
        yolo_df = pd.read_csv(yolo_results_path)
        print(f"Loaded YOLO classification results: {len(yolo_df)} entries")
        
        for _, row in yolo_df.iterrows():
            img_name = row['img_name']
            yolo_results[img_name] = row.to_dict()
    except Exception as e:
        print(f"Error loading YOLO results: {e}")
else:
    print(f"YOLO results file not found: {yolo_results_path}")

# Step 2: Classify images and save results to CSV
image_count = len(processed_images)
remaining_count = 0
remaining_images = []

for root, _, files in os.walk(main_folder):
    for image_file in files:
        if not image_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
            
        image_path = os.path.join(root, image_file)
        if image_path not in processed_images:
            remaining_images.append(image_path)
            remaining_count += 1

print(f"Total images: {len(remaining_images) + image_count}")
print(f"Already processed: {image_count}")
print(f"Remaining to process: {remaining_count}")

# Open CSV file in append mode if it exists, otherwise create new
file_mode = 'a' if os.path.exists(csv_path) and processed_images else 'w'
with open(csv_path, file_mode, newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    # Write header only if creating a new file
    if file_mode == 'w':
        writer.writerow(['img_name', 'top1', 'top1_prob', 'Family_BioClip', 'Family_Confidence_BioClip', 'Classification_Category_BioClip'])

    # Create progress bar for remaining images
    with tqdm(total=remaining_count, desc="Processing remaining images", unit="img") as pbar:
        for image_path in remaining_images:
            img_start_time = time.time()
            img_filename = os.path.basename(image_path)
            
            # Get family prediction
            family_predictions = classifier.predict(image_path, Rank.FAMILY)
            
            family_name = ''
            family_score = 0
            classification_category = 'uncertain'
            
            if family_predictions:
                best_family_prediction = max(family_predictions, key=lambda x: x["score"])
                family_name = best_family_prediction["family"].replace(" ", "_")
                family_score = best_family_prediction["score"]
                
                if family_name in valid_target_families:
                    classification_category = family_name
                else:
                    classification_category = 'other_families'
            
            # Get YOLO results if available
            yolo_data = yolo_results.get(img_filename, {})
            
            # Create combined row
            row_data = [
                img_filename,
                yolo_data.get('top1', ''),
                yolo_data.get('top1_prob', ''),
                family_name,
                family_score,
                classification_category
            ]
            
            writer.writerow(row_data)
            image_count += 1
            
            img_time = time.time() - img_start_time
            pbar.set_postfix({"Last": f"{img_time:.2f}s"})
            pbar.update(1)

# Calculate overall processing time
end_time = time.time()
total_time = end_time - start_time
formatted_time = str(timedelta(seconds=int(total_time)))

print(f"\nClassification results saved to {csv_path}")
print(f"Processing complete. {image_count} images classified in {formatted_time} (total {total_time:.2f} seconds)")
print(f"Average time per image: {total_time/max(image_count, 1):.2f} seconds")
print(f"\nClassification categories used:")
print(f"- Valid target families from {target_families_path}")
print(f"- 'other_families' for families not in the target list")
print(f"- 'uncertain' for images with no family prediction")
