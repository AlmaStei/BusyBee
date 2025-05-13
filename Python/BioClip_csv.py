#script to classify images using the TreeOfLifeClassifier and organize them by family
# store results in a csv file

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

# Paths
main_folder = "C:/Users/Almas/Ultralytics/runs/detect/20250225_predict_seppi-cam32/crops/arthropod" # Adjust path
# Just create a single output folder for the CSV
output_folder = "C:/Users/Almas/bioclip_test/BioClip_testrun"  
os.makedirs(output_folder, exist_ok=True)

csv_path = os.path.join(output_folder, "classifications_cam32.csv")

# Step 1: Initialize the classifier
classifier = TreeOfLifeClassifier()

# Get valid families from BioClip
label_data = classifier.get_label_data()
valid_families = set(label_data[Rank.FAMILY.get_label()])

# Load and filter target families with detailed reporting
with open("C:/Users/Almas/bioclip_test/MadHornet/gbif_families.txt", 'r') as f:
    target_families = [line.strip() for line in f.readlines() if line.strip()]
    # Find excluded families
    excluded_families = [f for f in target_families if f not in valid_families]
    valid_target_families = [f for f in target_families if f in valid_families]
    
    # Print detailed report
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
        # Read existing CSV to find processed images
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

# Step 2: Classify images and save results to CSV
image_count = len(processed_images)  # Start count from processed images
remaining_count = 0  # Count images that still need processing

# First, count total images to process and identify remaining ones
total_images = 0
remaining_images = []
for root, _, files in os.walk(main_folder):
    for image_file in files:
        if not image_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
            
        image_path = os.path.join(root, image_file)
        total_images += 1
        
        if image_path not in processed_images:
            remaining_images.append(image_path)
            remaining_count += 1

print(f"Total images: {total_images}")
print(f"Already processed: {image_count}")
print(f"Remaining to process: {remaining_count}")

# Open CSV file in append mode if it exists, otherwise create new
file_mode = 'a' if os.path.exists(csv_path) and processed_images else 'w'
with open(csv_path, file_mode, newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    # Write header only if creating a new file
    if file_mode == 'w':
        writer.writerow(['Image_Path', 'Family', 'Family_Confidence', 'Classification_Category'])

    # Create progress bar for remaining images
    with tqdm(total=remaining_count, desc="Processing remaining images", unit="img") as pbar:
        # Process only the remaining images
        for image_path in remaining_images:
            img_start_time = time.time()

            # Get family prediction
            family_predictions = classifier.predict(image_path, Rank.FAMILY)
            
            if not family_predictions:
                classification_path = 'uncertain'
                row_data = [image_path, '', 0, classification_path]
            else:
                best_family_prediction = max(family_predictions, key=lambda x: x["score"])
                family_name = best_family_prediction["family"].replace(" ", "_")
                family_score = best_family_prediction["score"]
                
                classification_path = family_name if family_name in valid_target_families else 'other_families'
                row_data = [image_path, family_name, family_score, classification_path]

            writer.writerow(row_data)
            image_count += 1  # Update total count
            
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