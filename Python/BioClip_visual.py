#script to classify images using the TreeOfLifeClassifier and organize them by order and family
#.\env_bioclip\Scripts\activate
import os
import random
import shutil
import csv
from bioclip import TreeOfLifeClassifier, Rank

# Paths
main_folder = "C:/Users/Almas/bioclip_test/test_subfolder_2"  # Adjust path
output_folder = "C:/Users/Almas/bioclip_test/BioClip_testrun/test4"
csv_path = os.path.join(output_folder, "classifications.csv")

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Step 1: Traverse subfolders and randomly select images
for root, dirs, files in os.walk(main_folder):
    image_files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    
    # Randomly select up to 15 images
    selected_images = random.sample(image_files, min(len(image_files), 188))
    
    for image in selected_images:
        src_path = os.path.join(root, image)
        dest_path = os.path.join(output_folder, image)
        shutil.copy(src_path, dest_path)

# Step 2: Initialize the classifier
classifier = TreeOfLifeClassifier(device='cuda')

# Load target families from file
with open("C:/Users/Almas/bioclip_test/MadHornet/gbif_families.txt", 'r') as f:
    target_families = [line.strip() for line in f.readlines()]

# Step 3: Classify images and organize by family
with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Image', 'Order', 'Order_Confidence', 'Family', 'Family_Confidence', 'Classification_Path'])

    for image_file in os.listdir(output_folder):
        if not image_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(output_folder, image_file)

        # First check if image contains insects
        class_predictions = classifier.predict(image_path, Rank.CLASS)
        
        if not class_predictions or not any(pred["class"] == "Insecta" for pred in class_predictions):
            classification_path = 'non_insect_images'
            target_folder = os.path.join(output_folder, "non_insect_images")
            # Set empty values for insect-specific classifications
            family_name = ''
            family_score = 0
            order_name = ''
            order_score = 0
        else:
            # Existing family classification logic
            family_predictions = classifier.predict(image_path, Rank.FAMILY)
            order_predictions = classifier.predict(image_path, Rank.ORDER)

            # Initialize values
            family_name = ''
            family_score = 0
            order_name = ''
            order_score = 0
            classification_path = ''

            # Get the best family prediction
            best_family_prediction = max(family_predictions, key=lambda x: x["score"])
            family_name = best_family_prediction["family"].replace(" ", "_")
            family_score = best_family_prediction["score"]

            # Get order information if available
            if order_predictions:
                best_order_prediction = max(order_predictions, key=lambda x: x["score"])
                order_name = best_order_prediction["order"].replace(" ", "_")
                order_score = best_order_prediction["score"]

            # Check if the predicted family is in our target families
            if family_name in target_families:
                target_folder = os.path.join(output_folder, family_name)
                classification_path = family_name
            else:
                target_folder = os.path.join(output_folder, "other_families")
                classification_path = 'other_families'

        # Create the target folder if it doesn't exist
        os.makedirs(target_folder, exist_ok=True)

        # Move the image to the appropriate folder
        shutil.move(image_path, os.path.join(target_folder, image_file))

        # Write to CSV
        writer.writerow([
            image_file,
            order_name,
            order_score,
            family_name,
            family_score,
            classification_path
        ])

print(f"Classification results saved to {csv_path}")
print("Processing complete. Images have been classified and organized.")