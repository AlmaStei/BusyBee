### pybioclip
# https://imageomics.github.io/pybioclip/python-tutorial/

# make sure to run env_bioclip\Scripts\activate this in the bash terminal before running this
## Predict species classification
#from bioclip import TreeOfLifeClassifier, Rank

# Create the classifier instance
#classifier = TreeOfLifeClassifier()

# Predict species classification for the image
#predictions = classifier.predict("2024-07-24_16-15-16-941965_ID7187_crop.jpg", Rank.SPECIES)



# Print each prediction
#for prediction in predictions:
#    print(prediction["species"], "-", prediction["score"])
#
# ------------------------------------------
#import os
#import random
#import shutil
#from bioclip import TreeOfLifeClassifier, Rank

# Paths
#main_folder = "C:/Users/Almas/YOLOv5/yolov5-master/runs/predict-cls/seppi-cam31/top1_classes/prob_0.8-1.0"  # Replace with the path to your main folder
#output_folder = "C:/Users/Almas/bioclip_test/BioClip_testrun/cam31"

# Create the output folder if it doesn't exist
#os.makedirs(output_folder, exist_ok=True)

# Step 1: Traverse subfolders and randomly select images
#for root, dirs, files in os.walk(main_folder):
#    image_files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
#    
#    # Randomly select up to 5 images
#    selected_images = random.sample(image_files, min(len(image_files), 15))
    
#    for image in selected_images:
#        src_path = os.path.join(root, image)
 #       dest_path = os.path.join(output_folder, image)
  #      shutil.copy(src_path, dest_path)

# Step 2: Initialize the classifier
#classifier = TreeOfLifeClassifier()

# Step 3: Classify images and organize by species
#for image_file in os.listdir(output_folder):
#    image_path = os.path.join(output_folder, image_file)
#
#    if not os.path.isfile(image_path):
#        continue

    # Predict species classification
#    predictions = classifier.predict(image_path, Rank.FAMILY) # Rank.SPECIES

#    if not predictions:
#        print(f"No predictions for {image_file}")
#        continue

    # Get the species with the highest score
#    best_prediction = max(predictions, key=lambda x: x["score"])
#    species_name = best_prediction["family"].replace(" ", "_")  # Replace spaces with underscores for folder names
 #   prediction_score = best_prediction["score"]

    # Determine the folder based on the prediction score
#    if prediction_score < 0.3:
 #       species_folder = os.path.join(output_folder, "uncertain_0.3")
  #  else:
   #     species_folder = os.path.join(output_folder, species_name)

    # Create the target folder if it doesn't exist
#    os.makedirs(species_folder, exist_ok=True)

    # Move the image to the appropriate folder
#    shutil.move(image_path, os.path.join(species_folder, image_file))

#print("Processing complete. Images have been classified and organized.")


# ------------------------------------------
import os
import random
import shutil
from bioclip import TreeOfLifeClassifier, Rank

# Paths
main_folder = "C:/Users/Almas/YOLOv5/yolov5-master/runs/predict-cls/seppi-cam31/top1_classes/prob_0.8-1.0"  # Adjust path
output_folder = "C:/Users/Almas/bioclip_test/BioClip_testrun/cam31"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Step 1: Traverse subfolders and randomly select images
for root, dirs, files in os.walk(main_folder):
    image_files = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    
    # Randomly select up to 15 images
    selected_images = random.sample(image_files, min(len(image_files), 15))
    
    for image in selected_images:
        src_path = os.path.join(root, image)
        dest_path = os.path.join(output_folder, image)
        shutil.copy(src_path, dest_path)

# Step 2: Initialize the classifier
classifier = TreeOfLifeClassifier()

# Define the insect orders to classify further
target_orders = ["Hymenoptera", "Diptera", "Lepidoptera", "Coleoptera"]

# Step 3: Classify images and organize by order and family
for image_file in os.listdir(output_folder):
    image_path = os.path.join(output_folder, image_file)

    if not os.path.isfile(image_path):
        continue

    # Predict taxa classification at ORDER level
    order_predictions = classifier.predict(image_path, Rank.ORDER)

    if not order_predictions:
        print(f"No predictions for {image_file}")
        continue

    # Get the best order prediction
    best_order_prediction = max(order_predictions, key=lambda x: x["score"])
    order_name = best_order_prediction["order"].replace(" ", "_")  # Normalize folder names
    prediction_score = best_order_prediction["score"]

    # If confidence is low, move to "uncertain_0.3"
    if prediction_score < 0.3:
        target_folder = os.path.join(output_folder, "uncertain_0.3")

    elif order_name in target_orders:
        # Merge the selected orders into one folder called "HyCoDiLe"
        target_folder = os.path.join(output_folder, "HyCoDiLe")

        # Classify further by FAMILY
        family_predictions = classifier.predict(image_path, Rank.FAMILY)

        if not family_predictions:
            print(f"No family-level predictions for {image_file}")
            continue

        # Get the best family prediction
        best_family_prediction = max(family_predictions, key=lambda x: x["score"])
        family_name = best_family_prediction["family"].replace(" ", "_")

        target_folder = os.path.join(target_folder, family_name)  # HyCoDiLe → Family

    else:
        # If order is not in the selected four, move to "Other" → Order
        target_folder = os.path.join(output_folder, "Other", order_name)

    # Create the target folder if it doesn't exist
    os.makedirs(target_folder, exist_ok=True)

    # Move the image to the appropriate folder
    shutil.move(image_path, os.path.join(target_folder, image_file))

print("Processing complete. Images have been classified and organized.")
