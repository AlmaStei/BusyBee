import os
import csv
import time
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# Define root directory
ROOT_DIR = "C:/Users/Almas/YOLOv5/yolov5-master/runs/predict-cls"
OUTPUT_CSV = "verification_results3.csv"
BOOKMARK_FILE = "bookmark.txt"

# Find all images in the classification folders
def get_image_list():
    image_paths = []
    for cam in os.listdir(ROOT_DIR):
        if cam in ["seppi-cam36", "seppi-cam37", "seppi-cam38"]:
            cam_path = os.path.join(ROOT_DIR, cam, "top1_classes", "prob_0.8-1.0")
            if os.path.exists(cam_path):
                for taxa in os.listdir(cam_path):
                    if "none_" in taxa.lower():
                        continue  # Skip folders with "none_" in the name
                    taxa_path = os.path.join(cam_path, taxa)
                    if os.path.isdir(taxa_path):
                        for img in os.listdir(taxa_path):
                            if img.lower().endswith((".jpg", ".png", ".jpeg")):
                                image_paths.append((os.path.join(taxa_path, img), taxa, img))
    return image_paths

# Load bookmark
def load_bookmark():
    if os.path.exists(BOOKMARK_FILE):
        with open(BOOKMARK_FILE, "r") as file:
            try:
                return int(file.read().strip())
            except ValueError:
                return 0
    return 0

# Save bookmark
def save_bookmark():
    with open(BOOKMARK_FILE, "w") as file:
        file.write(str(current_index))

# Load images
image_list = get_image_list()
current_index = load_bookmark()
start_time = time.time()

# Save classification result
def save_result(is_correct):
    global current_index, start_time
    if current_index < len(image_list):
        image_path, predicted_class, image_name = image_list[current_index]
        elapsed_time = time.time() - start_time
        with open(OUTPUT_CSV, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([image_path, predicted_class, image_name, "Correct" if is_correct else "Incorrect", round(elapsed_time, 2)])
        current_index += 1
        save_bookmark()
        start_time = time.time()  # Reset timer for next image
        load_next_image()

# Load the next image
def load_next_image():
    if current_index < len(image_list):
        image_path, predicted_class, image_name = image_list[current_index]
        img = Image.open(image_path)
        img = img.resize((500, 500))  # Resize for display
        img = ImageTk.PhotoImage(img)
        image_label.config(image=img)
        image_label.image = img
        class_label.config(text=f"Predicted: {predicted_class}")
        file_name_label.config(text=f"File: {image_name}")
    else:
        class_label.config(text="All images reviewed!")
        image_label.config(image='')
        file_name_label.config(text="")

# Handle key press events
def on_key_press(event):
    if event.keysym == "space":
        save_result(True)  # Space bar triggers "Correct" button

# Tkinter GUI setup
root = tk.Tk()
root.title("Image Verification Tool")

# UI Elements
image_label = tk.Label(root)
image_label.pack()
class_label = tk.Label(root, text="", font=("Arial", 14))
class_label.pack()
file_name_label = tk.Label(root, text="", font=("Arial", 12))
file_name_label.pack()

btn_correct = ttk.Button(root, text="Correct", command=lambda: save_result(True))
btn_correct.pack(side=tk.LEFT, padx=10, pady=10)

btn_incorrect = ttk.Button(root, text="Incorrect", command=lambda: save_result(False))
btn_incorrect.pack(side=tk.RIGHT, padx=10, pady=10)

# Start the app
start_time = time.time()
load_next_image()
root.mainloop()