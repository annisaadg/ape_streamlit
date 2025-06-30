import torch
from ultralytics import YOLO
import os

# Folder berisi model YOLOv8 (.pt)
model_folder = 'models'

# Loop semua file .pt di folder
for filename in os.listdir(model_folder):
    if filename.endswith('.pt'):
        model_path = os.path.join(model_folder, filename)
        try:
            print(f"\n==== Model: {filename} ====")
            model = YOLO(model_path)
            model.info(verbose=True)  # Tampilkan ringkasan detail layer dan jumlah parameter
        except Exception as e:
            print(f"Error loading {filename}: {e}")
