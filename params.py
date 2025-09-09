from ultralytics import YOLO

# Load your YOLOv8 model
model = YOLO('models/Baseline.pt')  # Replace with your model path

# Print the model summary
model.info()