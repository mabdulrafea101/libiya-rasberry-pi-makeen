import os
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from ultralytics import YOLO
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
import ultralytics

# Configuration
pd.set_option('display.max_columns', None)
plt.style.use('ggplot')

# Check Device
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

# Verify Ultralytics
print(f"Ultralytics version: {ultralytics.__version__}")

## 2. Dataset Analysis & Exploration
# Define dataset path
dataset_path = Path("bottle_classification.v2-version-2.folder")
train_path = dataset_path / "train"
valid_path = dataset_path / "valid"
test_path = dataset_path / "test"

# Function to count images
def count_images(path):
    counts = {}
    if not path.exists():
        return counts
    for class_dir in path.iterdir():
        if class_dir.is_dir():
            # Count common image formats
            count = len(list(class_dir.glob("*.jpg"))) + len(list(class_dir.glob("*.jpeg"))) + len(list(class_dir.glob("*.png")))
            counts[class_dir.name] = count
    return counts

train_counts = count_images(train_path)
valid_counts = count_images(valid_path)
test_counts = count_images(test_path)

print("Train counts:", train_counts)
print("Valid counts:", valid_counts)
print("Test counts:", test_counts)

## 3. Model Training
# Load a model
model = YOLO('yolo11n-cls.pt')  # load a pretrained model (recommended for training)

# Train the model
print("Starting training...")
# We use a small number of epochs for demonstration, increase for better results
# imgsz=224 is standard for classification models
# patience=10 stops training if no improvement for 10 epochs
results = model.train(data=dataset_path, epochs=20, imgsz=224, patience=10, batch=32, device=device, project='bottle_classification', name='yolo11n_cls')

## 4. Model Evaluation
# Validate the model
print("Validating model...")
metrics = model.val()
print(f"Top-1 Accuracy: {metrics.top1:.4f}")
print(f"Top-5 Accuracy: {metrics.top5:.4f}")

# Load the best model
best_model_path = Path(model.trainer.best)
print(f"Best model saved at: {best_model_path}")

# Detailed Evaluation on Test Set
if best_model_path.exists():
    print("Running predictions on Test Set...")
    
    # FIX: Explicitly collect image paths to avoid FileNotFoundError on directory
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    test_images = [str(p) for p in test_path.rglob('*') if p.suffix.lower() in image_extensions]
    
    if not test_images:
        print(f"No images found in {test_path}")
        test_results = []
    else:
        # Pass the list of images instead of the directory path
        test_results = model.predict(test_images, stream=True)

    y_true = []
    y_pred = []
    class_names = model.names

    for result in test_results:
        # Extract true class from path (assuming folder structure: test/class_name/image.jpg)
        path_obj = Path(result.path)
        true_label = path_obj.parent.name
        
        # Prediction
        pred_label_idx = result.probs.top1
        pred_label = class_names[pred_label_idx]
        
        y_true.append(true_label)
        y_pred.append(pred_label)

    # Classification Report
    if y_true:
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred, labels=list(class_names.values()))
        print("Confusion Matrix:")
        print(cm)
    else:
        print("No predictions to evaluate.")
else:
    print("Skipping evaluation as model not found.")

## 5. Model Export
if best_model_path.exists():
    # Export to ONNX
    export_path = model.export(format='onnx', imgsz=224, dynamic=False, opset=12)
    print(f"Model exported to: {export_path}")
    
    # Optional: Verify ONNX model with ONNX Runtime
    import onnxruntime as ort
    try:
        ort_session = ort.InferenceSession(export_path)
        print("ONNX model loaded successfully with ONNX Runtime.")
    except Exception as e:
        print(f"Failed to load ONNX model: {e}")
else:
    print("Skipping export.")
