import cv2
import numpy as np
import onnxruntime as ort
import os

# Configuration
MODEL_PATH = "PI_camera/best.onnx"
TEST_IMAGE_PATH = "PI_camera/test_misaligned.jpg"

# Initialize ONNX Runtime
try:
    session = ort.InferenceSession(MODEL_PATH)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    input_shape = session.get_inputs()[0].shape
    img_height, img_width = 224, 224
    if len(input_shape) == 4:
        h, w = input_shape[2], input_shape[3]
        if isinstance(h, int): img_height, img_width = h, w
    print(f"Model loaded. Input: {input_name}, Shape: {input_shape}. Target size: {img_width}x{img_height}")
except Exception as e:
    print(f"Failed to load model: {e}")
    exit(1)

def preprocess(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return None
    
    img = cv2.resize(img, (img_width, img_height))
    img = img.astype(np.float32)
    img /= 255.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    return img

def main():
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"Test image not found at {TEST_IMAGE_PATH}")
        return

    input_data = preprocess(TEST_IMAGE_PATH)
    if input_data is None:
        return
    
    try:
        outputs = session.run([output_name], {input_name: input_data})
        output = outputs[0]
        
        # Softmax
        probs = np.exp(output) / np.sum(np.exp(output), axis=1, keepdims=True)
        probs = probs[0]
        
        class_id = np.argmax(probs)
        confidence = probs[class_id]
        
        # Class mapping based on training data analysis
        # Train counts: {'no_label': 15, 'correct_label': 83, 'misaligned_label': 312}
        # YOLO usually sorts classes alphabetically or by index in data.yaml
        # We need to check the class mapping. Assuming alphabetical:
        # 0: correct_label
        # 1: misaligned_label
        # 2: no_label
        
        classes = ["correct_label", "misaligned_label", "no_label"]
        predicted_class = classes[class_id] if class_id < len(classes) else f"Unknown ({class_id})"

        print(f"Prediction for {TEST_IMAGE_PATH}:")
        print(f"Class ID: {class_id}")
        print(f"Class Name (Assumed): {predicted_class}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Raw Probabilities: {probs}")

    except Exception as e:
        print(f"Inference failed: {e}")

if __name__ == "__main__":
    main()
