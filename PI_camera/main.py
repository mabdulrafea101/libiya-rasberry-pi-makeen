import cv2
import numpy as np
import onnxruntime as ort
import paho.mqtt.client as mqtt
import time
import json
import ssl

# Configuration
MODEL_PATH = "best.onnx"
CAMERA_INDEX = 0

# HiveMQ Cloud Connection Details
MQTT_BROKER = "00abc31a42b94a7ea56216cf0b5b956d.s1.eu.hivemq.cloud"
MQTT_PORT = 8883
MQTT_USERNAME = "YOUR_USERNAME"  # <--- ENTER YOUR HIVEMQ USERNAME HERE
MQTT_PASSWORD = "YOUR_PASSWORD"  # <--- ENTER YOUR HIVEMQ PASSWORD HERE
MQTT_TOPIC_RESULT = "bottle/classification"
MQTT_TOPIC_COMMAND = "bottle/command"

CAPTURE_DURATION = 2.0  # seconds to capture video
CONFIDENCE_THRESHOLD = 0.5

# Initialize ONNX Runtime
try:
    session = ort.InferenceSession(MODEL_PATH)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    input_shape = session.get_inputs()[0].shape
    img_height, img_width = input_shape[2], input_shape[3]
    if isinstance(img_height, str) or isinstance(img_width, str):
         img_height, img_width = 224, 224
    print(f"Model loaded. Input: {input_name}, Shape: {input_shape}. Target size: {img_width}x{img_height}")
except Exception as e:
    print(f"Failed to load model: {e}")
    exit(1)

def preprocess(image):
    img = cv2.resize(image, (img_width, img_height))
    img = img.astype(np.float32)
    img /= 255.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    return img

def get_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def process_cycle(client):
    print("Starting capture cycle...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not open camera {CAMERA_INDEX}")
        return

    frames = []
    start_time = time.time()
    
    while (time.time() - start_time) < CAPTURE_DURATION:
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            print("Failed to capture frame")
            break
    
    cap.release()
    print(f"Captured {len(frames)} frames.")

    if not frames:
        print("No frames captured.")
        return

    best_frame = None
    max_sharpness = -1

    for frame in frames:
        sharpness = get_sharpness(frame)
        if sharpness > max_sharpness:
            max_sharpness = sharpness
            best_frame = frame

    print(f"Selected best frame with sharpness: {max_sharpness}")
    cv2.imwrite("best_frame.jpg", best_frame)

    input_data = preprocess(best_frame)
    
    try:
        outputs = session.run([output_name], {input_name: input_data})
        output = outputs[0]
        
        probs = np.exp(output) / np.sum(np.exp(output), axis=1, keepdims=True)
        probs = probs[0]
        
        class_id = np.argmax(probs)
        confidence = probs[class_id]
        
        result = {
            "class_id": int(class_id),
            "confidence": float(confidence),
            "raw_probs": probs.tolist(),
            "timestamp": time.time()
        }
        
        print(f"Classification Result: {result}")
        client.publish(MQTT_TOPIC_RESULT, json.dumps(result))
        print(f"Published to {MQTT_TOPIC_RESULT}")

    except Exception as e:
        print(f"Inference failed: {e}")

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker!")
        client.subscribe(MQTT_TOPIC_COMMAND)
        print(f"Subscribed to {MQTT_TOPIC_COMMAND}")
    else:
        print(f"Failed to connect, return code {rc}")

def on_message(client, userdata, msg):
    print(f"Received command on {msg.topic}: {msg.payload.decode()}")
    process_cycle(client)

def main():
    client = mqtt.Client()
    
    # TLS Setup for HiveMQ Cloud
    client.tls_set(tls_version=ssl.PROTOCOL_TLS)
    client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
    
    client.on_connect = on_connect
    client.on_message = on_message

    print(f"Connecting to {MQTT_BROKER}...")
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_forever()
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    main()
