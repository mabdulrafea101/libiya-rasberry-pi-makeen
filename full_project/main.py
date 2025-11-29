import cv2
import numpy as np
import onnxruntime as ort
import paho.mqtt.client as mqtt
import time
import json
import ssl
import threading

# --- Configuration ---
MODEL_PATH = "best.onnx"
CAMERA_INDEX = 0

# MQTT Credentials (from user history)
MQTT_BROKER = "00abc31a42b94a7ea56216cf0b5b956d.s1.eu.hivemq.cloud"
MQTT_PORT = 8883
MQTT_USERNAME = "rasberrypi5"
MQTT_PASSWORD = "_Testing_123"

# Topics
TOPIC_CONTROL = "bottle/control"   # Sub: {"command": "start/stop", "mode": "auto/manual"}
TOPIC_STATUS = "bottle/status"     # Pub: {"state": "...", "mode": "..."}
TOPIC_RESULT = "bottle/result"     # Pub: Classification result
TOPIC_HISTORY = "bottle/history"   # Pub: List of recent results

# GPIO Pins (BCM Numbering)
PIN_CONVEYOR = 17
PIN_LABELER = 27
PIN_SENSOR = 22

# Timing (Seconds)
TIME_TO_LABELER = 2.0  # Time from start to labeler position
TIME_LABELING = 1.0    # Duration of labeling
TIME_TO_CAMERA = 2.0   # Time from labeler to camera

# --- GPIO Setup (Mockable) ---
try:
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(PIN_CONVEYOR, GPIO.OUT)
    GPIO.setup(PIN_LABELER, GPIO.OUT)
    GPIO.setup(PIN_SENSOR, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    MOCK_GPIO = False
    print("RPi.GPIO loaded.")
except ImportError:
    print("RPi.GPIO not found. Using Mock GPIO.")
    MOCK_GPIO = True
    class MockGPIO:
        LOW = 0
        HIGH = 1
        def output(self, pin, state): print(f"[GPIO] Pin {pin} -> {state}")
        def input(self, pin): return 0 # Default no bottle
    GPIO = MockGPIO()

# --- Global State ---
state = {
    "mode": "manual",  # manual, auto
    "running": False,  # Is the auto loop running?
    "conveyor": False
}

stats = {
    "total": 0,
    "correct": 0,
    "misaligned": 0,
    "no_label": 0
}

# --- AI Model ---
try:
    session = ort.InferenceSession(MODEL_PATH)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    input_shape = session.get_inputs()[0].shape
    img_height, img_width = 224, 224 # Default
    if len(input_shape) == 4:
        h, w = input_shape[2], input_shape[3]
        if isinstance(h, int): img_height, img_width = h, w
    print(f"Model loaded. Target: {img_width}x{img_height}")
except Exception as e:
    print(f"Warning: Model not found ({e}). AI will return mock results.")
    session = None

# --- Helper Functions ---
def set_conveyor(on):
    state["conveyor"] = on
    GPIO.output(PIN_CONVEYOR, GPIO.HIGH if on else GPIO.LOW)
    publish_status()

def trigger_labeler():
    print("Triggering Labeler...")
    GPIO.output(PIN_LABELER, GPIO.HIGH)
    time.sleep(0.5)
    GPIO.output(PIN_LABELER, GPIO.LOW)

def capture_and_classify():
    print("Capturing image...")
    # Pause conveyor for clear shot?
    # set_conveyor(False) 
    # time.sleep(0.5)
    
    cap = cv2.VideoCapture(CAMERA_INDEX)
    frames = []
    if cap.isOpened():
        for _ in range(5): # Capture a few frames
            ret, frame = cap.read()
            if ret: frames.append(frame)
        cap.release()
    
    # Resume conveyor if needed
    # set_conveyor(True)

    if not frames:
        print("Camera failed. Using mock data.")
        return {"class_id": 0, "confidence": 0.0}

    # Sharpness
    best_frame = max(frames, key=lambda f: cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    
    # Inference
    if session:
        img = cv2.resize(best_frame, (img_width, img_height)).astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        outputs = session.run([output_name], {input_name: img})
        probs = np.exp(outputs[0]) / np.sum(np.exp(outputs[0]), axis=1, keepdims=True)
        class_id = np.argmax(probs[0])
        conf = float(probs[0][class_id])
        probs = probs[0].tolist() # Convert to list for JSON
    else:
        # Mock data when model is not available
        class_id = 0
        conf = 0.99
        probs = [0.99, 0.005, 0.005] # Example mock probabilities

    # Update Stats (total count)
    stats["total"] += 1

    # Mapping: 0=Correct, 1=Misaligned, 2=No Label
    if class_id == 0: stats["correct"] += 1
    elif class_id == 1: stats["misaligned"] += 1
    else: stats["no_label"] += 1

    # Class mapping
    class_names = {0: "Correct", 1: "Misaligned", 2: "No Label"}
    class_name = class_names.get(class_id, "Unknown")

    print(f"Classification Result: {class_name} ({class_id}), Confidence: {conf:.2f}")
    
    # Publish result
    result_payload = {
        "class_id": int(class_id),
        "class_name": class_name,
        "confidence": float(conf),
        "raw_probs": [float(p) for p in probs],
        "timestamp": time.time()
    }
    client.publish(TOPIC_RESULT, json.dumps(result_payload))
    print(f"Published to {TOPIC_RESULT}")
    
    publish_status()
    
    return result_payload

def publish_status():
    payload = {
        "mode": state["mode"],
        "running": state["running"],
        "conveyor": state["conveyor"],
        "stats": stats
    }
    client.publish(TOPIC_STATUS, json.dumps(payload), retain=True)

def publish_result(result):
    # This function is now redundant as capture_and_classify publishes directly
    # Keeping it for now, but it's not used after the change.
    client.publish(TOPIC_RESULT, json.dumps(result))
    # Also publish to history (could be same topic, handled by Node-RED)

# --- Control Logic ---
def run_manual_cycle():
    print("Starting Manual Cycle")
    set_conveyor(True)
    time.sleep(TIME_TO_LABELER)
    
    set_conveyor(False) # Stop for labeling
    trigger_labeler()
    time.sleep(TIME_LABELING)
    
    set_conveyor(True)
    time.sleep(TIME_TO_CAMERA)
    
    set_conveyor(False) # Stop for camera
    result = capture_and_classify()
    print(f"Result: {result}")
    publish_result(result)
    publish_status() # Update stats

def auto_loop():
    print("Auto Mode Started")
    while state["running"] and state["mode"] == "auto":
        # Check Sensor
        if GPIO.input(PIN_SENSOR) == GPIO.HIGH or MOCK_GPIO: # Mock always true for testing if needed
            print("Bottle Detected!")
            # Logic similar to manual but continuous
            # Move bottle to labeler
            set_conveyor(True)
            time.sleep(TIME_TO_LABELER)
            
            set_conveyor(False)
            trigger_labeler()
            time.sleep(TIME_LABELING)
            
            set_conveyor(True)
            time.sleep(TIME_TO_CAMERA)
            
            set_conveyor(False)
            result = capture_and_classify()
            publish_result(result)
            publish_status()
            
            # Wait for bottle to clear or next sensor trigger
            time.sleep(1.0) 
        else:
            time.sleep(0.1)
    
    set_conveyor(False)
    print("Auto Mode Stopped")

# --- MQTT Callbacks ---
def on_connect(client, userdata, flags, rc):
    print(f"Connected (RC: {rc})")
    client.subscribe(TOPIC_CONTROL)
    publish_status()

def on_message(client, userdata, msg):
    try:
        data = json.loads(msg.payload.decode())
        print(f"Command: {data}")
        
        if "mode" in data:
            state["mode"] = data["mode"]
            # If switching modes, stop running
            state["running"] = False
            
        if "command" in data:
            cmd = data["command"]
            if cmd == "start":
                if state["mode"] == "manual":
                    threading.Thread(target=run_manual_cycle).start()
                elif state["mode"] == "auto":
                    if not state["running"]:
                        state["running"] = True
                        threading.Thread(target=auto_loop).start()
            elif cmd == "stop":
                state["running"] = False
                set_conveyor(False)
        
        publish_status()
        
    except Exception as e:
        print(f"Error handling message: {e}")

# --- Main ---
client = mqtt.Client()
client.tls_set(tls_version=ssl.PROTOCOL_TLS)
client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
client.on_connect = on_connect
client.on_message = on_message

if __name__ == "__main__":
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_forever()
    except KeyboardInterrupt:
        print("Exiting...")
        if not MOCK_GPIO: GPIO.cleanup()
