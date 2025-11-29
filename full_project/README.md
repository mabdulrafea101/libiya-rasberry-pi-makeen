# Full Bottle Labeling System

## Overview

This is a complete industrial-grade bottle labeling system powered by Raspberry Pi 5, AI (ONNX), and Node-RED. It features two operational modes (Manual & Auto) and a comprehensive dashboard for control and analytics.

## Hardware Requirements

- **Raspberry Pi 5**
- **Camera Module**
- **Conveyor Belt Motor** (Connected to GPIO 17)
- **Labeler Trigger** (Connected to GPIO 27)
- **IR Sensor** (Connected to GPIO 22)

## Software Setup

### 1. Installation

Navigate to the project folder and install dependencies:

```bash
cd full_project
pip install -r requirements.txt
```

### 2. Model Setup

Ensure your trained model `best.onnx` is in this folder. If not, copy it:

```bash
cp ../PI_camera/best.onnx .
```

### 3. Node-RED Dashboard

1. Open Node-RED (`http://<PI_IP>:1880`).
2. Install the Dashboard palette if not already installed (`node-red-dashboard`).
3. Import `node_red_flow.json`.
4. **Important**: Configure the "HiveMQ Cloud" broker node with your credentials (`rasberrypi5` / `_Testing_123`).
5. Deploy.

### 4. Running the System

Start the main control script:

```bash
python main.py
```

## Dashboard Guide

Open the Dashboard (`http://<PI_IP>:1880/ui`).

### Tab 1: Control Panel

- **Mode Switch**: Toggle between **Manual** and **Auto**.
- **START**:
  - In **Manual**: Runs one complete cycle (Conveyor -> Label -> Camera).
  - In **Auto**: Starts the continuous loop (Sensor -> Label -> Camera).
- **STOP**: Emergency stop for conveyor and auto loop.
- **Status**: Shows current system state.

### Tab 2: History

- Shows a real-time table of the last 10 classification results with timestamps.

### Tab 3: Report

- **Pie Chart**: Visual distribution of Correct, Misaligned, and No Label bottles.

## Customization

- **Timings**: Edit `main.py` to adjust `TIME_TO_LABELER`, `TIME_LABELING`, etc. to match your physical conveyor speed.
- **GPIO**: Edit `PIN_...` variables in `main.py` to match your wiring.
