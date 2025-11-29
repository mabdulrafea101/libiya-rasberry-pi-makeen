# Bottle Label Classification Model Training Plan

## Project Overview
Develop a high-accuracy YOLO classification model to detect bottle label quality in a real-time production environment using Raspberry Pi 5 and Logitech camera.

### Classification Categories
1. **Correct Label** - Properly aligned and positioned labels
2. **Misaligned Label** - Labels that are crooked, shifted, or incorrectly positioned
3. **No Label** - Bottles without labels

### Target Deployment
- **Hardware**: Raspberry Pi 5
- **Camera**: Logitech camera (specifications in reference documents)
- **Model Format**: ONNX (optimized for edge deployment)
- **Application**: Real-time bottle labeling machine quality control

---

## Dataset Information

### Current Dataset Structure
```
bottle classification.v2-version-2.folder/
├── train/
│   ├── correct_label/
│   ├── misaligned_label/
│   └── no_label/
├── valid/
│   ├── correct_label/
│   ├── misaligned_label/
│   └── no_label/
└── test/
    ├── correct_label/
    └── misaligned_label/
    └── no_label/
```

### Dataset Specifications
- **Total Images**: 469 images
- **Image Size**: 224x224 pixels (stretched)
- **Format**: Folder-based classification (YOLO-CLS format)
- **Pre-processing Applied**:
  - Auto-orientation with EXIF stripping
  - Resize to 224x224 (Stretch)
- **Augmentations Applied**:
  - 50% horizontal flip
  - Random rotation (-5° to +5°)
  - Random brightness adjustment (-15% to +15%)
  - Salt and pepper noise (0.1% pixel coverage)

---

## Notebook Structure & Implementation Plan

### 1. Environment Setup & Dependencies
**Objective**: Install and configure all required libraries

#### Tasks:
- Install Ultralytics (YOLOv8/v11)
- Install Roboflow for dataset management
- Install ONNX and ONNX Runtime for model export
- Install visualization libraries (matplotlib, seaborn, plotly)
- Install metrics libraries (scikit-learn)
- Install Pillow for image processing

#### Code Sections:
```python
# Install dependencies
- ultralytics (latest for YOLOv11 or specific version for YOLOv8)
- roboflow
- onnx
- onnxruntime
- matplotlib, seaborn, plotly
- scikit-learn
- opencv-python
- pandas, numpy
```

#### Validation:
- Print version numbers
- Test GPU availability (if available)
- Verify CUDA/MPS support for acceleration

---

### 2. Dataset Analysis & Exploration
**Objective**: Understand dataset distribution and quality

#### Tasks:
- Load dataset using Roboflow API or local folder structure
- Count images per class in train/valid/test splits
- Visualize class distribution (bar charts, pie charts)
- Display sample images from each class
- Check for class imbalance
- Analyze image quality and resolution
- Verify image dimensions and aspect ratios

#### Outputs:
- Class distribution tables
- Sample image grids (3x3 or 4x4) for each class
- Dataset statistics summary
- Imbalance ratio calculations

#### Quality Checks:
- Verify no corrupted images
- Check for duplicate images
- Validate label consistency
- Identify potential data quality issues

---

### 3. Data Preparation & Augmentation Strategy
**Objective**: Prepare data for optimal training

#### Tasks:
- Create/verify data.yaml configuration file for YOLO
- Define additional augmentation strategies for training
- Set up data loaders
- Implement class weighting if imbalanced
- Create stratified splits if needed

#### Data Configuration (data.yaml):
```yaml
path: /path/to/dataset
train: train
val: valid
test: test
nc: 3
names: ['correct_label', 'misaligned_label', 'no_label']
```

#### Augmentation Strategy:
- Use Ultralytics built-in augmentations
- Consider additional augmentations:
  - Color jittering (simulate lighting variations)
  - Gaussian blur (simulate camera focus issues)
  - Contrast adjustments
  - Rotation (for label alignment variations)
  - Scale variations

---

### 4. Model Selection & Configuration
**Objective**: Choose optimal YOLO model for Raspberry Pi deployment

#### Model Options:
1. **YOLOv8n-cls** (Nano) - Fastest, smallest
2. **YOLOv8s-cls** (Small) - Balanced speed/accuracy
3. **YOLOv8m-cls** (Medium) - Higher accuracy
4. **YOLOv11n-cls** (Nano) - Latest architecture
5. **YOLOv11s-cls** (Small) - Latest, balanced

#### Recommendation for Raspberry Pi 5:
- Start with **YOLOv8n-cls** or **YOLOv11n-cls** (nano versions)
- If accuracy insufficient, try small versions
- Consider quantization for further optimization

#### Configuration Parameters:
```python
model_configs = {
    'imgsz': 224,  # Match dataset size
    'batch_size': 16-32,  # Adjust based on available memory
    'epochs': 100-300,  # With early stopping
    'patience': 50,  # Early stopping patience
    'optimizer': 'AdamW' or 'SGD',
    'lr0': 0.001,  # Initial learning rate
    'lrf': 0.01,  # Final learning rate factor
    'momentum': 0.9,
    'weight_decay': 0.0005,
    'warmup_epochs': 3,
    'cos_lr': True,  # Cosine learning rate scheduler
}
```

---

### 5. Model Training
**Objective**: Train multiple models and find best configuration

#### Training Strategy:
1. **Baseline Training**: Train with default parameters
2. **Hyperparameter Tuning**: Experiment with different configurations
3. **Transfer Learning**: Use pretrained COCO weights
4. **Ensemble Approach**: Train multiple models for comparison

#### Training Loop:
```python
# Pseudo-code structure
for model_variant in ['yolov8n-cls', 'yolov11n-cls']:
    - Initialize model with pretrained weights
    - Configure training parameters
    - Train model with validation
    - Save checkpoints
    - Log metrics (loss, accuracy, etc.)
    - Generate training curves
```

#### Monitoring:
- Track training/validation loss
- Track accuracy metrics
- Monitor learning rate schedule
- Save best model checkpoint
- Save last model checkpoint
- Log to Weights & Biases or TensorBoard (optional)

#### Training Optimizations:
- Use mixed precision training (FP16) if available
- Enable cache for faster data loading
- Use multi-worker data loading
- Implement gradient clipping
- Use class weights for imbalanced data

---

### 6. Model Validation & Evaluation
**Objective**: Comprehensive model performance assessment

#### Validation Metrics:
1. **Classification Metrics**:
   - Overall Accuracy
   - Per-class Accuracy
   - Precision (per class and macro/micro average)
   - Recall (per class and macro/micro average)
   - F1-Score (per class and macro/micro average)
   - Top-1 Accuracy
   - Top-2 Accuracy (if applicable)

2. **Confusion Matrix**:
   - Generate normalized and non-normalized versions
   - Visualize with heatmaps
   - Identify misclassification patterns

3. **ROC Curves & AUC** (if probabilities available):
   - One-vs-Rest ROC curves for each class
   - Calculate AUC scores

4. **Prediction Confidence Analysis**:
   - Distribution of confidence scores
   - Low confidence predictions review
   - Threshold optimization

#### Validation Process:
```python
# Evaluation steps
1. Load best model checkpoint
2. Run predictions on validation set
3. Calculate all metrics
4. Generate confusion matrix
5. Create classification report
6. Visualize results
7. Analyze failure cases
8. Test on test set (final evaluation)
```

#### Visualization Outputs:
- Training/validation loss curves
- Accuracy curves
- Confusion matrix heatmap
- Per-class precision-recall bars
- Sample predictions with confidence scores
- Misclassified examples analysis

---

### 7. Error Analysis & Model Refinement
**Objective**: Identify weaknesses and improve model

#### Analysis Tasks:
- Review misclassified images
- Identify common failure patterns
- Check for systematic errors
- Analyze class-specific performance
- Review low-confidence predictions
- Test edge cases

#### Refinement Strategies:
- Collect more data for weak classes
- Adjust augmentation for problematic cases
- Fine-tune model with adjusted class weights
- Experiment with different architectures
- Implement focal loss for hard examples
- Use test-time augmentation (TTA)

---

### 8. Model Export for Raspberry Pi Deployment
**Objective**: Export optimized model for edge deployment

#### Export Formats to Test:
1. **ONNX** (Recommended for Raspberry Pi):
   - Standard ONNX (FP32)
   - Quantized ONNX (INT8) for speed
   
2. **TensorFlow Lite** (Alternative):
   - TFLite (FP32)
   - TFLite (FP16)
   - TFLite (INT8)

3. **NCNN** (Alternative for ARM):
   - Optimized for mobile/embedded devices

#### Export Process:
```python
# ONNX Export
model.export(
    format='onnx',
    imgsz=224,
    dynamic=False,  # Fixed input size
    simplify=True,   # Simplify ONNX graph
    opset=12         # ONNX opset version
)

# Quantized ONNX (if needed)
- Use ONNX Runtime quantization tools
- Test INT8 quantization for speed
- Validate accuracy after quantization

# TFLite Export (alternative)
model.export(
    format='tflite',
    imgsz=224,
    int8=True  # For quantization
)
```

#### Post-Export Validation:
- Verify exported model loads correctly
- Test inference on sample images
- Compare accuracy: PyTorch vs ONNX vs TFLite
- Measure inference speed
- Check model size
- Test with ONNX Runtime on local machine
- Validate input/output shapes

---

### 9. Raspberry Pi Optimization & Testing
**Objective**: Optimize for Raspberry Pi 5 deployment

#### Optimization Strategies:
1. **Model Quantization**:
   - Dynamic quantization
   - Static quantization (INT8)
   - Compare accuracy vs speed tradeoff

2. **Input Size Optimization**:
   - Test smaller input sizes (192x192, 160x160)
   - Balance accuracy vs inference speed

3. **Inference Optimization**:
   - Enable ONNX Runtime optimizations
   - Use appropriate execution providers
   - Batch processing vs single image
   - Thread optimization for ARM

#### Raspberry Pi Specific Considerations:
- CPU Architecture: ARM Cortex-A76
- Limited RAM (4GB/8GB variants)
- No GPU acceleration (CPU only)
- Power constraints
- Real-time performance requirements

#### Performance Benchmarking:
```python
# Create benchmark script
- Measure FPS (frames per second)
- Measure latency (ms per inference)
- Monitor CPU usage
- Monitor memory usage
- Test with different batch sizes
- Test with different threads
```

#### Target Metrics:
- **Inference Speed**: < 100ms per image (ideal: < 50ms)
- **Accuracy**: > 95% overall
- **Model Size**: < 10MB (ideally < 5MB)

---

### 10. Integration Testing & Camera Compatibility
**Objective**: Ensure model works with Logitech camera setup

#### Camera Integration Tasks:
- Review camera specifications from provided PDFs
- Determine camera resolution and frame rate
- Test image preprocessing pipeline
- Implement real-time capture simulation
- Handle lighting variations
- Test with different bottle positions

#### Pre-processing Pipeline for Deployment:
```python
# Camera to model pipeline
1. Capture image from camera
2. Resize to 224x224
3. Normalize pixel values
4. Convert to appropriate format
5. Run inference
6. Post-process predictions
7. Return classification result
```

#### Testing Scenarios:
- Different lighting conditions
- Various bottle orientations
- Conveyor belt simulation (motion blur)
- Multiple bottles in frame
- Edge cases (partially visible bottles)

---

### 11. Documentation & Deployment Guide
**Objective**: Create comprehensive deployment documentation

#### Documentation Sections:
1. **Model Training Summary**:
   - Final model architecture
   - Training hyperparameters
   - Performance metrics
   - Dataset information

2. **Deployment Instructions**:
   - Raspberry Pi setup
   - Required libraries installation
   - Model loading code
   - Inference script
   - Camera integration code

3. **Performance Benchmarks**:
   - Accuracy on test set
   - Inference speed
   - Resource usage
   - Comparison table (different models/formats)

4. **Troubleshooting Guide**:
   - Common issues and solutions
   - Performance optimization tips
   - Accuracy improvement suggestions

5. **Future Improvements**:
   - Additional data collection
   - Model architecture updates
   - Feature enhancements

---

## Detailed Notebook Sections Breakdown

### Section 1: Setup (Estimated: 5-10 minutes)
- [ ] Install dependencies
- [ ] Import libraries
- [ ] Set random seeds for reproducibility
- [ ] Configure device (CPU/GPU/MPS)
- [ ] Set up logging

### Section 2: Data Loading (Estimated: 10-15 minutes)
- [ ] Load dataset from folders
- [ ] Verify folder structure
- [ ] Count images per class
- [ ] Create data.yaml
- [ ] Visualize sample images

### Section 3: Exploratory Data Analysis (Estimated: 15-20 minutes)
- [ ] Class distribution analysis
- [ ] Image dimension verification
- [ ] Data quality checks
- [ ] Identify class imbalance
- [ ] Statistical summary

### Section 4: Data Preparation (Estimated: 10 minutes)
- [ ] Verify train/val/test splits
- [ ] Configure data augmentation
- [ ] Set up class weights (if needed)
- [ ] Prepare data loaders

### Section 5: Model Configuration (Estimated: 5-10 minutes)
- [ ] Select YOLO model variant
- [ ] Configure training parameters
- [ ] Set up callbacks and monitoring
- [ ] Initialize model

### Section 6: Model Training (Estimated: 30-120 minutes depending on epochs)
- [ ] Train baseline model (YOLOv8n-cls)
- [ ] Train alternative model (YOLOv11n-cls)
- [ ] Monitor training progress
- [ ] Save checkpoints
- [ ] Generate training curves

### Section 7: Model Evaluation (Estimated: 20-30 minutes)
- [ ] Load best model
- [ ] Evaluate on validation set
- [ ] Calculate metrics
- [ ] Generate confusion matrix
- [ ] Create classification report
- [ ] Visualize results
- [ ] Analyze errors

### Section 8: Test Set Evaluation (Estimated: 10-15 minutes)
- [ ] Final evaluation on test set
- [ ] Compare with validation results
- [ ] Generate final metrics report

### Section 9: Model Export (Estimated: 15-20 minutes)
- [ ] Export to ONNX (FP32)
- [ ] Test ONNX inference
- [ ] Quantize to INT8 (optional)
- [ ] Export to TFLite (optional)
- [ ] Compare export formats
- [ ] Validate exported models

### Section 10: Deployment Preparation (Estimated: 20-30 minutes)
- [ ] Create inference script
- [ ] Test on sample images
- [ ] Benchmark performance
- [ ] Create deployment documentation
- [ ] Generate requirements.txt

---

## Expected Outcomes

### Model Performance Targets
- **Overall Accuracy**: > 95%
- **Per-Class Precision**: > 90% for each class
- **Per-Class Recall**: > 90% for each class
- **F1-Score**: > 90% for each class

### Deployment Targets
- **Model Size**: < 10MB (ONNX or TFLite)
- **Inference Speed on Raspberry Pi 5**: < 100ms per image
- **CPU Usage**: < 80% at peak
- **Memory Usage**: < 500MB

---

## Risk Mitigation

### Potential Challenges & Solutions

1. **Class Imbalance**:
   - Solution: Use class weights, focal loss, or collect more data

2. **Low Accuracy on Specific Class**:
   - Solution: Augment data for that class, collect more samples

3. **Slow Inference on Raspberry Pi**:
   - Solution: Use smaller model, quantization, reduce input size

4. **Model Overfitting**:
   - Solution: More augmentation, dropout, regularization, early stopping

5. **Camera Calibration Issues**:
   - Solution: Preprocessing pipeline adjustment, normalization

6. **Lighting Variations**:
   - Solution: Brightness augmentation, histogram equalization

---

## Tools & Resources

### Key Libraries
- **Ultralytics**: YOLOv8/v11 implementation
- **Roboflow**: Dataset management and version control
- **ONNX Runtime**: Efficient inference
- **OpenCV**: Image processing
- **Scikit-learn**: Metrics and evaluation

### Hardware Requirements for Training
- **GPU**: Recommended (CUDA, MPS, or ROCm)
- **RAM**: Minimum 8GB, 16GB+ recommended
- **Storage**: 10GB+ for models and datasets

### Deployment Hardware
- **Raspberry Pi 5**: 4GB or 8GB RAM variant
- **Logitech Camera**: As specified in documentation
- **Storage**: MicroSD card (32GB+ recommended)
- **Power Supply**: Official Raspberry Pi power adapter

---

## Timeline Estimate

| Phase | Estimated Time |
|-------|----------------|
| Environment setup | 30 minutes |
| Data exploration & preparation | 1-2 hours |
| Model training (multiple variants) | 2-4 hours |
| Evaluation & analysis | 1-2 hours |
| Model export & optimization | 1 hour |
| Deployment testing | 1-2 hours |
| Documentation | 1 hour |
| **Total** | **7-12 hours** |

*Note: Training time varies based on available hardware and number of epochs*

---

## Success Criteria

### Training Success
- ✅ Model converges without overfitting
- ✅ Validation accuracy > 95%
- ✅ All classes have balanced performance
- ✅ Confusion matrix shows clear diagonal pattern
- ✅ Low confidence predictions < 5%

### Deployment Success
- ✅ Model exports successfully to ONNX
- ✅ Exported model maintains accuracy
- ✅ Inference speed meets real-time requirements
- ✅ Model runs on Raspberry Pi without issues
- ✅ Integration with camera works seamlessly

---

## Next Steps After Notebook Completion

1. **Raspberry Pi Setup**:
   - Install OS and dependencies
   - Configure camera
   - Deploy ONNX model
   - Test inference pipeline

2. **Production Integration**:
   - Integrate with labeling machine
   - Implement error handling
   - Add logging and monitoring
   - Create alert system for quality issues

3. **Continuous Improvement**:
   - Collect production data
   - Retrain model periodically
   - Monitor model drift
   - Update dataset with edge cases

---

## References & Resources

- **Ultralytics Documentation**: https://docs.ultralytics.com/
- **Roboflow Notebooks**: https://github.com/roboflow/notebooks
- **ONNX Runtime**: https://onnxruntime.ai/
- **Raspberry Pi 5 Documentation**: https://www.raspberrypi.com/documentation/
- **YOLOv8 Classification**: https://docs.ultralytics.com/tasks/classify/

---

## Final Notes

This plan provides a comprehensive roadmap for creating a production-ready bottle label classification system. The notebook should be well-documented, reproducible, and optimized for both accuracy and deployment efficiency. Each section should include clear explanations, visualizations, and validation steps to ensure maximum reliability for industrial application.

**Key Focus Areas**:
1. **Accuracy**: Rigorous validation and testing
2. **Speed**: Optimization for real-time processing
3. **Reliability**: Robust error handling and edge case coverage
4. **Maintainability**: Clean code and comprehensive documentation
