# ML Model Process - 8 Key Points

## 1. Data Collection & Preparation
- **Dataset Source**: Kaggle dataset "Real Life Violence Situations Dataset"
- **Data Structure**: Two folders - "Violence" and "NonViolence" containing video files
- **Video Processing**: 500 videos processed from each class (total 1,000 videos)
- **Frame Extraction Strategy**: 1 frame extracted every 10 frames (frame_extract_rate = 10) to reduce redundancy
- **Total Frames Extracted**: 13,801 frames (11,040 training + 2,761 test)
- **Preprocessing Steps**:
  - Resize frames to (224, 224) pixels
  - Convert BGR to RGB color space
  - Normalize pixel values to [0, 1] range (float32 division by 255.0)
- **Data Format**: NumPy arrays with shape (n_samples, 224, 224, 3)

## 2. Transfer Learning with MobileNetV2
- **Base Architecture**: MobileNetV2 (pre-trained on ImageNet dataset)
- **Transfer Learning Strategy**: Uses pre-trained ImageNet weights for feature extraction
- **Input Shape**: (224, 224, 3) - RGB images
- **Base Model Configuration**: 
  - `weights="imagenet"` - Loads ImageNet pre-trained weights
  - `include_top=False` - Excludes top classification layer
  - `input_shape=(224, 224, 3)` - Fixed input dimensions
- **Layer Freezing**: All base MobileNetV2 layers are frozen (trainable=False) during initial training
- **Advantages**: Leverages learned features from ImageNet, reduces training time, improves accuracy with limited data

## 3. Custom Classification Head & Data Augmentation
- **Custom Head Architecture**: 
  - GlobalAveragePooling2D: Reduces spatial dimensions to 1D feature vector
  - Dense(256, activation='relu'): Fully connected layer with 256 units
  - Dropout(0.3): 30% dropout for regularization to prevent overfitting
  - Dense(1, activation='sigmoid'): Binary classification output (violence/non-violence)
- **Data Augmentation**: Applied during training to improve generalization
  - Rotation range: 12 degrees
  - Zoom range: 0.15 (15% zoom in/out)
  - Width shift: 0.1 (10% horizontal shift)
  - Height shift: 0.1 (10% vertical shift)
  - Horizontal flip: Enabled
- **Model Compilation**:
  - Optimizer: Adam with learning rate 1e-4 (0.0001)
  - Loss Function: Binary cross-entropy
  - Metrics: Accuracy
- **Output**: Single probability value between 0 and 1 (sigmoid activation)

## 4. Model Training Process
- **Training Platform**: Kaggle (GPU: Tesla P100-PCIE-16GB with 15.5GB memory)
- **Train/Test Split**: 
  - 80% training (11,040 samples) 
  - 20% test (2,761 samples)
  - Stratified split to maintain class balance
- **Training Configuration**:
  - Epochs: 15 (with early stopping)
  - Batch size: 32
  - Data augmentation applied during training via ImageDataGenerator
- **Training Callbacks**:
  - **EarlyStopping**: Monitors validation loss, patience=5, restores best weights
  - **ReduceLROnPlateau**: Reduces learning rate by factor 0.2 when validation loss plateaus (patience=2)
  - **ModelCheckpoint**: Saves best model weights to `mobilenetv2_violence.weights.h5` based on validation loss
- **Training Results**:
  - Final test accuracy: **97.75%**
  - Final test loss: 0.0680
  - Training time: ~98-121 seconds per epoch
- **Performance Metrics**: 
  - Precision: 0.98 (Violence), 0.97 (NonViolence)
  - Recall: 0.97 (Violence), 0.98 (NonViolence)
  - F1-score: 0.98 (both classes)

## 5. Model Evaluation & Metrics
- **Test Set Evaluation**: 
  - Test accuracy: 97.75% (2,761 samples)
  - Test loss: 0.0680
- **Confusion Matrix**: 
  - True Negatives (NonViolence): 1,243 (correctly predicted as non-violent)
  - False Positives: 26 (non-violent predicted as violent)
  - False Negatives: 45 (violent predicted as non-violent)
  - True Positives (Violence): 1,447 (correctly predicted as violent)
- **Classification Report Metrics**:
  - NonViolence: Precision=0.97, Recall=0.98, F1=0.98
  - Violence: Precision=0.98, Recall=0.97, F1=0.98
  - Overall accuracy: 0.98
- **ROC Curve Analysis**: 
  - AUC (Area Under Curve) calculated for binary classification performance
  - Plots False Positive Rate vs True Positive Rate
- **Visualization**: Training curves (accuracy/loss), confusion matrix, and ROC curve generated

## 6. Video Processing Pipeline
- **Frame Extraction**: OpenCV extracts frames from input video at original FPS
- **Frame Sampling**: Optional frame sampling (e.g., every 10th frame) for faster processing
- **Frame-by-Frame Analysis**: Each frame is processed independently
- **Preprocessing per Frame**:
  - Convert BGR to RGB color space
  - Resize to (224, 224)
  - Normalize pixel values to [0, 1] (float32 division by 255.0)
  - Expand dimensions for batch prediction: `np.expand_dims(img, axis=0)`
- **Inference**: Model predicts violence probability for each frame
- **Video Prediction Function**: 
  - Processes frames with configurable frame step
  - Calculates mean prediction score across all frames
  - Final label: "Violence" if mean score > 0.5, else "Non-Violence"
- **Threshold**: Default 0.5 (configurable) - scores above threshold classified as violence

## 7. Real-time Inference & Annotation
- **Model Loading**: Lazy loading with thread-safe singleton pattern
- **Prediction**: Single frame prediction using `model.predict()` with verbose=0
- **Visual Annotation**: OpenCV adds labels to frames:
  - Green "SAFE" label for non-violence (score < threshold)
  - Red "VIOLENCE" label with confidence score for violence (score ≥ threshold)
- **Video Output**: Annotated frames compiled into output video with original FPS and resolution
- **Performance**: Optimized for real-time processing with minimal latency

## 8. Deployment & Integration
- **Model Storage**: Pre-trained weights loaded from `mobilenetv2_violence.weights.h5`
- **Framework Integration**: TensorFlow/Keras model integrated into Flask web application
- **Background Processing**: Asynchronous video processing using Python threading
- **Thread Safety**: Model access protected with locks for concurrent requests
- **Scalability**: Model loaded once and reused for multiple video predictions
- **API Endpoints**: RESTful API for video upload and status checking
- **Error Handling**: Graceful error handling with status updates and error messages
- **Post-processing & Analysis**:
  - Segment summarization: Groups consecutive violent frames into time segments
  - Minimum duration filter: Segments shorter than 0.5 seconds are filtered out
  - Metrics calculation: Total duration, violence duration, violence ratio, max confidence
  - Report generation: JSON and CSV reports with detailed segment information

---

## Model Architecture Summary

```
Input: (224, 224, 3) RGB Image
    ↓
MobileNetV2 Base (Pre-trained, frozen)
    ↓
GlobalAveragePooling2D
    ↓
Dense(256, ReLU)
    ↓
Dropout(0.3)
    ↓
Dense(1, Sigmoid)
    ↓
Output: Probability [0, 1]
```

## Key Technical Details

- **Framework**: TensorFlow 2.13.0 with Keras
- **Training Hardware**: Kaggle GPU (Tesla P100-PCIE-16GB, 15.5GB memory)
- **Model Size**: Lightweight MobileNetV2 architecture for efficient inference
- **Training Time**: ~98-121 seconds per epoch (15 epochs with early stopping)
- **Final Test Accuracy**: 97.75%
- **Inference Speed**: Optimized for real-time video processing (~23ms per batch on GPU)
- **Memory Management**: Singleton pattern prevents multiple model instances
- **Threshold Configuration**: Adjustable detection threshold (0.0 - 1.0) via settings page
- **Data Augmentation**: Applied during training to improve model generalization
- **Evaluation Metrics**: Confusion matrix, classification report, ROC curve, and AUC score

## Training Summary

- **Total Training Samples**: 11,040 frames
- **Total Test Samples**: 2,761 frames
- **Training Strategy**: Transfer learning with frozen base layers + data augmentation
- **Optimization**: Adam optimizer (learning rate: 1e-4) with adaptive learning rate reduction
- **Regularization**: Dropout (0.3) + Early stopping + Learning rate reduction
- **Final Performance**: 97.75% accuracy with balanced precision and recall across both classes

