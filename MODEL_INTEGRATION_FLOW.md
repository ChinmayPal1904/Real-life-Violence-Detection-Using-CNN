# Model Integration & Flow - 5 Key Points

---

## 1. Model Loading & Architecture
```
MobileNetV2 Base → Custom Head → Singleton Pattern
```
- **Architecture**: MobileNetV2 (pre-trained) + Custom classification head
- **Loading**: Lazy-loaded singleton with thread-safe locking
- **Storage**: Weights loaded once from `mobilenetv2_violence.weights.h5`

---

## 2. Video Processing Pipeline
```
Frame Extraction → Preprocessing → Model Inference → Annotation → Output
```
- Frame-by-frame extraction using OpenCV
- Preprocessing: Resize (224×224) → Normalize (÷255) → Batch dimension
- Real-time inference with visual annotation on frames

---

## 3. Background Processing
```
Upload → Background Thread → Job Tracking → Progress Updates
```
- Asynchronous processing via `threading.Thread`
- Job status tracking: queued → processing → completed
- Progress updates every 10 frames with percentage completion

---

## 4. Classification & Threshold
```
Sigmoid Output → Threshold Comparison → Visual Labels → Segment Detection
```
- Binary classification: probability > 0.5 = "VIOLENCE"
- Visual annotation: Red "VIOLENCE" / Green "SAFE" labels
- Segment grouping for consecutive violent frames

---

## 5. Output & Reporting
```
Annotated Video → JSON/CSV Reports → Job History → Downloadable Results
```
- Annotated video saved to `web_outputs/`
- Dual report formats: JSON (detailed) + CSV (timeline)
- Job history tracking with downloadable results via Flask routes

---


