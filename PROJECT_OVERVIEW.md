# Real-life Violence Detection Web Application

## 🔧 Technology Stack

### Backend
| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.x | Programming language |
| **Flask** | 3.0.0 | Web framework |
| **Flask-Login** | 0.6.3 | User authentication |
| **Werkzeug** | - | Security & password hashing |

### Machine Learning & AI
| Technology | Version | Purpose |
|------------|---------|---------|
| **TensorFlow** | 2.13.0 | Deep learning framework |
| **Keras** | - | High-level neural network API |
| **MobileNetV2** | - | Pre-trained CNN architecture |
| **scikit-learn** | 1.3.2 | ML utilities |

### Computer Vision
| Technology | Version | Purpose |
|------------|---------|---------|
| **OpenCV** | 4.9.0.80 | Video processing & frame extraction |
| **NumPy** | 1.24.4 | Numerical computing |

### Data Analysis
| Technology | Version | Purpose |
|------------|---------|---------|
| **Pandas** | 2.1.4 | Data manipulation |
| **Matplotlib** | 3.8.2 | Data visualization |
| **Seaborn** | 0.13.2 | Statistical visualization |

### Frontend
| Technology | Version | Purpose |
|------------|---------|---------|
| **HTML5** | - | Markup |
| **CSS3** | - | Styling & animations |
| **JavaScript** | ES6+ | Client-side scripting |
| **Jinja2** | - | Template engine |
| **Chart.js** | - | Data visualization (via CDN) |

### External Services
| Service | Purpose |
|---------|---------|
| **Google Fonts** | Inter font family |
| **CDN.js (jsdelivr)** | Chart.js library delivery |

---

## 📡 API Endpoints

### Internal REST API

#### Upload Video
```
POST /api/upload
Content-Type: multipart/form-data
Body: { video: File }
Response: { "job_id": "uuid" }
```

#### Get Status
```
GET /api/status/<job_id>
Response: {
  "status": "queued|processing|completed|error",
  "progress": 0-100,
  "message": "string",
  "result": {...}
}
```

---

## 🏗️ Architecture

- **Pattern**: MVC (Model-View-Controller)
- **Processing**: Asynchronous background threading
- **Storage**: In-memory job store (thread-safe)
- **Authentication**: Single-user session-based
- **File Storage**: Local filesystem (uploads/, web_outputs/, reports/)

---

## 📁 Project Structure

```
├── app.py                 # Flask application
├── requirements.txt       # Python dependencies
├── mobilenetv2_violence.weights.h5  # ML model weights
├── templates/            # HTML templates (Jinja2)
│   ├── base.html
│   ├── dashboard.html
│   ├── login.html
│   ├── result.html
│   └── ...
├── static/               # Static assets
│   ├── app.js
│   └── style.css
├── uploads/              # Uploaded videos
├── web_outputs/          # Processed videos
└── reports/              # JSON & CSV reports
```

---

## 🚀 Features

- ✅ Video upload (MP4, AVI, MOV, MKV)
- ✅ Real-time processing status
- ✅ Frame-by-frame violence detection
- ✅ Visual timeline of violent segments
- ✅ Downloadable processed video & CSV reports
- ✅ User authentication & session management
- ✅ Dashboard with statistics
- ✅ Alert system for violent content
- ✅ Processing history

---

## 🔐 Security

- Password hashing (Werkzeug)
- Session-based authentication (Flask-Login)
- Login-required endpoints
- Secure file upload validation

---

## 📦 Deployment

- **Server**: Flask development server
- **Host**: 0.0.0.0 (configurable)
- **Port**: 5000 (default)
- **Environment**: Python 3.x

---

## 🎯 Key Workflows

1. **Upload** → User uploads video via `/api/upload`
2. **Process** → Background thread processes video frame-by-frame
3. **Detect** → MobileNetV2 model analyzes each frame
4. **Annotate** → OpenCV adds labels to frames
5. **Report** → Generate JSON & CSV reports
6. **Display** → Show results with timeline & charts

---

## 📊 Data Flow

```
Video Upload → Background Thread → Frame Extraction → 
ML Inference → Annotation → Video Output → Report Generation
```


