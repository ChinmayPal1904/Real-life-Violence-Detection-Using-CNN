# Tech Stack - Real-life Violence Detection Web Application

## Backend Framework
- **Flask** (v3.0.0) - Python web framework
- **Flask-Login** (v0.6.3) - User authentication and session management
- **Werkzeug** - Password hashing and security utilities

## Machine Learning & AI
- **TensorFlow** (v2.13.0) - Deep learning framework
- **Keras** - High-level neural networks API (integrated with TensorFlow)
- **MobileNetV2** - Pre-trained convolutional neural network architecture for violence detection
- **scikit-learn** (v1.3.2) - Machine learning utilities

## Computer Vision & Video Processing
- **OpenCV** (opencv-python v4.9.0.80) - Computer vision library for video frame extraction and processing
- **NumPy** (v1.24.4) - Numerical computing for array operations

## Data Analysis & Visualization
- **Pandas** (v2.1.4) - Data manipulation and analysis
- **Matplotlib** (v3.8.2) - Data visualization and plotting
- **Seaborn** (v0.13.2) - Statistical data visualization

## Frontend Technologies
- **HTML5** - Markup language
- **CSS3** - Styling (custom CSS with CSS Grid, Flexbox, animations)
- **JavaScript (Vanilla)** - Client-side scripting (no framework dependencies)
- **Jinja2** - Template engine (integrated with Flask)

## Frontend Features
- **Google Fonts (Inter)** - Typography
- **CSS Animations** - Custom animations (fade-in, slide-up, shimmer effects)
- **Responsive Design** - Mobile-first approach with CSS Grid and Flexbox
- **Glassmorphism UI** - Modern glass-like visual effects

## Development & Tools
- **Python** - Programming language
- **Jupyter Notebook** - Interactive development environment (for ML model development)
- **Threading** - Multi-threaded video processing

## File Formats & Storage
- **Video Formats**: MP4, AVI, MOV, MKV
- **Image Formats**: JPG/JPEG (for snapshots)
- **Data Formats**: JSON (reports), CSV (reports)

## APIs Used

### Internal REST API Endpoints (Flask)
- **`POST /api/upload`** - Upload video file for processing, returns job ID
- **`GET /api/status/<job_id>`** - Get real-time processing status and progress

### External APIs & CDN Services
- **Google Fonts API** - Loads Inter font family (https://fonts.googleapis.com)
- **CDN.js (jsdelivr)** - Loads Chart.js library for data visualization (https://cdn.jsdelivr.net)

### Library APIs
- **TensorFlow/Keras API** - For ML model operations (MobileNetV2, model prediction)
- **OpenCV API** - For video capture, frame processing, and video writing
- **Flask API** - Web framework routing and request handling

## Architecture & Patterns
- **MVC Pattern** - Model-View-Controller architecture
- **RESTful API** - API endpoints for video upload and status checking
- **Background Processing** - Asynchronous video processing using threading
- **In-memory Job Store** - Thread-safe job management

## Deployment
- **Development Server** - Flask built-in development server
- **Host**: 0.0.0.0 (configurable)
- **Port**: 5000 (default)

## Security
- **Password Hashing** - Werkzeug's password hashing
- **Session Management** - Flask-Login session handling
- **Authentication** - Single-user authentication system

## Additional Libraries
- **UUID** - Unique identifier generation for jobs
- **JSON** - Data serialization
- **CSV** - Report generation
- **Time** - Timestamp management
- **OS** - File system operations

