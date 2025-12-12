# 🦺 SafetyVision AI - PPE Compliance Detection System

**Oil & Gas + Utilities AI Hackathon - December 2025**

An intelligent PPE (Personal Protective Equipment) compliance detection system that uses computer vision and AI to automatically identify safety violations and generate actionable recommendations.

---

## 🎯 Overview

SafetyVision AI leverages YOLOv8 object detection and Google Gemini AI to provide real-time hard hat compliance monitoring at construction sites, oil & gas facilities, and utility operations. The system detects workers, analyzes whether they're wearing hard hats, and generates comprehensive safety reports with AI-powered recommendations.

### Key Features

- **Real-time Detection**: Upload images or use webcam for live compliance checking
- **AI-Powered Analysis**: Google Gemini generates contextual safety recommendations
- **Annotated Visualizations**: Color-coded bounding boxes (Green = Compliant, Red = Violation)
- **PDF Report Generation**: Professional safety reports with timestamps and recommendations
- **Multi-color Hardhat Support**: Detects yellow, orange, red, blue, green, and white hardhats
- **Detailed Metrics**: Compliance rates, violation counts, and worker statistics

---

## 🛠️ Tech Stack

| Component                    | Technology              |
| ---------------------------- | ----------------------- |
| **Object Detection**   | YOLOv8 (Ultralytics)    |
| **AI Recommendations** | Google Gemini 2.5 Flash |
| **Web Framework**      | Streamlit               |
| **Computer Vision**    | OpenCV, NumPy           |
| **PDF Generation**     | ReportLab               |
| **Language**           | Python 3.12             |

---

## 📋 Installation

### Prerequisites

- Python 3.12 or higher
- Virtual environment (recommended)
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))

### Setup Instructions

1. **Clone the repository**

   ```bash
   git clone https://github.com/emresenDEV/PPE_Vision_Detector.git
   cd PPE_Vision_Detector
   ```
2. **Create and activate virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate  # On Windows
   ```
3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```
4. **Configure API Key**

   Create a `.env` file in the project root:

   ```bash
   GEMINI_API_KEY=your_api_key_here
   ```

   **Alternative**: Create `.streamlit/secrets.toml`:

   ```toml
   GEMINI_API_KEY = "your_api_key_here"
   ```

---

## 🚀 Usage

### Running the Application

```bash
source venv/bin/activate  # Activate virtual environment
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

### Workflow

1. **Select Input Method**

   - Choose "Upload Image" or "Use Webcam"
2. **Capture/Upload Photo**

   - Upload an image file (JPG, PNG, WEBP)
   - Or take a live photo using your webcam
3. **Analyze Compliance**

   - Click "🚀 Analyze PPE Compliance"
   - View real-time detection results
4. **Review Results**

   - See compliance status, metrics, and annotated image
   - Read AI-generated safety recommendations
5. **Generate Report**

   - Click "📥 Generate PDF Report"
   - Download comprehensive safety report

---

## 🔍 How It Works

### Detection Pipeline

1. **Person Detection** (YOLOv8)

   - Uses pre-trained COCO model to detect people
   - Confidence threshold: 0.5
2. **Hardhat Detection** (Color-based Heuristic)

   - Analyzes top 30% of person bounding box (head region)
   - Uses HSV color space for robust color detection
   - Detects vibrant colors: Yellow, Orange, Red, Blue, Green, White
   - **Skin tone exclusion**: Prevents false positives from faces/hands
   - **High saturation requirement**: Ensures only bright safety colors are matched
   - Threshold: 12% of head region must be hardhat-colored
3. **Compliance Classification**

   - Calculates: Total people, compliant workers, violations
   - Determines overall status: COMPLIANT vs VIOLATION DETECTED
   - Generates compliance rate percentage
4. **AI Recommendations** (Google Gemini 2.5 Flash)

   - Analyzes compliance data and violation patterns
   - Generates 3 specific, actionable safety recommendations
   - Prioritizes by safety impact
   - Falls back to rule-based recommendations if API unavailable

### Color Detection Logic

```python
Hardhat Colors Detected:
├─ Yellow:  H=18-35,  S≥120, V≥120  (High saturation = vibrant)
├─ Orange:  H=5-18,   S≥120, V≥120
├─ Red:     H=0-5, 160-180, S≥120, V≥120
├─ Blue:    H=90-130, S≥120, V≥120
├─ Green:   H=40-85,  S≥120, V≥120
└─ White:   H=any,    S≤30,  V≥200  (High brightness)

Excluded:
└─ Skin tones: H=0-25, S=20-170 (Prevents false positives)
```

---

## ⚠️ Current Limitations (MVP/Demo)

This is a **hackathon MVP** with the following limitations:

1. **Color-based Detection**:

   - Uses heuristic approach instead of trained model
   - May have false positives/negatives in poor lighting
   - Designed for demo purposes only
2. **Hard Hat Only**:

   - Currently only detects hard hats
   - Does not detect other PPE (gloves, vests, goggles)
3. **Static Images**:

   - Processes individual images/frames
   - Not real-time video stream processing

---

## 🚀 Future Improvements (Production)

### Custom YOLOv8 Model Training

For production deployment, train a custom YOLOv8 model on the **Roboflow Hardhat Detection Dataset**:

- **Dataset**: [Hardhat Detection by Michael](https://universe.roboflow.com/michael-8jeqe/hardhat-detection-iukt9)
- **Size**: 2,800+ annotated images
- **Training Time**: 1-2 hours on GPU
- **Expected Accuracy**: 93%+ mAP
- **Classes**: hardhat, no-hardhat, person

**Training Command**:

```python
from ultralytics import YOLO

# Load base model
model = YOLO('yolov8n.pt')

# Train on custom dataset
model.train(
    data='hardhat-detection.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)
```

### Additional Enhancements

- [ ] Multi-PPE detection (vests, gloves, goggles)
- [ ] Real-time video stream processing
- [ ] Integration with site access control systems
- [ ] Historical analytics dashboard
- [ ] Mobile app deployment
- [ ] Edge device deployment (NVIDIA Jetson)

---

## 📊 Debug Output

The system provides detailed console output for troubleshooting:

```
Running detection on: /tmp/tmpXYZ.jpg
Found 3 detections

Person 1:
  Head region analysis - Hardhat colors: 28.5%, Skin: 15.2%
    Yellow: 27.8%, Orange: 0.5%, Red: 0.0%
    Blue: 0.0%, Green: 0.0%, White: 0.2%
  ✓ HARDHAT DETECTED (28.5% >= 12%)

Person 2:
  Head region analysis - Hardhat colors: 3.2%, Skin: 45.3%
    Yellow: 0.0%, Orange: 0.0%, Red: 0.0%
    Blue: 0.0%, Green: 0.0%, White: 3.2%
  ✗ NO HARDHAT (3.2% < 12%)
```

---

## 🎨 Fine-Tuning Detection

### Adjust Sensitivity

Edit `model/detector.py` line 199 to adjust detection threshold:

**More lenient** (detect more hardhats, may increase false positives):

```python
threshold = 10.0  # Was 12.0
```

**More strict** (fewer false positives, may miss some hardhats):

```python
threshold = 15.0  # Was 12.0
```

### Adjust Color Saturation

Edit saturation values in `model/detector.py` lines 142-161:

**More lenient** (detect faded/dirty hardhats):

```python
yellow_lower = np.array([18, 100, 100])  # S>=100 instead of S>=120
```

**More strict** (only very vibrant colors):

```python
yellow_lower = np.array([18, 140, 140])  # S>=140 instead of S>=120
```

---

## 📁 Project Structure

```
PPE_Vision_Detector/
├── streamlit_app.py           # Main Streamlit application
├── model/
│   ├── detector.py            # YOLOv8 detection + hardhat logic
│   └── __init__.py
├── utils/
│   ├── visualization.py       # Annotated image generation
│   ├── pdf_generator.py       # PDF report creation
│   ├── gemini_recommendations.py  # AI recommendations
│   └── __init__.py
├── .env                       # API key configuration (not in git)
├── requirements.txt           # Python dependencies
├── .streamlit/
│   └── config.toml           # Streamlit configuration
└── README.md                 # This file
```

---

## 🔧 Troubleshooting

### Issue: "No secrets found" error

**Solution**: Ensure `.env` file exists in project root with:

```
GEMINI_API_KEY=your_api_key_here
```

### Issue: Detection showing all people as compliant

**Solution**: Check console output for debug info. If saturation is too low, people may be detected. Increase threshold in `detector.py` line 199.

### Issue: Gemini API error

**Solution**:

1. Verify API key is correct in `.env`
2. Check you have API quota remaining
3. System will fall back to rule-based recommendations

### Issue: Poor detection accuracy

**Solution**:

1. Ensure good lighting conditions
2. Hardhats should be bright, vibrant colors
3. Adjust threshold in `detector.py` line 199
4. Check console output to see what colors are being detected

---

## 👤 Author

**Monica Nieckula**
Oil & Gas + Utilities AI Hackathon - December 2025

---

## 📄 License

This project is developed for the Oil & Gas + Utilities AI Hackathon 2025.

---

## 🙏 Acknowledgments

- **YOLOv8**: Ultralytics for state-of-the-art object detection
- **Google Gemini**: AI-powered safety recommendations
- **Streamlit**: Rapid web app development framework
- **Roboflow**: Hardhat detection dataset reference
- **OpenCV**: Computer vision operations

---

## 📞 Support

For issues or questions:

1. Check the Troubleshooting section above
2. Review console debug output
3. Adjust thresholds as needed for your environment

---

**Built with ❤️ for safer workplaces (due to time constrains, Claude Ai was used to assist)**
