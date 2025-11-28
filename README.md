
# Aerial Bird vs Drone Classifier  
**Real-time detection with 97.67% accuracy**  
Trained and fine-tuned ResNet50V2 model for distinguishing birds from drones in aerial imagery.


---

### Overview
This project implements a high-accuracy binary classifier capable of distinguishing birds from drones in real-world aerial images. The final model achieves **97.67% test accuracy** and **97.33% F1-score** using a fine-tuned ResNet50V2 backbone.

The Streamlit web application supports:
- Single image upload
- Real-time webcam inference
- Confidence scoring and latency display
- Production-ready deployment

---

### Model Performance (Final Leaderboard)

| Model                | Test Acc | F1-Score | Inference Time (ms/img) | Parameters |
|----------------------|----------|----------|--------------------------|------------|
| **ResNet50V2 (fine-tuned)** | **97.67%** | **97.33%** | ~5.9 ms (CPU) | ~23.6 M |
| CustomCNN            | 83.26%   | 83.84%   | ~18.5 ms                | ~5.2 M    |
| EfficientNetB0       | 78.60%   | 74.16%   | ~40 ms                  | ~4.0 M    |
| MobileNetV3-Small    | 66.98%   | 68.16%   | ~16 ms                  | ~0.9 M    |

**ResNet50V2** is the champion model used in the deployed app.

---

### Project Structure
```
.
├── app.py                        → Streamlit web application
├── requirements.txt              → Dependencies for deployment
├── models/
│   └── champion_resnet50v2_weights.h5    → Final fine-tuned weights (.h5)
│   └── champion_resnet50v2_savedmodel/   → SavedModel format (for export)
└── README.md
```

---

### Local Setup & Running

```bash
# 1. Clone the repository
git clone https://github.com/your-username/aerial-bird-vs-drone.git
cd aerial-bird-vs-drone

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate    # Linux/Mac
# or
venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app will launch at `http://localhost:8501`

---

### Deployment

This app is fully compatible with **Streamlit Community Cloud** (free tier).

1. Push your code to a public GitHub repository
2. Go to [https://share.streamlit.io](https://share.streamlit.io)
3. Click "New app" → connect your repo → select `app.py`
4. Deploy!

No server management required.

---

### Export for Edge Devices (Optional)

```bash
# Convert to ONNX
python -m tf2onnx.convert --saved-model models/champion_resnet50v2_savedmodel \
    --output resnet50v2_champion.onnx --opset 16

# Convert to TensorRT (NVIDIA Jetson, drones, edge GPUs)
trtexec --onnx=resnet50v2_champion.onnx \
    --saveEngine=resnet50v2_champion.trt --fp16
```

Achieves **>170 FPS** on modern NVIDIA hardware.

---

### Requirements
See [`requirements.txt`](requirements.txt) for exact versions (tested on Python 3.10–3.11).

---


**Built with TensorFlow 2.15 • Keras • Streamlit • OpenCV**

For questions or contributions, open an issue or submit a pull request.
``` 
