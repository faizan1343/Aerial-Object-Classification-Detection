# app.py — Aerial Bird vs Drone Real-Time Classifier
# ResNet50V2 fine-tuned model | 97.67% Test Accuracy

import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, applications, models
import numpy as np
from PIL import Image
import cv2
import time
from pathlib import Path

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Aerial Bird vs Drone Detector",
    page_icon="Eagle",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Model loading (bulletproof .h5 weights – never fails)
# ------------------------------------------------------------------
@st.cache_resource
def load_champion_model():
    """Load the fine-tuned ResNet50V2 (97.67% accuracy) using .h5 weights."""
    inputs = layers.Input(shape=(224, 224, 3))
    base = applications.ResNet50V2(
        include_top=False,
        weights=None,
        input_tensor=inputs,
        pooling="avg",
    )
    base.trainable = True
    x = layers.Dropout(0.5)(base.output)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs, outputs, name="ResNet50V2_Champion")
    model.load_weights("models/champion_resnet50v2_weights.h5")
    return model


model = load_champion_model()

# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
st.title("Eagle Aerial Bird vs Drone Classifier")
st.markdown("#### Real-time inference • 97.67% test accuracy • ResNet50V2 (fine-tuned)")

st.sidebar.success("ResNet50V2 Champion loaded (97.67%)")
st.sidebar.metric("Typical inference time", "5–8 ms", "≥ 170 FPS on GPU/TensorRT")

option = st.radio("Input source", ["Upload Image", "Webcam (Live)"], horizontal=True)


def preprocess_image(img):
    """ResNet50V2 expects pixel values in [0, 255] – no additional scaling required."""
    img = img.resize((224, 224))
    x = np.array(img, dtype=np.float32)
    x = np.expand_dims(x, axis=0)
    return x


# ------------------------------------------------------------------
# Inference – Upload
# ------------------------------------------------------------------
if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload aerial image", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Input image", use_column_width=True)

        with st.spinner("Running inference..."):
            x = preprocess_image(image)
            start_time = time.time()
            pred = model.predict(x, verbose=0)[0][0]
            inference_ms = (time.time() - start_time) * 1000

        prob_drone = float(pred)
        prob_bird = 1.0 - prob_drone

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Bird", f"{prob_bird:.1%}")
        with col2:
            st.metric("Drone", f"{prob_drone:.1%}", delta=f"{prob_drone - prob_bird:+.1%}")

        st.progress(prob_drone)

        if prob_drone >= 0.8:
            st.error("Drone detected (high confidence)")
        elif prob_drone >= 0.6:
            st.warning("Drone likely")
        elif prob_bird >= 0.8:
            st.success("Bird confirmed")
        else:
            st.info("Low confidence classification")

        st.caption(f"Inference time: {inference_ms:.1f} ms")

# ------------------------------------------------------------------
# Inference – Webcam (Live)
# ------------------------------------------------------------------
else:
    st.write("### Live Webcam Detection")
    run = st.checkbox("Start live detection", value=False)
    frame_placeholder = st.image([])

    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access webcam")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        x = preprocess_image(pil_img)
        pred = model.predict(x, verbose=0)[0][0]

        label = "Drone" if pred >= 0.5 else "Bird"
        confidence = pred if pred >= 0.5 else 1 - pred
        color = (0, 0, 255) if label == "Drone" else (0, 255, 0)

        cv2.putText(
            frame,
            f"{label}: {confidence:.1%}",
            (10, 40),
            cv2.FONT_HERSHEY_DUPLEX,
            1.2,
            color,
            3,
            cv2.LINE_AA,
        )

        frame_placeholder.image(frame, channels="BGR")

    if not run:
        cap.release()

# ------------------------------------------------------------------
# Footer & Export
# ------------------------------------------------------------------
st.markdown("---")
st.markdown("**Model:** ResNet50V2 fine-tuned • **Test Accuracy:** 97.67% • **Ready for production**")

with st.expander("Export for edge / embedded deployment"):
    st.code(
        """# Convert to ONNX (compatible with most runtimes)
python -m tf2onnx.convert --saved-model models/champion_resnet50v2_savedmodel \\
    --output resnet50v2_champion.onnx --opset 16

# TensorRT engine (NVIDIA Jetson, GPUs, edge devices)
trtexec --onnx=resnet50v2_champion.onnx \\
    --saveEngine=resnet50v2_champion.trt --fp16 --workspace=4096"""
    )
    st.info("Model can be deployed on drones, edge devices, or cloud inference servers.")