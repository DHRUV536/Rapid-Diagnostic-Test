import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import joblib

# Disable GPU (optional for deployment)
tf.config.set_visible_devices([], 'GPU')

# Load models once
st.session_state.yolo_model = YOLO("Saved Models/yolov8n.pt")
st.session_state.feature_extractor = tf.keras.models.load_model("Saved Models/mobilenetv2_feature_extractor.h5")
st.session_state.svm = joblib.load("Saved Models/svm_model.joblib")
class_names = ['positive', 'negative', 'faded']

def extract_features_from_array(img_array):
    img = cv2.resize(img_array, (224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = st.session_state.feature_extractor.predict(x, verbose=0)
    return features.flatten()

# Streamlit UI
st.title("Rapid Diagnostic Test Kit Classification")
st.markdown("Upload an image of your test kit. The model will detect and classify the result.")

uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    img_pil = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img_pil)
    st.image(img_np, caption="Uploaded Image", use_container_width=True)

    results = st.session_state.yolo_model(img_np)[0]
    
    if not results.boxes:
        st.warning("No test kit detected.")
    else:
        st.subheader("Detection Results")

        curr = 0 
        for i, box in enumerate(results.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cropped = img_np[y1:y2, x1:x2]
            if cropped.size == 0:
                st.info(f"Skipped box {i+1} (empty crop)")
                continue
            st.info(f"Detected Object : {curr}")
            features = extract_features_from_array(cropped)
            pred = st.session_state.svm.predict([features])[0]
            proba = st.session_state.svm.predict_proba([features])[0]
            st.image(cropped, caption=f"Detected Kit {i+1}", use_container_width=True,width=100)
            if pred==2:
                st.warning(f"Faded or Invalid Bounding Box")
            else:
                st.success(f"Prediction: **{class_names[pred]}** (confidence: {proba[pred]:.2f})")
