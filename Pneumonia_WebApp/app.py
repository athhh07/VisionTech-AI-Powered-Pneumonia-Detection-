import os
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model


# Load Model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "pneumonia_detection_model.h5")

model = load_model(MODEL_PATH)


# Page Configuration

st.set_page_config(
    page_title="VisionTech",
    page_icon="🫁",
    layout="centered"
)


# Header

st.title("🫁 VisionTech")
st.subheader("AI-Powered Pneumonia Detection")

st.write(
    "Upload a chest X-ray image and the trained VGG16 model "
    "will predict whether Pneumonia is present."
)

st.divider()


# Upload Image

uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)


# Prediction

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded X-ray Image",
        use_container_width=True
    )

    # Preprocessing

    img = image.convert("RGB")
    img = img.resize((224, 224))

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Model Prediction

    prediction = model.predict(img_array, verbose=0)

    probability = float(prediction[0][0])

    st.divider()

    st.subheader("Prediction Result")

    st.metric(
        "Pneumonia Probability",
        f"{probability * 100:.2f}%"
    )

    st.progress(probability)

    # Result

    if probability > 0.5:

        st.error("🛑 Pneumonia Detected")

        st.write(
            "The uploaded X-ray shows patterns associated with pneumonia. "
            "Please consult a qualified healthcare professional for diagnosis."
        )

    else:

        st.success("✅ Normal")

        st.write(
            "The uploaded X-ray does not show significant indicators of pneumonia."
        )
