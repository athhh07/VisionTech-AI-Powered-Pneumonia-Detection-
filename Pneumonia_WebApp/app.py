import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("pneumonia_detection_model.h5")

# Title
st.title("VisionTech")
st.write("VGG16 Based Model for Pneumonia Prediction")

# Upload Image
uploaded_file = st.file_uploader("Choose an X-ray Image to Processed...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    # Show uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-ray Image", use_column_width=True)

    # Preprocess image
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0

    # Convert grayscale to RGB if needed
    if img_array.shape[-1] != 3:
        img_array = np.stack((img_array,)*3, axis=-1)

    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)

    # Output result
    if prediction[0][0] > 0.5:
        st.error("🛑 Prediction: PNEUMONIA Detected")
    else:
        st.success("✅ Prediction: NORMAL")

    st.write("Prediction Confidence:", prediction[0][0])
