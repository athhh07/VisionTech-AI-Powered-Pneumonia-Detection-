<div align="center">

# 🩻 VisionTech — AI-Powered Pneumonia Detection

### Deep Learning–powered chest X-ray classification to support faster, more consistent pneumonia diagnosis

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Model](https://img.shields.io/badge/Model-CNN%20(ResNet--ready)-blue)](#-model-details)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-~95%25-success)](#-results)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](#-license)

**[🚀 Live Demo](#-live-demo) · [📦 Installation](#-installation) · [📊 Results](#-results) · [🧩 How It Works](#-how-it-works)**

</div>

---

## 📋 Overview

**VisionTech** is a deep learning–based system that detects **pneumonia from chest X-ray images** using a Convolutional Neural Network (CNN). It automates the classification process — **Pneumonia vs. Normal** — to support faster and more consistent preliminary screening, reducing manual workload for radiologists.



---

## 🚀 Live Demo

🔗 **[Try the app here](#)** : *https://visiontech.streamlit.app/*

---

## ✨ Key Features

- 🫁 **Pneumonia vs. Normal classification** from chest X-rays
- 🧠 **CNN-based image processing pipeline**
- 🔁 **End-to-end ML pipeline** — preprocessing, training, evaluation, inference
- ⚡ **Fast prediction** on new, unseen X-ray images
- 🔌 **Easily extendable** for web deployment and additional disease classes

---

## 🧩 How It Works

```
Chest X-ray Image Input
        │
        ▼
Preprocessing (Resize → Normalize)
        │
        ▼
CNN Model (Conv + Pool + Dense layers)
        │
        ▼
Sigmoid Output → [Normal | Pneumonia]
        │
        ▼
Prediction + Confidence Score
```

**Pipeline steps:**
1. **Preprocess** X-ray images (resize, normalize pixel values)
2. **Train** the CNN model on labeled chest X-ray data
3. **Validate** model performance on held-out data
4. **Predict** on unseen X-ray images

---

## 🏗️ Model Details

| Attribute | Detail |
|---|---|
| **Architecture** | Convolutional Neural Network (CNN) |
| **Framework** | TensorFlow / Keras |
| **Loss Function** | Binary Crossentropy |
| **Optimizer** | Adam |
| **Metrics Tracked** | Accuracy, Precision, Recall |
| **Task Type** | Binary image classification |

---

## 📊 Results

### Test Accuracy: **~95%**

- Strong generalization observed on unseen chest X-ray images
- Balanced performance across precision and recall metrics


---

## 🛠️ Tech Stack

- **Language:** Python
- **Deep Learning:** TensorFlow, Keras
- **Data Handling:** NumPy, Pandas
- **Visualization & Image Processing:** Matplotlib, OpenCV

---

## 📦 Installation

### 1. Clone the repository
```bash
git clone https://github.com/athhh07/VisionTech-AI-Powered-Pneumonia-Detection-.git
cd VisionTech-AI-Powered-Pneumonia-Detection-
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app *(if applicable)*
```bash
streamlit run app.py
```

---

## 💡 Applications

- 🏥 Medical diagnosis assistance / preliminary screening
- 🔬 AI in healthcare research
- 🎓 Educational deep learning project

---

## 🔭 Future Improvements

- [ ] Transfer learning with ResNet / EfficientNet for higher accuracy
- [ ] Full web deployment via Streamlit
- [ ] Extend to multi-disease detection (TB, COVID-19, etc.)
- [ ] Add Grad-CAM visualizations for explainability
- [ ] Add confusion matrix and ROC curve to results section

---

## ⚠️ Disclaimer

This project is intended **strictly for educational and research purposes**. It is **not** a certified medical device and must not be used for actual clinical diagnosis. Always consult a qualified medical professional for health-related decisions.

---

## 👤 Author

**Atharva Desai**

[![GitHub](https://img.shields.io/badge/GitHub-athhh07-181717?style=flat&logo=github)](https://github.com/athhh07)
[![Email](https://img.shields.io/badge/Email-desaiatharva703%40gmail.com-D14836?style=flat&logo=gmail&logoColor=white)](mailto:desaiatharva703@gmail.com)

---

## ⭐ Support

If you found this project useful, please consider **starring** ⭐ the repository — it helps a lot and motivates further development!

</div>
