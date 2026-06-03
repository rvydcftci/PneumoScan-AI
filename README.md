# 🩺 PneumoScan AI

Deep learning based pneumonia detection system developed using Chest X-Ray images and Convolutional Neural Networks (CNN).

---

## 📖 Project Overview

PneumoScan AI is a web-based application that analyzes chest X-ray images and predicts whether the patient has:

- NORMAL
- PNEUMONIA

The system also provides additional image processing visualizations such as:

- CLAHE Contrast Enhancement
- Edge Detection
- Saliency Heatmap

---

## 🚀 Features

- 📤 Upload chest X-ray images
- 🧠 CNN-based pneumonia detection
- 📊 Prediction confidence score
- 🔥 Saliency Heatmap visualization
- 🎨 CLAHE contrast enhancement
- 📐 Edge detection visualization
- 🖥️ User-friendly Streamlit interface

---

## 🛠 Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Streamlit

---

## 📂 Dataset

Chest X-Ray Images (Pneumonia)

Source:

https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

⚠️ Dataset images are not included in this repository.

---

## 🧠 CNN Architecture

The model consists of:

- Conv2D (32 Filters)
- MaxPooling2D
- Conv2D (64 Filters)
- MaxPooling2D
- Conv2D (128 Filters)
- MaxPooling2D
- Conv2D (256 Filters)
- MaxPooling2D
- Dense (256)
- Dropout (0.5)
- Dense (128)
- Dropout (0.3)
- Output Layer (Sigmoid)

---

## 📊 Model Performance

| Metric | Value |
|----------|----------|
| Accuracy | 82.69% |

The model achieved successful classification performance on unseen chest X-ray images.

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/rvydcftci/PneumoScan-AI.git
cd PneumoScan-AI
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

---

## 🖼️ Application Screenshots

### 🏠 Home Screen

![](1.jpeg)

---

### 📤 Image Upload

![](2.jpeg)

---

### 🧠 Prediction Result

![](3.jpeg)

---

### 🔥 Saliency Heatmap

![](4.jpeg)

---

### 🎨 CLAHE Processing

![](5.jpeg)

---

### 📐 Edge Detection

![](6.jpeg)

---

## 👩‍💻 Developer

**Rüveyda Çiftci**

---

## ⚠️ Disclaimer

This project was developed for educational and research purposes only.

It should not be used as a substitute for professional medical diagnosis.
