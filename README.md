# 🫁 PneumoScan AI

AI-powered pneumonia detection system developed using chest X-ray images and Convolutional Neural Networks (CNN).

---

## 📖 Overview

PneumoScan AI is a web-based medical image analysis application that assists in the detection of pneumonia from chest X-ray images.

The system uses a custom Convolutional Neural Network (CNN) trained on the Chest X-Ray Pneumonia dataset and provides prediction results together with image processing and explainable AI visualizations.

---

## 🚀 Features

- Chest X-ray image classification
- Pneumonia detection using CNN
- Confidence score visualization
- CLAHE contrast enhancement
- Canny edge detection
- Saliency Heatmap visualization
- Training and validation performance graphs
- User-friendly Streamlit interface

---

## 🛠 Technologies Used

### Artificial Intelligence & Data Science

- Python
- TensorFlow
- Keras
- NumPy
- Scikit-Learn

### Image Processing

- OpenCV
- CLAHE
- Canny Edge Detection

### Visualization

- Matplotlib

### Web Interface

- Streamlit

---

## 🧠 Model Architecture

The project uses a custom Convolutional Neural Network consisting of:

- 4 Convolutional Layers
- Max Pooling Layers
- Dense Layers
- Dropout Regularization
- Sigmoid Output Layer

### Training Parameters

| Parameter | Value |
|------------|------------|
| Image Size | 224 × 224 |
| Batch Size | 32 |
| Optimizer | Adam |
| Learning Rate | 0.0001 |
| Loss Function | Binary Crossentropy |
| Epochs | 10 |

---

## 📊 Model Performance

| Metric | Value |
|----------|----------|
| Test Accuracy | 82.69% |
| Classes | NORMAL / PNEUMONIA |

The model achieved promising results in distinguishing pneumonia cases from normal chest radiographs.

---

## 📂 Dataset

This project uses the public Chest X-Ray Pneumonia Dataset:

📎 https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

**Note:** The dataset is not included in this repository due to licensing and storage limitations.

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/rvydcftci/PneumoScan-AI.git
cd PneumoScan-AI
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

---

## 🏠 Home Screen

![](1.jpeg)

---

## 📤 Image Upload & Analysis

![](2.jpeg)

---

## 🧠 Prediction Results

![](3.jpeg)

---

## 🎨 CLAHE & Edge Detection

![](4.jpeg)

---

## 🔥 Saliency Heatmap

![](5.jpeg)

---

## 📈 Model Performance

![](6.jpeg)

---

## 👩‍💻 Developer

**Rüveyda Çiftci**

Software Engineering Student  
Fırat University

2026
