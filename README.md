# 🧠 Image Analysis of Intracranial Hemorrhage (ICH) Using CNN

This project implements a Convolutional Neural Network (CNN) for detecting **intracranial hemorrhage (ICH)** in brain CT scans. ICH is a life-threatening condition that requires rapid and accurate diagnosis. This deep learning model helps in automating the image analysis process and improving diagnostic efficiency.

---

## 📌 Table of Contents
- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Model Architecture](#model-architecture)
- [Evaluation](#evaluation)
- [Technologies Used](#technologies-used)
- [Results](#results)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Future Enhancements](#future-enhancements)
- [License](#license)

---

## 🔍 Overview

Intracranial Hemorrhage, or bleeding within the skull, is a medical emergency. Manual interpretation of CT scans is time-consuming and subject to human error. This project leverages **CNN-based image classification** to detect ICH efficiently and accurately using brain CT images.

---

## ❗ Problem Statement

- Manual detection of hemorrhages from CT scans is time-intensive and dependent on the radiologist's expertise.
- The project aims to automate this task using deep learning, reducing diagnosis time and increasing accuracy.

---

## 🗂 Dataset

- **Source**: RSNA Intracranial Hemorrhage Detection dataset (or similar)
- **Input**: 2D brain CT images (grayscale or RGB)
- **Classes**:
  - No Hemorrhage (Negative)
  - Hemorrhage (Positive)

> Images are preprocessed and resized for consistent input to the CNN model.

---

## 🧪 Methodology

1. **Image Preprocessing**:
   - Resizing (e.g., 224x224)
   - Normalization
   - Data Augmentation (rotation, flipping, zoom)

2. **Model Building**:
   - Custom CNN architecture designed for medical image classification
   - Batch Normalization and Dropout to avoid overfitting

3. **Training**:
   - Split data into training, validation, and test sets
   - Loss Function: Binary Crossentropy
   - Optimizer: Adam
   - Evaluation: Accuracy, ROC-AUC

4. **Visualization**:
   - Accuracy/Loss curves
   - Confusion matrix
   - Grad-CAM (optional for explainability)

---

## 🧠 Model Architecture (Sample)
Input Layer (224x224x1)
↓
Conv2D → ReLU → MaxPooling
↓
Conv2D → ReLU → MaxPooling
↓
Flatten
↓
Dense → ReLU → Dropout
↓
Output Layer (Sigmoid)

## yaml

> The architecture may vary depending on performance tuning and training data.

---

## 📈 Evaluation

- **Accuracy**
- **Precision / Recall**
- **F1 Score**
- **Confusion Matrix**
- **ROC-AUC Score**

The model showed high accuracy in distinguishing between normal and hemorrhagic scans.

---

## ⚙️ Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy, Pandas
- Matplotlib, Seaborn
- Jupyter Notebook

---

## 🧾 Project Structure
Image-analysis-of-intracranial-hemorrhage-using-CNN/
├── dataset/
│ └── [CT Scan Images]
├── model/
│ └── cnn_model.h5
├── notebooks/
│ └── ICH_detection_CNN.ipynb
├── results/
│ └── accuracy_curve.png
│ └── confusion_matrix.png
├── README.md
└── requirements.txt


---

## ▶️ How to Run

## ▶️ How to Run

## ▶️ How to Run

1. **Clone the repository:**

```bash
git clone https://github.com/RuchiCodesDev/-Image-analysis-of-intracranial-hemorrhage-using-CNN.git
cd Image-analysis-of-intracranial-hemorrhage-using-CNN
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Launch the notebook:**

```bash
jupyter notebook notebooks/ICH_detection_CNN.ipynb
```
📌 Future Enhancements
✅ Use pretrained models (ResNet, EfficientNet) for improved performance

✅ Implement Grad-CAM for model explainability

📊 Deploy as a web app using Streamlit

🧠 Incorporate 3D CNN for volumetric CT data

⚕️ Integrate with a PACS system for real-time hospital deployment


 

2. Install dependencies:
pip install -r requirements.txt

3. Launch the notebook:
jupyter notebook notebooks/ICH_detection_CNN.ipynb

---



