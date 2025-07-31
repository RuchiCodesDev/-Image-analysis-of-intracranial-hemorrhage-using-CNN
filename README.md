# 🧠 Image Analysis of Intracranial Hemorrhage using CNN

Welcome to my project! This repository is part of my AI/ML learning journey, where I built a deep learning model using **Convolutional Neural Networks (CNN)** to detect and classify **intracranial hemorrhages (ICH)** from brain CT images.

## 📌 Objective

The goal of this project is to:
- Automatically detect ICH in brain CT scans
- Classify different types of hemorrhages (e.g. subdural, epidural, intraparenchymal, etc.)
- Support early diagnosis using AI in healthcare

## 👩‍💻 About Me

I’m **Ruchi Shukla**, a passionate learner in Artificial Intelligence and Machine Learning, exploring the intersection of healthcare and technology.  
This project reflects my interest in **medical image analysis** and using CNNs for real-world applications.

## 🛠️ Tools & Technologies

- Python 🐍
- TensorFlow / Keras
- OpenCV & NumPy
- Matplotlib for visualization
- CNN Architecture (custom + pre-trained exploration)
- Dataset from: RSNA / Kaggle (or local CT scan samples)

## 🗂️ Project Structure
├── src/                     # Python scripts for model, training, etc.
├── data/                    # CT scan dataset folder
├── notebooks/               # Jupyter notebooks (EDA, training) 
├── models/                  # Saved models 
├── outputs/                 # Evaluation results or figures 
├── requirements.txt         # Python dependencies 
├── README.md                # Project documentation 

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/RuchiCodesDev/-Image-analysis-of-intracranial-hemorrhage-using-CNN.git
   cd -Image-analysis-of-intracranial-hemorrhage-using-CNN

2. Install requirements:

pip install -r requirements.txt

3. Run the training script or notebook:

python src/train.py

4. Evaluate:

python src/evaluate.py

## 📊 Results

Metric	Value

Accuracy	96%
AUC Score	0.91
F1-Score	0.88

## ✨ Features

Automatic CT scan preprocessing

CNN-based classification model

Binary or multi-label outputs for hemorrhage types

Visualization of model focus using Grad-CAM

## 🙋‍♀️ Future Improvements

Use pre-trained CNNs (e.g. ResNet, VGG)

Add segmentation support

Build a web interface for predictions

Hyperparameter tuning and cross-validation
