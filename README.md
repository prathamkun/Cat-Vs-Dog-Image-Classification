# 🐱🐶 Cat vs Dog Image Classifier (End-to-End ML Project)

An end-to-end Deep Learning project that classifies Cats and Dogs using TensorFlow and performs real-time prediction through a webcam.

This project demonstrates the complete Machine Learning workflow:

- Dataset preprocessing
- Data cleaning
- Model training (Transfer Learning)
- Real-time inference
- Deployment-ready structure

---

## 🚀 Features

✅ Train CNN model using TensorFlow/Keras  
✅ Transfer Learning with MobileNetV2  
✅ Automatic dataset cleaning (remove corrupt images)  
✅ Real-time webcam classification  
✅ High accuracy (~98% validation accuracy)  
✅ Clean project structure  
✅ GitHub-ready repository

---

## 🧠 Tech Stack

- Python 3.11
- TensorFlow / Keras
- OpenCV
- NumPy
- Streamlit (optional app interface)

---

## 📁 Project Structure

```
CatDogClassifier/
│
├── dataset/ # Training dataset
│ ├── Cat/
│ └── Dog/
│
├── models/
│ └── cat_dog_classifier.h5
│
├── train.py # Model training script
├── clean_dataset.py # Remove corrupt images
├── test_image.py # Test single image
├── webcam.py # Real-time webcam detection
├── app.py # Streamlit UI (optional)
├── requirements.txt
└── README.md
```

---

## 📦 Installation

Clone repository:

git clone https://github.com/your-username/Cat-Vs-Dog-Image-Classification.git

cd Cat-Vs-Dog-Image-Classification


Create virtual environment:

 python3.11 -m venv .venv
source .venv/bin/activate 


Install dependencies:



---

## 🧹 Clean Dataset (IMPORTANT)

The Kaggle Dogs vs Cats dataset contains corrupt images.

Run:


```python clean_dataset.py```


---

## 🏋️ Train Model


```python train.py```


After training, model will be saved inside:

models/cat_dog_classifier.h5


---

## 📷 Real-Time Webcam Detection

python webcam.py


This will:

- Open webcam
- Detect image continuously
- Show prediction + confidence

Press **Q** to quit.

---

## 📊 Model Details

- Architecture: MobileNetV2 (Transfer Learning)
- Input size: 224x224
- Optimizer: Adam
- Loss: Binary Crossentropy
- Validation Accuracy: ~98%

---

## ⚠️ Known Issues

- Some datasets contain corrupted JPEG files (fixed using clean script).
- TensorFlow requires NumPy < 2.0.

---

## 🎯 Future Improvements

- Object detection bounding box
- Real-time tracking
- Better UI overlays
- Streamlit deployment
- Mobile deployment

---

## 👨‍💻 Author

Pratham Kun

---

⭐ If you like this project, consider giving it a star!





