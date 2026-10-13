# 🩺 Diabetic Retinopathy Detection Using Deep Learning

## 📌 Project Overview

Diabetic Retinopathy Detection is a deep learning-based web application that predicts the severity of diabetic retinopathy from retinal fundus images.

The project uses **EfficientNetB0**, a lightweight Convolutional Neural Network (CNN), with TensorFlow/Keras to classify retinal images into five severity levels. The application provides a user-friendly interface for uploading an image and viewing the model's prediction.

## 🚀 Live Demo

**Frontend:** https://mlvision.netlify.app/login.html



## 🛠️ Technologies Used

- Python
- Flask (if used in `app.py`)
- TensorFlow / Keras
- EfficientNetB0
- HTML, CSS, JavaScript
- Netlify for frontend deployment
- APTOS 2019 Blindness Detection dataset

## 🧠 Model Details

- **Model:** EfficientNetB0
- **Approach:** Transfer learning
- **Dataset:** APTOS retinal fundus images
- **Classification:** Five DR severity classes
- **Augmentation:** Lesion-based image augmentation

### Classification Categories

| Class | Severity |
|---|---|
| 0 | No DR |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |
| 4 | Proliferative DR |

## ✨ Features

- Upload retinal fundus images.
- Predict diabetic retinopathy severity.
- Display prediction results through a web interface.
- Use a lightweight deep learning model for image classification.
- Access the frontend through a deployed Netlify website.

## 🏗️ System Architecture

1. The user opens the Netlify-hosted frontend.
2. The user uploads a retinal fundus image.
3. The frontend sends the image to the Python backend.
4. The backend preprocesses the image and runs the trained EfficientNetB0 model.
5. The predicted DR severity is returned to the frontend.
6. The frontend displays the prediction.

## ⚙️ Backend Setup

### Prerequisites

- Python installed
- Required Python packages
- Trained model file
- Backend source code

### Run the Backend Locally

1. Clone or download the repository.

```bash
git clone https://github.com/varunkumarelluru/DR-Detection
```

2. Install the required dependencies.

```bash
pip install -r requirements.txt
```

3. Start the Python backend.

```bash
python app.py
```

4. The backend will start at the local address configured in `app.py`, commonly `http://127.0.0.1:5000` for Flask applications.

## 🌐 Deployment

- **Frontend:** Deployed using Netlify.
- **Backend:** Python application started using `python app.py` in the local development environment or a configured Python hosting environment.
- **Model:** The backend loads the trained model to generate predictions.

**Note:** Netlify primarily hosts the frontend. Running `python app.py` on your own computer does not make the backend publicly accessible. For the deployed Netlify frontend to process images, the backend must be hosted on a publicly accessible server, and the frontend must send requests to that backend URL.

## 📊 Evaluation Metrics

Model performance can be assessed using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Add your actual evaluation results if available.

## ⚠️ Disclaimer

This application is intended for educational and research purposes only. Its predictions are not a substitute for professional medical examination or diagnosis.


