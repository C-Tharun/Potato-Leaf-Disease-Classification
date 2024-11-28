# Potato-Leaf-Disease-Classification

A comprehensive machine learning project to identify and classify potato diseases using advanced image classification techniques. This project includes model training, API deployment, frontend integration, and mobile application support, creating an end-to-end pipeline for detecting diseases in potato plants.

# Project Overview

This project aims to aid farmers and agronomists by providing an efficient solution for diagnosing potato plant health. By leveraging convolutional neural networks (CNNs) for image classification, the system identifies common diseases like Early Blight and Late Blight, along with healthy leaves. The model is deployed via RESTful APIs, accessible through web and mobile applications, and scalable using Google Cloud Platform (GCP).

# Features
Disease Classification: Accurately predicts common potato diseases.
API Integration: Provides real-time predictions via FastAPI.
Cross-Platform Support: Includes both a ReactJS web frontend and a React Native mobile app.
Google Cloud Integration: Enables scalable deployment for real-world use.
TF Lite Support: Optimized for lightweight models to run on mobile devices.

# Technologies Used
Programming and Frameworks
Python: For model training and API development.
TensorFlow/Keras: For building and optimizing the classification model.
FastAPI: For creating a robust and scalable API.
ReactJS: For developing the web frontend.
React Native: For building the mobile app.

# Cloud and Deployment Tools
Google Cloud Platform (GCP): For hosting and scaling the trained model.
TensorFlow Serving: For serving predictions via Docker containers.

# Setup Guide
1. Python Setup
Install Python (Follow the official Python installation guide).
Install required Python packages:

```
pip install -r training/requirements.txt  
pip install -r api/requirements.txt  
```

Install TensorFlow Serving (Follow TensorFlow Serving installation guide).

2. ReactJS Setup
Install Node.js (Setup instructions) and NPM.
Install dependencies:

```
cd frontend  
npm install --from-lock-json  
npm audit fix  
```

Create .env from .env.example and update the API URL as needed.

3. React Native Setup
Follow the React Native CLI Quickstart guide.
Install dependencies:

```
cd mobile-app  
yarn install  
```

For macOS users:

```
cd ios && pod install && cd ../  
```

Create .env from .env.example and update the API URL as needed.

# Model Training
Download the potato dataset from Kaggle.
Retain folders specific to potato diseases.
Run Jupyter Notebook:

```
jupyter notebook  
```

Open training/potato-disease-training.ipynb and update the dataset path in cell #2.
Execute all cells sequentially to train the model.
Save the generated model in the models folder with a version number.

# API Deployment
Using FastAPI
Navigate to the api folder:

```
cd api  
```

Run the server:

```
uvicorn main:app --reload --host 0.0.0.0  
```

Using FastAPI with TensorFlow Serving

Copy models.config.example to models.config and update paths.
Run TensorFlow Serving with Docker:

```
docker run -t --rm -p 8501:8501 -v C:/Code/potato-disease-classification:/potato-disease-classification tensorflow/serving --rest_api_port=8501 --model_config_file=/potato-disease-classification/models.config  
```

Run the API server:

```
uvicorn main-tf-serving:app --reload --host 0.0.0.0  
```

# Frontend Setup
Navigate to the frontend folder:

```
cd frontend  
```

Create .env and update REACT_APP_API_URL.
Start the React app:

```
npm run start  
```

# TF Lite Deployment
Open training/tf-lite-converter.ipynb in Jupyter Notebook.
Update the dataset path in cell #2.
Run all cells to convert the model.
Save the TF Lite model in the tf-lite-models folder.

Deploy TF Lite on GCP by:

Creating a GCP project and bucket.
Uploading the TF Lite model.
Deploying via Google Cloud Functions.

# Full Model Deployment on GCP
Create a GCP bucket and upload the .h5 model.
Authenticate using Google Cloud SDK:

```
gcloud auth login  
```

Deploy the model with:

```
cd gcp  
gcloud functions deploy predict --runtime python38 --trigger-http --memory 512 --project project_id  
```

Test using Postman with the Trigger URL.



