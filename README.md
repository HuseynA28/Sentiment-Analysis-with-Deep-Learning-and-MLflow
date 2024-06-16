Sentiment Analysis with Deep Learning and MLflow



This project demonstrates the deployment of a deep learning model for sentiment analysis using MLflow Model Registry. The deployment utilizes Docker, FastAPI, and MinIO for managing machine learning lifecycle operations.

Project Overview
The goal of this project is to train a deep learning model to classify sentences into positive and negative sentiments. The model training process is tracked and managed using MLflow, and the final model is served through a FastAPI application.

Table of Contents
Prerequisites
Setup Instructions
Start MLflow
Copy/Push Project to VM
Create Virtual Environment
Install Requirements
Get Data
Train and Log Model
Register Model in MLflow
Start FastAPI Server
Open API Documentation
Shut Down Docker Compose
Project Structure
Technologies Used
Contact
Prerequisites
Docker and Docker Compose
MinIO
MySQL
Conda
Python 3.8+
Setup Instructions
Start MLflow
sh
Copy code
cd ~/mlops_project/
docker-compose up -d mysql mlflow minio
Copy/Push Project to VM
sh
Copy code
mv sentiment_analysis_with_mlflow/ sentiment_analysis
cd sentiment_analysis
Create Virtual Environment
sh
Copy code
conda create --name sentiment_env python=3.8
conda activate sentiment_env
Install Requirements
sh
Copy code
pip install -r requirements.txt
Get Data
Data Source: UCI Sentiment Labeled Sentences

sh
Copy code
mkdir -p ~/datasets
cd ~/datasets
wget -O sentiment_labeled_sentences.zip https://archive.ics.uci.edu/ml/machine-learning-databases/00331/sentiment%20labelled%20sentences.zip
unzip sentiment_labeled_sentences.zip
mv sentiment\ labelled\ sentences sentiment_labeled_sentences
Train and Log Model
sh
Copy code
cd ~/sentiment_analysis
python train_model.py
Register Model in MLflow
Open the MLflow UI.
Register the trained model.
Note the model name and version from the MLflow web UI.
Update the model name and version in app.py.
Start FastAPI Server
sh
Copy code
uvicorn app:app --host 0.0.0.0 --port 8002 --reload
Open API Documentation
Open your browser and navigate to http://localhost:8002/docs

Shut Down Docker Compose
sh
Copy code
cd ~/mlops_project/
docker-compose down
Project Structure
plaintext
Copy code
sentiment_analysis/
├── app.py                  # FastAPI application for serving the model
├── Dockerfile              # Docker configuration for FastAPI application
├── requirements.txt        # List of required Python packages
├── train_model.py          # Script for training the deep learning model
└── datasets/               # Directory for storing the dataset
Technologies Used
Python
TensorFlow/Keras
FastAPI
MLflow
Docker
MinIO
MySQL
Conda
Contact
For any questions or inquiries, please contact Your Name.
