Sentiment Analysis with Deep Learning and MLflow
This project demonstrates deploying a deep learning model for sentiment analysis using MLflow Model Registry. The deployment utilizes Docker, FastAPI, and MinIO for managing machine learning lifecycle operations.

Project Overview
This project involves training a deep learning model to perform sentiment analysis on a dataset of labeled sentences. The trained model is logged and managed using MLflow, and the deployment is handled through FastAPI for serving predictions via an API.

Prerequisites
Docker and Docker Compose
MinIO
MySQL
Conda
Python 3.8+
Setup Instructions
Step 1: Start MLflow
sh
Copy code
cd ~/02_mlops_docker/
docker-compose up -d mysql mlflow minio
Step 2: Copy/Push Your Project to VM
sh
Copy code
mv 11_deploy_fastapi_deeplearning_mlflow/ 11
cd 11
Step 3: Activate/Create Conda/Virtual Env
sh
Copy code
conda activate fastapi
Step 4: Install Requirements
sh
Copy code
pip install -r requirements.txt
Step 4.1: Get Data
Data Source: UCI Sentiment Labeled Sentences

sh
Copy code
cd ~/datasets
wget -O sentiment_labeled_sentences.zip https://archive.ics.uci.edu/ml/machine-learning-databases/00331/sentiment%20labelled%20sentences.zip
unzip sentiment_labeled_sentences.zip
mv sentiment\ labelled\ sentences sentiment_labeled_sentences
ll sentiment_labeled_sentences
Step 5: Train and Log Your Experiment to MLflow
sh
Copy code
cd ~/11
python train.py
Step 6: Register Model on MLflow UI
Open MLflow UI.
Register the trained model.
Note the model name and version from the MLflow web UI.
Update the model name and version in main.py.
Step 7: Start Uvicorn
sh
Copy code
uvicorn main:app --host 0.0.0.0 --port 8002 --reload
Step 8: Open API Documentation
Open your browser and navigate to http://localhost:8002/docs#

Step 9: Close Docker Compose
sh
Copy code
cd ~/02_mlops_docker/
docker-compose down
Project Structure
train.py: Script for training the deep learning model and logging it to MLflow.
main.py: FastAPI application for serving the model predictions.
requirements.txt: List of required Python packages.
Dockerfile: Docker configuration for the FastAPI application.
datasets/: Directory for storing the sentiment analysis dataset.
Technologies Used
Python
TensorFlow/Keras
FastAPI
MLflow
Docker
MinIO
MySQL
Contact
For any questions or inquiries, please contact Your Name.
