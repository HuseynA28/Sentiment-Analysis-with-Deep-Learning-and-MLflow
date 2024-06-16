
# Sentiment Analysis with Deep Learning and MLflow


**Project Name:** Sentiment Analysis with Deep Learning and MLflow

**Description:**
Developed and deployed a sentiment analysis model using deep learning techniques. The project involved training the model on labeled sentences, logging and managing the model lifecycle with MLflow, and deploying the model using FastAPI. The deployment utilized Docker and MinIO for a scalable and reliable infrastructure. Implemented continuous integration and deployment practices to ensure seamless updates and maintenance.

**Technologies:** Python, TensorFlow, Keras, FastAPI, MLflow, Docker, MinIO, MySQL, Conda

**Highlights:**
- Trained a deep learning model achieving high accuracy in sentiment classification.
- Integrated MLflow for experiment tracking and model registry.
- Deployed the model as a REST API with FastAPI and Docker.
- Ensured scalability and reliability using Docker Compose and Kubernetes.
- Implemented continuous deployment with automated pipelines.

![MLflow](https://img.shields.io/badge/MLflow-1.20.0-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.68.1-green)
![Docker](https://img.shields.io/badge/Docker-20.10.7-blue)

This project demonstrates the deployment of a deep learning model for sentiment analysis using MLflow Model Registry. The deployment utilizes Docker, FastAPI, and MinIO for managing machine learning lifecycle operations.

## Project Overview

The goal of this project is to train a deep learning model to classify sentences into positive and negative sentiments. The model training process is tracked and managed using MLflow, and the final model is served through a FastAPI application.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Setup Instructions](#setup-instructions)
  - [Start MLflow](#start-mlflow)
  - [Copy/Push Project to VM](#copypush-project-to-vm)
  - [Create Virtual Environment](#create-virtual-environment)
  - [Install Requirements](#install-requirements)
  - [Get Data](#get-data)
  - [Train and Log Model](#train-and-log-model)
  - [Register Model in MLflow](#register-model-in-mlflow)
  - [Start FastAPI Server](#start-fastapi-server)
  - [Open API Documentation](#open-api-documentation)
  - [Shut Down Docker Compose](#shut-down-docker-compose)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Contact](#contact)

## Prerequisites

- Docker and Docker Compose
- MinIO
- MySQL
- Conda
- Python 3.8+

## Setup Instructions

### Start MLflow

```sh
cd ~/mlops_project/
docker-compose up -d mysql mlflow minio
```

### Copy/Push Project to VM

```sh
mv sentiment_analysis_with_mlflow/ sentiment_analysis
cd sentiment_analysis
```

### Create Virtual Environment

```sh
conda create --name sentiment_env python=3.8
conda activate sentiment_env
```

### Install Requirements

```sh
pip install -r requirements.txt
```

### Get Data

Data Source: [UCI Sentiment Labeled Sentences](https://archive.ics.uci.edu/ml/machine-learning-databases/00331/)

```sh
mkdir -p ~/datasets
cd ~/datasets
wget -O sentiment_labeled_sentences.zip https://archive.ics.uci.edu/ml/machine-learning-databases/00331/sentiment%20labelled%20sentences.zip
unzip sentiment_labeled_sentences.zip
mv sentiment\ labelled\ sentences sentiment_labeled_sentences
```

### Train and Log Model

```sh
cd ~/sentiment_analysis
python train_model.py
```

### Register Model in MLflow

- Open the MLflow UI.
- Register the trained model.
- Note the model name and version from the MLflow web UI.
- Update the model name and version in `app.py`.

### Start FastAPI Server

```sh
uvicorn app:app --host 0.0.0.0 --port 8002 --reload
```

### Open API Documentation

Open your browser and navigate to [http://localhost:8002/docs](http://localhost:8002/docs)

### Shut Down Docker Compose

```sh
cd ~/mlops_project/
docker-compose down
```

## Project Structure

```plaintext
sentiment_analysis/
├── app.py                  # FastAPI application for serving the model
├── Dockerfile              # Docker configuration for FastAPI application
├── requirements.txt        # List of required Python packages
├── train_model.py          # Script for training the deep learning model
└── datasets/               # Directory for storing the dataset
```

## Technologies Used

- **Python**
- **TensorFlow/Keras**
- **FastAPI**
- **MLflow**
- **Docker**
- **MinIO**
- **MySQL**
- **Conda**

## Contact

For any questions or inquiries, please contact 



