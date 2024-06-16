from fastapi import FastAPI, HTTPException
from schemas import Comment
import os
import mlflow.keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set environment variables for MLflow
os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5000/'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000/'

# Load model from MLflow model registry
model_name = "DL_Sentiment_Classification"
model_version = 1
try:
    model = mlflow.keras.load_model(model_uri=f"models:/{model_name}/{model_version}")
    logger.info(f"Loaded model {model_name} version {model_version} from MLflow.")
except Exception as e:
    logger.error(f"Error loading model from MLflow: {e}")
    model = None

# Load tokenizer
tokenizer_path = "saved_models/keras-sentence-classification-tokenizer.pkl"
try:
    tokenizer_loaded = joblib.load(tokenizer_path)
    logger.info("Loaded tokenizer successfully.")
except Exception as e:
    logger.error(f"Error loading tokenizer: {e}")
    tokenizer_loaded = None

app = FastAPI()

# Function to make prediction
def make_prediction(model, comment: str):
    try:
        # Converting text to integers
        token = tokenizer_loaded.texts_to_sequences([comment])
        maxlen = 100
        token = pad_sequences(token, padding='post', maxlen=maxlen)

        # Predict
        prediction = model.predict(token)
        if prediction[0] > 0.5:
            return "positive"
        return "negative"
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise

# Prediction endpoint
@app.post("/prediction/comment")
async def predict_advertising(request: Comment):
    if not model or not tokenizer_loaded:
        raise HTTPException(status_code=500, detail="Model or tokenizer not loaded.")
    
    try:
        prediction = make_prediction(model, request.comment)
        logger.info(f"Prediction made: {prediction}")
        return {"prediction": prediction}
    except Exception as e:
        logger.error(f"Error in prediction endpoint: {e}")
        raise HTTPException(status_code=500, detail="Error making prediction.")
