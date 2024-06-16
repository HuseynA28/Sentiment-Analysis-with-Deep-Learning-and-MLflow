import pandas as pd
import mlflow
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import joblib
import logging

# Setting up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Filepath dictionary
filepath_dict = {
    'yelp': '/home/train/datasets/sentiment_labeled_sentences/yelp_labelled.txt',
    'amazon': '/home/train/datasets/sentiment_labeled_sentences/amazon_cells_labelled.txt',
    'imdb': '/home/train/datasets/sentiment_labeled_sentences/imdb_labelled.txt'
}

# Set up MLflow environment
os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5000/'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000/'
mlflow.keras.autolog()

def load_data(filepath_dict):
    """Load data from given filepaths and concatenate into a single DataFrame."""
    df_list = []
    for source, filepath in filepath_dict.items():
        try:
            df = pd.read_csv(filepath, names=['sentence', 'label'], sep='\t')
            df['source'] = source
            df_list.append(df)
        except Exception as e:
            logger.error(f"Error reading {filepath}: {e}")
            continue
    return pd.concat(df_list, ignore_index=True)

def preprocess_data(df, source):
    """Preprocess data: split into training and testing sets, tokenize and pad sequences."""
    df_source = df[df['source'] == source]
    sentences = df_source['sentence'].values
    y = df_source['label'].values

    sentences_train, sentences_test, y_train, y_test = train_test_split(
        sentences, y, test_size=0.25, random_state=1000)

    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(sentences_train)

    # Save tokenizer
    joblib.dump(tokenizer, "saved_models/keras-sentence-classification-tokenizer.pkl")

    X_train = tokenizer.texts_to_sequences(sentences_train)
    X_test = tokenizer.texts_to_sequences(sentences_test)

    maxlen = 100
    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

    return X_train, X_test, y_train, y_test, len(tokenizer.word_index) + 1

def build_model(vocab_size, embedding_dim=50, maxlen=100):
    """Build and compile the Keras model."""
    model = Sequential()
    model.add(layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen))
    model.add(layers.GlobalMaxPool1D())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    """Train the model and evaluate its performance."""
    model.fit(X_train, y_train, epochs=10, verbose=1, validation_data=(X_test, y_test), batch_size=10)
    loss, accuracy = model.evaluate(X_train, y_train, verbose=0)
    logger.info(f"Training Accuracy: {accuracy:.4f}")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    logger.info(f"Testing Accuracy:  {accuracy:.4f}")

def main():
    # Load data
    df = load_data(filepath_dict)
    logger.info(f"Data loaded with shape: {df.shape}")

    # Preprocess data
    X_train, X_test, y_train, y_test, vocab_size = preprocess_data(df, 'yelp')
    logger.info(f"Data preprocessed. Vocabulary size: {vocab_size}")

    # Build model
    model = build_model(vocab_size)
    model.summary()

    # Train and evaluate model
    train_and_evaluate_model(model, X_train, y_train, X_test, y_test)

    # Predict and log results for sample input
    predictions = model.predict(X_test[:5])
    logger.info(f"Sample predictions: {predictions}")

if __name__ == "__main__":
    main()
