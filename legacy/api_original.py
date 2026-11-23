import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from fastapi import FastAPI, File, UploadFile
import io

# Load trained autoencoder model and scaler
autoencoder = tf.keras.models.load_model("final/autoencoder_fraud_detection.h5")
#scaler = joblib.load("final/scaler.pkl")

# Initialize FastAPI app
app = FastAPI()

@app.post("/detect_anomalies/")
async def detect_anomalies(file: UploadFile = File(...)):
    # Read CSV file into DataFrame
    contents = await file.read()
    test_data = pd.read_csv(io.BytesIO(contents))
    # Drop unnecessary columns (e.g., user ID or other identifiers)
    unnecessary_columns = ['user_id', 'account_number', 'transaction_id']

    test_data = test_data.drop(columns=[col for col in unnecessary_columns if col in test_data.columns])

    # Encode categorical features
    categorical_features = test_data.select_dtypes(include=['object']).columns

    for col in categorical_features:
        le = LabelEncoder()
        test_data[col] = le.fit_transform(test_data[col])

    # Selecting only numerical columns for autoencoder training
    num_features = test_data.select_dtypes(include=[np.number]).columns

    scaler = MinMaxScaler()

    test_scaled = scaler.fit_transform(test_data[num_features])

    reconstructed = autoencoder.predict(df_scaled)
    # Compute reconstruction error
    reconstruction_error = np.mean(np.square(df_scaled - reconstructed), axis=1)
    
    # Define anomaly threshold (e.g., top 5% as anomalies)
    threshold = np.percentile(reconstruction_error, 96)

    anomalies = df[reconstruction_error > threshold]
    
    # Extract only the user IDs of flagged users
    flagged_users = anomalies["user_id"].tolist()

    print(flagged_users)
    
    return {"flagged_users": flagged_users}
