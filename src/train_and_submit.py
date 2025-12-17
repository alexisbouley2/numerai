"""
Modal script to train XGBoost model and submit predictions to Numerai.

This script:
1. Downloads training and validation data from Numerai
2. Concatenates train and validation datasets for training
3. Trains an XGBoost model with the specified hyperparameters
4. Gets the current tournament round
5. Downloads live tournament data for the current round
6. Generates predictions using feature pattern matching
7. Uploads predictions to Numerai
"""

import os
import json
import warnings
import pandas as pd
import numpy as np
import xgboost as xgb
from numerapi import NumerAPI

import modal

# Create Modal app
app = modal.App("numerai-predictions")

# Define the image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numerapi==2.21.0",
        "numerai-tools==0.5.1",
        "pandas==2.3.3",
        "pyarrow==22.0.0",
        "xgboost==3.1.2",
        "scikit-learn==1.8.0",
        "numpy",
    )
    .apt_install("git")
)

# Set up secrets for Numerai API credentials
secrets = [
    modal.Secret.from_name("numerai", required=True),
]


@app.function(
    image=image,
    secrets=secrets,
    gpu="T4",  # Use GPU for XGBoost training
    timeout=3600,  # 1 hour timeout
)
def train_and_submit():
    """
    Main function to train model and submit predictions.
    """
    # Initialize Numerai API with credentials from Modal secrets
    napi = NumerAPI(
        public_id=os.getenv("NUMERAI_PUBLIC_ID"),
        secret_key=os.getenv("NUMERAI_SECRET_KEY"),
    )
    
    # Configuration
    DATA_VERSION = "v5.1"
    MODEL_NAME = os.getenv("NUMERAI_MODEL_NAME", "default_model")
    
    # Get current round
    current_round = napi.get_current_round()
    
    print("=" * 60)
    print("NUMERAI MODEL TRAINING AND SUBMISSION")
    print("=" * 60)
    print(f"Model Name: {MODEL_NAME}")
    print(f"Data Version: {DATA_VERSION}")
    print(f"Current Round: {current_round}")
    print()
    
    # Step 1: Download feature metadata
    print("Step 1: Downloading feature metadata...")
    napi.download_dataset(f"{DATA_VERSION}/features.json")
    feature_metadata = json.load(open(f"{DATA_VERSION}/features.json"))
    feature_sets = feature_metadata["feature_sets"]
    feature_set = feature_sets["all"]
    print(f"Using {len(feature_set)} features from 'all' feature set")
    print()
    
    # Step 2: Download training and validation data
    print("Step 2: Downloading training and validation data...")
    napi.download_dataset(f"{DATA_VERSION}/train.parquet")
    train = pd.read_parquet(
        f"{DATA_VERSION}/train.parquet",
        columns=["era", "target"] + feature_set
    )
    print(f"Training data shape: {train.shape}")
    
    napi.download_dataset(f"{DATA_VERSION}/validation.parquet")
    validation = pd.read_parquet(
        f"{DATA_VERSION}/validation.parquet",
        columns=["era", "target"] + feature_set
    )
    validation = validation.dropna()
    print(f"Validation data shape: {validation.shape}")
    
    # Concatenate train and validation for training (like notebook 4)
    train = pd.concat([train, validation], ignore_index=True)
    print(f"Combined training data shape: {train.shape}")
    print()
    
    # Step 3: Train the model
    print("Step 3: Training XGBoost model...")
    model = xgb.XGBRegressor(
        n_estimators=2000,
        learning_rate=0.01,
        max_depth=5,
        colsample_bytree=0.1,
        verbosity=0,
        seed=42,
        tree_method="hist",
        device="cuda",
    )
    
    print("Fitting model on training data...")
    model.fit(train[feature_set], train["target"])
    print("Model training completed!")
    print()
    
    # Step 4: Download live tournament data for current round
    print("Step 4: Downloading live tournament data...")
    napi.download_dataset(f"{DATA_VERSION}/live_{current_round}.parquet")
    live_data = pd.read_parquet(f"{DATA_VERSION}/live_{current_round}.parquet")
    print(f"Live data shape: {live_data.shape}")
    
    # Extract features (using pattern matching like Numerai docs)
    live_features = live_data[[f for f in live_data.columns if "feature" in f]]
    print(f"Live features shape: {live_features.shape}")
    print()
    
    # Step 5: Generate predictions
    print("Step 5: Generating predictions...")
    # Batch prediction to avoid memory issues
    batch_size = 10000
    predictions = []
    
    for i in range(0, len(live_features), batch_size):
        batch = live_features.iloc[i:i+batch_size]
        batch_predictions = model.predict(batch)
        predictions.append(batch_predictions)
    
    live_predictions = np.concatenate(predictions)
    print(f"Generated predictions for {len(live_predictions)} samples")
    print()
    
    # Step 6: Prepare submission (format like Numerai docs)
    print("Step 6: Preparing submission...")
    submission = pd.Series(live_predictions, index=live_features.index).to_frame("prediction")
    submission_file = f"prediction_{current_round}.csv"
    submission.to_csv(submission_file)
    print(f"Submission shape: {submission.shape}")
    print(f"Saved to: {submission_file}")
    print()
    
    # Step 7: Upload predictions
    print("Step 7: Uploading predictions to Numerai...")
    try:
        submission_id = napi.upload_predictions(
            submission_file,
            model_id=MODEL_NAME,
        )
        print(f"✅ Successfully uploaded predictions!")
        print(f"Submission ID: {submission_id}")
    except Exception as e:
        print(f"❌ Error uploading predictions: {str(e)}")
        raise
    
    print()
    print("=" * 60)
    print("PROCESS COMPLETED SUCCESSFULLY")
    print("=" * 60)
    
    return submission_id


@app.local_entrypoint()
def main():
    """
    Local entrypoint to trigger the Modal function.
    """
    train_and_submit.remote()

