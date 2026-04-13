import argparse
import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

# --- Configuration ---
DATA_FILE = "housing_data.csv"
MODEL_ONNX_FILE = "house_price_model.onnx"
SCALER_FILE = "scalers.pkl"

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")

def load_and_preprocess_data(
        data_path,
        test_size=0.2,
        validation_size=0.2,
        random_state=42
):
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: data file not found at {data_path}")
        return None, None, None, None, None, None, None, None

    # Define features (X) and target (y)
    features = ['square_footage', 'bedrooms', 'bathrooms']
    target = "price_thousands"

    X = df[features].values
    y = df[[target]].values

    # Initialize minmax scalers for features and targets
    features_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    # Fit and Transform features and target
    X_Scaled = features_scaler.fit_transform(X)
    y_Scaled = target_scaler.fit_transform(y)


    # First split: seperate out the test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_Scaled, y_Scaled,test_size=test_size, random_state=random_state
    )

    # Split remaining data into training and validation set
    val_ratio_in_train_val = validation_size / (1-test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio_in_train_val, random_state=random_state
    )

    # Convert to PyTorch tensors and move to selected devices
    X_train_tensor = torch.tensor(
        X_train, dtype=torch.float32
    ).to(DEVICE)

    y_train_tensor = torch.tensor(
        y_train, dtype=torch.float32
    ).to(DEVICE)

    X_val_tensor = torch.tensor(
        X_val, dtype=torch.float32
    ).to(DEVICE)

    y_val_tensor = torch.tensor(
        y_val, dtype=torch.float32
    ).to(DEVICE)

    X_test_tensor = torch.tensor(
        X_test, dtype=torch.float32
    ).to(DEVICE)

    y_test_tensor = torch.tensor(
        y_test, dtype=torch.float32
    ).to(DEVICE)

    print(f"Data loaded and preprocessed from {data_path}")
    print(f"Training samples: {X_train_tensor.shape[0]}, Validation samples: {X_val_tensor.shape[0]}, Test samples: {X_test_tensor.shape[0]}")


    return X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor, features_scaler, target_scaler



if __name__ == "__main__":
    
    # Command line flags
    parser = argparse.ArgumentParser(
        description="A simple neural network for housing price prediction using PyTorch and ONNX."
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the model and save it as an ONNX file"
    )
    
    parser.add_argument(
        "--data-path",
        type=str,
        default=DATA_FILE,
        help=f"Path to CSV file (default {DATA_FILE})"
    )

    args = parser.parse_args()

    if args.train:
        print("--- Training Mode ---")
        X_train, y_train, X_val, y_val, X_test, y_test, feature_scaler, target_scaler = load_and_preprocess_data(
            args.data_path
        )
        print(f"Shape: {X_train.shape[0]}") 