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


class HousePricePredictor(nn.Module):
    def __init__(self):
        super(HousePricePredictor, self).__init__()
        # Input layer (3 features) -> Hidden layer 1 (64 neurons)
        self.fc1 = nn.Linear(3, 64)
        # Hidden layer 1 (64 neurons) -> Hidden layer 2 (32 neurons)
        self.fc2 = nn.Linear(64, 32)
        # Hidden layer 1 (32 neurons) -> Hidden layer 2 (1 neuron)
        self.fc3 = nn.Linear(32, 1)

        # Activate function
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x) # Output layer so no activation
        return x

def train_model(model, X_train, y_train, X_val, y_val, X_test, y_test, target_scaler, num_epochs=1000, learning_rate=0.001, patience=50, min_delta=0.0001):
    """
    Trains the neural network model with early stopping based on validation loss
    and reports evaluation metrics on the test set.
    Args:
        model (nn.Module): The neural network model to train.
        X_train (torch.Tensor): Training features.
        y_train (torch.Tensor): Training target.
        X_val (torch.Tensor): Validation features.
        y_val (torch.Tensor): Validation target.
        X_test (torch.Tensor): Test features.
        y_test (torch.Tensor): Test target.
        target_scaler (MinMaxScaler): The scaler used for the target variable, needed for inverse transform.
        num_epochs (int): Maximum number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        patience (int): Number of epochs to wait for improvement before stopping.
        min_delta (float): Minimum change in monitored quantity to qualify as an improvement.
    """
    # Define a loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')
    epochs_no_improvement = 0
    early_stop = False
    best_model_state = None # To store the state_dict of the best model
    
    
    print(f"Starting model training for {num_epochs} epochs with early stopping (patience= {patience})...")

    try:
        with tqdm(range(num_epochs), desc="Training Progress") as pbar:
            for epochs in pbar:
                if early_stop:
                    break
                # Set model to training model
                model.train()

                # Forward pass (training)
                outputs = model(X_train)
                loss = criterion(outputs, y_train)

                # Backwards pass and optimize
                optimizer.zero_grad() # Clear the gradients
                loss.backward() # Computing gradients
                optimizer.step()

                # Evaluate on validation set
                model.eval() # Set the model to evaluation mode
                with torch.no_grad(): # Disable gradient calculation
                    val_outputs = model(X_val)
                    val_loss = criterion(val_outputs, y_val)

                # Early stopping logic based on validation loss
                if val_loss.item() <best_val_loss - min_delta:
                    best_val_loss = val_loss.item()
                    epochs_no_improvement = 0
                    best_model_state = model.state_dict() # Saving best model state
                else:
                    epochs_no_improvement += 1
                    if epochs_no_improvement == patience:
                        print(f"Early stopping triggered at epoch {epochs+1} (no improvement for {patience} epochs.)")
                        early_stop = True
                
                # Update tqdm post-fix with current loss values
                pbar.set_postfix_str(f"Train loss: {loss.item():.4f}, Val loss: {val_loss.item():.4f}")
        
        print("Training completed")

        # Load the best model state if early stopping occurred and best state was saved
        if best_model_state:
            model.load_state_dict(best_model_state)
            print("Loaded best model state for final evaluation and saving")
        else:
            # If no improvement ever found. (e.g., patience=0 or very small min_delta)
            # the last state of the model used
            print("No improvement found during training; using final model state for evaluation")

        
        # Final model evaluation for test set
        print("\n--- Final Model Evaluation on Test Set ---")
        model.eval()
        with torch.no_grad():
            # Get predictions on the test
            test_predictions_scaled = model(X_test).cpu().numpy()
            y_test_cpu = y_test.cpu().numpy()

            # Inverse transform predictions and actual values to original scale for interpretable metrics
            actual_prices = target_scaler.inverse_transform(y_test_cpu)
            predicted_prices = target_scaler.inverse_transform(test_predictions_scaled)

            # Calculate metrics
            final_test_mse = mean_squared_error(actual_prices, predicted_prices)
            mae = mean_absolute_error(actual_prices, predicted_prices)
            r2 = r2_score(actual_prices, predicted_prices)

            print(f"Test mse (Mean Squared Error): ${final_test_mse:.2f}(thousands squared)")
            print(f"Mean Absolute Error(MAE): ${mae:.2f} thousands")
            print(f"R-squared: {r2:.4f}")

    except Exception as e:
        print(f"\nAn e error occurred during training: {e}")
        print("Training terminated prematurely")


def save_model_as_onnx(model, onnx_path, feature_scaler, target_scaler, scaler_path):
    # Set model to evaluation mode before export
    model.eval()

    # Create a dummy input tensor for ONNX export and move to device
    dummy_input = torch.randn(1, 3, requires_grad=True).to(DEVICE)

    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            # The names to assign to the input nodes of the graph
            input_names=['input'],
            # The names to assign to the output nodes of the graph
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        print(f"Model successfully saved to ONNX format at: {onnx_path}")

        # Save the scalers
        with open(scaler_path, 'wb') as f:
            pickle.dump({
                "feature_scaler": feature_scaler,
                "target_scaler": target_scaler
            }, f)
        print(f"Scalers successfully saved to: {scaler_path}")

    except Exception as e:
        print(f"Error saving model to ONNX: {e}")


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
        "--predict",
        action="store_true",
        help="Load an ONNX model and make a prediction"
    )
    
    parser.add_argument(
        "--data-path",
        type=str,
        default=DATA_FILE,
        help=f"Path to CSV file (default {DATA_FILE})"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=MODEL_ONNX_FILE,
        help=f"Path to save/load the ONNX model (default: {MODEL_ONNX_FILE})"
    )
    
    parser.add_argument(
        "--scaler-path",
        type=str,
        default=SCALER_FILE,
        help=f"Path to save/load the MinMax Scaler (default: {SCALER_FILE})"
    )

    
    parser.add_argument(
        "--input-features",
        type=str,
        help="Comma-seperated input features for prediction (e.g '2500,4,2'). Required with --predict"
    )

    
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Number of training epochs (default: 1000). Only applicable with --train"
    )

    
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate for training (default: 0.001). Only applicable with --train"
    )

    
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Number of epochs to wait for improvement before early stopping (default: 50). Only applicable with --train"
    )

    
    parser.add_argument(
        "--min-delta",
        type=float,
        default=0.0001,
        help="Minimum change in test loss to qualify as an improvement for early stopping (default: 0.0001). Only applicable with --train"
    )

    args = parser.parse_args()

    if args.train:
        print("--- Training Mode ---")
        X_train, y_train, X_val, y_val, X_test, y_test, feature_scaler, target_scaler = load_and_preprocess_data(
            args.data_path
        )
        if X_train is not None:
            # Get a model for house price prediction
            model = HousePricePredictor().to(DEVICE)
            
            # Train the model
            train_model(model, X_train, y_train, X_val, y_val, X_test, y_test, target_scaler, args.epochs, args.lr, args.patience, args.min_delta)
            
            # Save the model as an ONNX file
            save_model_as_onnx(model, args.model_path, feature_scaler, target_scaler, args.scaler_path)

        else:
            print("Training aborted do to data loading issiues")

    elif args.predict:
         print("--- Prediction Mode ---")
    else:
        print("Please specify either --train or --predict")
        parser.print_help()