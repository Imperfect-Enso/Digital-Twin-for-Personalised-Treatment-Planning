import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import pickle

from ml.lstm import build_lstm_model


def load_and_prepare_data(csv_path=None):
    """
    Reads patients.csv and reshapes it into model-ready arrays.

    Returns:
        X: shape (num_patients, 12, 4) — vital sequences
        y: shape (num_patients,)       — severity labels (0 = mild, 1 = severe)
    """
    if csv_path is None:
        BASE_DIR = Path(__file__).resolve().parent.parent
        csv_path = BASE_DIR / "data" / "patients.csv"

    print(f"\nLoading CSV from: {csv_path}")

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    if df["severity_label"].dtype == object:
        df["severity_label"] = df["severity_label"].map({"mild": 0, "severe": 1})

    vitals = ["blood_pressure", "glucose", "heart_rate", "inflammation"]

    X = []
    y = []

    for patient_id in df["patient_id"].unique():
        patient_rows = (
            df[df["patient_id"] == patient_id]
            .sort_values("timestep")
        )

        if len(patient_rows) != 12:
            raise ValueError(f"Patient {patient_id} does not have 12 timesteps.")

        sequence = patient_rows[vitals].values
        label    = patient_rows["severity_label"].iloc[-1]

        X.append(sequence)
        y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    print(f"Data loaded → X shape: {X.shape}, y shape: {y.shape}")
    print(f"Unique labels: {np.unique(y)}")

    return X, y


def scale_data(X_train, X_test):
    """
    Fits a StandardScaler on training data and applies it to both splits.
    Flattens to 2D for scaling, then reshapes back to 3D.
    """
    num_train, timesteps, features = X_train.shape
    num_test = X_test.shape[0]

    X_train_flat = X_train.reshape(-1, features)
    X_test_flat  = X_test.reshape(-1, features)

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_test_scaled  = scaler.transform(X_test_flat)

    X_train_scaled = X_train_scaled.reshape(num_train, timesteps, features)
    X_test_scaled  = X_test_scaled.reshape(num_test,  timesteps, features)

    return X_train_scaled, X_test_scaled, scaler


def train():
    """Full training pipeline — loads data, trains LSTM, saves model and scaler."""

    X, y = load_and_prepare_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTrain patients: {X_train.shape[0]}")
    print(f"Test patients:  {X_test.shape[0]}")

    X_train, X_test, scaler = scale_data(X_train, X_test)

    model = build_lstm_model(timesteps=12, num_features=4)

    models_dir = Path(__file__).resolve().parent.parent / "models"
    models_dir.mkdir(exist_ok=True)

    # Stop early if validation loss stops improving; save the best checkpoint
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
    )
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=models_dir / "lstm_model.keras",
        monitor="val_loss", save_best_only=True, verbose=1
    )

    print("\nTraining started...\n")

    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=16,
        validation_data=(X_test, y_test),
        callbacks=[early_stop, checkpoint],
        verbose=1
    )

    print("\nFinal evaluation on test set:")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    print(f"Test Loss:     {loss:.4f}")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    with open(models_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print("\nSaved → models/lstm_model.keras")
    print("Saved → models/scaler.pkl")
    print("\nTraining complete!")

    return model, history


if __name__ == "__main__":
    train()
