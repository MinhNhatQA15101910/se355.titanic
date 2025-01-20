# Artificial Neural Network

## Data Preprocessing
import numpy as np
import pandas as pd
import tensorflow as tf

dataset_train = pd.read_csv("../train.csv")
X_train = dataset_train.drop(["Survived"], axis=1)
y_train = dataset_train["Survived"]
X_test = pd.read_csv("../test.csv")

numeric_features = ["Age", "SibSp", "Parch", "Fare"]
categorical_features = ["Sex", "Pclass", "Embarked"]

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
)

from sklearn.preprocessing import OneHotEncoder

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

## Applying preprocessing
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

## Split the training data into training and validation sets
from sklearn.model_selection import train_test_split

X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_processed, y_train, test_size=0.2, random_state=42
)

## Build the ANN model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

ann = Sequential(
    [
        Dense(64, activation="relu", input_shape=(X_train_processed.shape[1],)),
        Dropout(0.3),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid"),
    ]
)

## Compile the model
ann.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

## Set up early stopping
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)

## Train the model
history = ann.fit(
    X_train_split,
    y_train_split,
    validation_data=(X_val_split, y_val_split),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=2,
)

## Evaluate the model on the training set
train_loss, train_accuracy = ann.evaluate(X_train_split, y_train_split, verbose=0)
print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")

## Evaluate the model on the validation set
val_loss, val_accuracy = ann.evaluate(X_val_split, y_val_split, verbose=0)
print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

## Make predictions on the test set
y_pred = (ann.predict(X_test_processed) > 0.5).astype(int).flatten()

## Export to csv
results = pd.DataFrame({"PassengerId": X_test["PassengerId"], "Survived": y_pred})
results.to_csv("ann_submission.csv", index=False)
