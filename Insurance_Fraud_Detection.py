import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras

# Load dataset
df = pd.read_csv('claims_data.csv')

# Explore data
print(df.head())
print(df.info())
print(df.describe())

# Preprocess data
X = df.drop(['id', 'fraudulent'], axis=1)  # features
y = df['fraudulent']  # target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# **Machine Learning Approach (Random Forest)**
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("Random Forest Metrics:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

# **Deep Learning Approach (Neural Network) - Optional**
if True:  # Set to False to skip deep learning
    # Normalize data
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build neural network model
    nn_model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train neural network
    nn_model.fit(X_train_scaled, y_train, epochs=10, batch_size=128, validation_data=(X_test_scaled, y_test))

    # Evaluate neural network
    nn_loss, nn_acc = nn_model.evaluate(X_test_scaled, y_test)
    print("Neural Network Metrics:")
    print("Accuracy:", nn_acc)
