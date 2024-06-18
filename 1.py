import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
import pandas_ta as ta
import matplotlib.pyplot as plt

def load_and_preprocess_data(filepath):
    # Load the dataset
    dataset = pd.read_csv(filepath)

    # Calculate technical indicators
    dataset.ta.sma(length=10, append=True)
    dataset.ta.rsi(length=14, append=True)
    dataset.ta.macd(append=True)

    # Handle missing values from technical indicators
    dataset.fillna(0, inplace=True)

    # Split the dataset into features and target
    X = dataset[['Open', 'High', 'Close', 'Low', 'Volume', 'SMA_10', 'RSI_14', 'MACD']]
    y = dataset['Buy/Sell']
    
    return X, y

def build_model(input_dim):
    # Create the model
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=input_dim))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

def preprocess_external_data(filepath, scaler):
    # Load the external data
    external_data = pd.read_csv(filepath)

    # Preprocess the external data
    external_data.ta.sma(length=10, append=True)
    external_data.ta.rsi(length=14, append=True)
    external_data.ta.macd(append=True)

    # Handle missing values
    external_data.fillna(0, inplace=True)

    # Select the same features as the training data
    X_external = external_data[['Open', 'High', 'Close', 'Low', 'Volume', 'SMA_10', 'RSI_14', 'MACD']]

    # Scale the external data using the previously fitted scaler
    X_external_scaled = scaler.transform(X_external)
    
    return external_data, X_external_scaled

# Load and preprocess training data
X, y = load_and_preprocess_data('stock_data.csv')

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the model
model = build_model(X_train.shape[1])

# Setup early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Preprocess the external data
external_data, X_external_scaled = preprocess_external_data('external_data.csv', scaler)

# Predict the buy/sell using the trained model
predictions = model.predict(X_external_scaled)
predictions = [1 if p >= 0.5 else 0 for p in predictions]

# Add predictions to external_data for visualization
external_data['Predictions'] = predictions

# Plot the pattern of buy/sell formed by external data
plt.figure(figsize=(14, 7))
plt.plot(external_data.index, external_data['Close'], label='Close Price', color='blue')
plt.scatter(external_data.index, external_data['Close'][external_data['Predictions'] == 1], label='Buy Signal', marker='^', color='green')
plt.scatter(external_data.index, external_data['Close'][external_data['Predictions'] == 0], label='Sell Signal', marker='v', color='red')
plt.title('Buy/Sell Signals on External Data')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()
