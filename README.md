Sure, let's modify the approach to use a simple Artificial Neural Network (ANN) instead of an LSTM. We'll use a feedforward neural network with fully connected layers. Here's the complete code:

### Step 1: Install Necessary Libraries

Ensure the required libraries are installed:

```bash
pip install numpy pandas matplotlib tensorflow yfinance scikit-learn
```

### Step 2: Import Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
```

### Step 3: Download Historical Stock Data

```python
# Download historical data for a stock (e.g., Apple Inc.)
ticker = 'AAPL'
data = yf.download(ticker, start='2010-01-01', end='2023-01-01')
```

### Step 4: Data Preprocessing

```python
# Select the 'Close' price column
data = data[['Close']]

# Convert the dataframe to a numpy array
dataset = data.values

# Get the number of rows to train the model on
training_data_len = int(np.ceil(len(dataset) * .95))

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Create the training data set
train_data = scaled_data[0:int(training_data_len), :]

# Create the test data set
test_data = scaled_data[training_data_len:, :]

# Split the data into x_train and y_train data sets
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

# Convert x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Split the data into x_test and y_test data sets
x_test = []
y_test = dataset[training_data_len:, :]

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

# Convert x_test to numpy array
x_test = np.array(x_test)
```

### Step 5: Build the ANN Model

```python
# Build the ANN model
model = Sequential()
model.add(Dense(units=50, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dropout(0.2))
model.add(Dense(units=25, activation='relu'))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=10)
```

### Step 6: Make Predictions

```python
# Get the model's predicted price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
```

### Step 7: Visualize the Results

```python
# Create a dataframe for visualization
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

# Visualize the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date')
plt.ylabel('Close Price USD ($)')
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()
```

### Step 8: Predict the Next 10 Days

To predict the next 10 days, we need to extend the model slightly:

```python
# Get the last 60 days of the closing price
last_60_days = data[-60:].values

# Scale the data
last_60_days_scaled = scaler.transform(last_60_days)

# Create an empty list to store the predicted prices
predicted_prices = []

# Predict the next 10 days
for _ in range(10):
    X_test = np.array([last_60_days_scaled])
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
    
    # Get the predicted scaled price
    pred_price = model.predict(X_test)
    
    # Undo the scaling
    pred_price_unscaled = scaler.inverse_transform(pred_price)
    predicted_prices.append(pred_price_unscaled[0, 0])
    
    # Append the predicted price to the last_60_days_scaled array
    last_60_days_scaled = np.append(last_60_days_scaled, pred_price, axis=0)
    last_60_days_scaled = last_60_days_scaled[1:]

# Print the predicted prices for the next 10 days
print("Predicted prices for the next 10 days:")
for i, price in enumerate(predicted_prices, 1):
    print(f"Day {i}: {price}")
```

This code provides a complete implementation for predicting stock prices using a feedforward ANN model. Adjust the epochs, batch size, and other hyperparameters as needed for better results.
