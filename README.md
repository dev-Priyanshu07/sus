Certainly! Here’s how you can modify the code to avoid converting data into numpy arrays and instead work directly with Pandas DataFrames.

### Step 1: Install Necessary Libraries

Ensure the required libraries are installed:

```bash
pip install numpy pandas matplotlib tensorflow scikit-learn
```

### Step 2: Import Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
```

### Step 3: Load Data from CSV

Assume your CSV file is named `stock_data.csv` and has a column `Close` for the closing prices.

```python
# Load the data
data = pd.read_csv('stock_data.csv')

# Ensure the 'Date' column is parsed as datetime (if present) and set as the index
if 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

# Select the 'Close' price column
data = data[['Close']]
```

### Step 4: Data Preprocessing

```python
# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
data['Close'] = scaler.fit_transform(data[['Close']])

# Create the training and test datasets
train_data = data.iloc[:int(len(data) * 0.95)]
test_data = data.iloc[int(len(data) * 0.95):]

# Create x_train and y_train
x_train, y_train = [], []
for i in range(60, len(train_data)):
    x_train.append(train_data['Close'].iloc[i-60:i].values)
    y_train.append(train_data['Close'].iloc[i])

x_train, y_train = pd.DataFrame(x_train), pd.Series(y_train)

# Create x_test and y_test
x_test, y_test = [], []
for i in range(60, len(test_data)):
    x_test.append(test_data['Close'].iloc[i-60:i].values)
    y_test.append(test_data['Close'].iloc[i])

x_test, y_test = pd.DataFrame(x_test), pd.Series(y_test)
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
predictions = scaler.inverse_transform(predictions)
```

### Step 7: Visualize the Results

```python
# Create a dataframe for visualization
train = data[:int(len(data) * 0.95)]
valid = data[int(len(data) * 0.95):]
valid['Predictions'] = predictions

# Visualize the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date')
plt.ylabel('Close Price USD ($)')
plt.plot(train['Close'], label='Train')
plt.plot(valid['Close'], label='Val')
plt.plot(valid['Predictions'], label='Predictions')
plt.legend(loc='lower right')
plt.show()
```

### Step 8: Predict the Next 10 Days

To predict the next 10 days, we need to extend the model slightly:

```python
# Get the last 60 days of the closing price
last_60_days = data['Close'].iloc[-60:].values.reshape(-1, 1)

# Scale the data
last_60_days_scaled = scaler.transform(last_60_days)

# Create an empty list to store the predicted prices
predicted_prices = []

# Predict the next 10 days
for _ in range(10):
    X_test = pd.DataFrame(last_60_days_scaled).transpose()
    
    # Get the predicted scaled price
    pred_price = model.predict(X_test)
    
    # Undo the scaling
    pred_price_unscaled = scaler.inverse_transform(pred_price)
    predicted_prices.append(pred_price_unscaled[0, 0])
    
    # Append the predicted price to the last_60_days_scaled array
    last_60_days_scaled = np.append(last_60_days_scaled, pred_price, axis=0)
    last_60_days_scaled = last_60_days_scaled[1:].reshape(-1, 1)

# Print the predicted prices for the next 10 days
print("Predicted prices for the next 10 days:")
for i, price in enumerate(predicted_prices, 1):
    print(f"Day {i}: {price}")
```

This code avoids converting data into numpy arrays and uses Pandas DataFrames directly for preprocessing and prediction. Adjust the epochs, batch size, and other hyperparameters as needed for better results.
