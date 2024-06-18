Yes, you can use two machine learning models in synchronization for your stock prediction system. The first model will predict the stock closing prices for the next 10 days, and the second model will determine the uptrend or downtrend based on those predictions. Here’s how you can implement it:

1. **Train the first model to predict stock closing prices for the next 10 days.**
2. **Train the second model to predict uptrend or downtrend based on the 10-day predictions.**
3. **Set up a server to generate real-time stock data and send it to a client.**
4. **Set up a client to receive the data, make predictions using both models, and determine the trend.**

### Step 1: Train the First Model to Predict Closing Prices

#### Training Script for First Model (`train_model_prices.py`):

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import pickle

# Load historical stock data
data = pd.read_csv('historical_stock_data.csv')

# Assuming the CSV has columns: 'open', 'high', 'low', 'close', 'volume'
X = data[['open', 'high', 'low', 'volume']].values
y = data['close'].values

# Prepare the dataset for LSTM
def create_dataset(X, y, time_steps=10):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 10
X, y = create_dataset(X, y, time_steps)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Save the trained model
model.save('stock_price_model.h5')

print("Stock price prediction model training complete and saved to stock_price_model.h5")
```

### Step 2: Train the Second Model to Predict Uptrend or Downtrend

#### Training Script for Second Model (`train_model_trend.py`):

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pickle

# Load historical stock data
data = pd.read_csv('historical_stock_data.csv')

# Assuming the CSV has columns: 'close'
y = data['close'].values

# Generate the target variable for trend (1 for uptrend, 0 for downtrend)
def generate_trend(y, time_steps=10):
    trend = []
    for i in range(len(y) - time_steps):
        if y[i + time_steps] > y[i]:
            trend.append(1)
        else:
            trend.append(0)
    return np.array(trend)

time_steps = 10
y_trend = generate_trend(y, time_steps)

# Prepare the dataset for the trend model
def create_dataset_for_trend(y, time_steps=10):
    Xs = []
    for i in range(len(y) - time_steps):
        Xs.append(y[i:(i + time_steps)])
    return np.array(Xs)

X_trend = create_dataset_for_trend(y, time_steps)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_trend, y_trend, test_size=0.2, random_state=42)

# Build the ANN model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Save the trained model
model.save('stock_trend_model.h5')

print("Stock trend prediction model training complete and saved to stock_trend_model.h5")
```

### Step 3: Set Up a TCP Server

#### Server Script (`server.py`):

```python
import socket
import time
import random
import datetime
import json

def generate_stock_data(ticker):
    """
    Simulates generating stock data for a given ticker symbol.
    """
    open_price = round(random.uniform(100.0, 500.0), 2)
    close_price = round(open_price + random.uniform(-10.0, 10.0), 2)
    high_price = round(max(open_price, close_price) + random.uniform(0.0, 5.0), 2)
    low_price = round(min(open_price, close_price) - random.uniform(0.0, 5.0), 2)
    volume = random.randint(1000, 10000)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return {
        "timestamp": timestamp,
        "ticker": ticker,
        "open": open_price,
        "high": high_price,
        "low": low_price,
        "close": close_price,
        "volume": volume
    }

def start_server(host='localhost', port=65432):
    """
    Starts a TCP server to send stock data to connected clients every minute.
    """
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen()

    print(f"Server started at {host}:{port}")

    try:
        while True:
            client_socket, addr = server_socket.accept()
            print(f"Connection from {addr}")
            try:
                while True:
                    stock_data = generate_stock_data("AAPL")
                    stock_data_json = json.dumps(stock_data)
                    client_socket.sendall(stock_data_json.encode('utf-8') + b'\n')
                    time.sleep(60)  # Wait for 1 minute
            except (ConnectionResetError, BrokenPipeError):
                print(f"Connection lost with {addr}")
            finally:
                client_socket.close()
    finally:
        server_socket.close()

if __name__ == "__main__":
    start_server()
```

### Step 4: Set Up a TCP Client and Use Both Models

#### Client Script (`client.py`):

```python
import socket
import json
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Load the trained models
price_model = load_model('stock_price_model.h5')
trend_model = load_model('stock_trend_model.h5')

# Create an empty DataFrame to store real-time data
real_time_df = pd.DataFrame(columns=['timestamp', 'ticker', 'open', 'high', 'low', 'close', 'volume', 'predicted_close'])

def start_client(host='localhost', port=65432):
    """
    Starts a TCP client to receive and make predictions on stock data from the server.
    """
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))

    try:
        while True:
            data = client_socket.recv(1024)
            if not data:
                break

            stock_data = json.loads(data.decode('utf-8').strip())
            features = np.array([[stock_data['open'], stock_data['high'], stock_data['low'], stock_data['volume']]])
            features = features.reshape((1, 1, features.shape[1]))
            price_prediction = price_model.predict(features)[0][0]

            # Add the received data and price prediction to the DataFrame
            stock_data['predicted_close'] = price_prediction
            real_time_df.loc[len(real_time_df)] = stock_data

            # Use the last 10 days of predicted prices for trend prediction
            if len(real_time_df) >= 10:
                recent_prices = real_time_df['predicted_close'][-10:].values.reshape(1, -1)
                trend_prediction = trend_model.predict(recent_prices)
                trend = 'Uptrend' if trend_prediction[0][0] > 0.5 else 'Downtrend'

                print(f"Received data: {stock_data}")
                print(f"Predicted close price: {price_prediction}")
                print(f"Predicted trend: {trend}")
            else:
                print(f"Received data: {stock_data}")
                print(f"Predicted close price: {price_prediction}")
                print("Insufficient data for trend prediction")

    finally:
        client_socket.close()

if __name__ == "__main__":
    start_client()
```

### Running the Server and Client

1. **Run the server**:
   ```bash
   python server.py
   ```

2. **Run the client**:
   ```bash
   python client.py
   ```

### Explanation

- **First Model (Price Prediction)**:
  - Loads historical data, preprocesses it, and trains an LSTM model to predict stock closing prices for the next 10 days.
 
