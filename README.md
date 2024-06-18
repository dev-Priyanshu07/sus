To implement a system where an ANN model is trained using historical data and then tested with real-time data received via a TCP socket, you can follow these steps:

### Step 1: Prepare Historical Data and Train the ANN Model

#### Training Script (`train_model.py`):

This script will load historical stock data, train an ANN model, and save the model and scaler for future use.

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

# Assuming the CSV has columns: 'open', 'high', 'low', 'close', 'volume'
X = data[['open', 'high', 'low', 'volume']].values
y = data['close'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the ANN model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))  # Output layer for regression

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Save the trained model and scaler
model.save('stock_model.h5')
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Model training complete and saved to stock_model.h5")
```

### Step 2: Set Up a TCP Server

This server script generates synthetic stock data at one-minute intervals and sends it to connected clients via a TCP socket.

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

### Step 3: Set Up a TCP Client

This client script connects to the server, receives real-time stock data, preprocesses it, and makes predictions using the trained ANN model.

#### Client Script (`client.py`):

```python
import socket
import json
import numpy as np
from tensorflow.keras.models import load_model
import pickle

# Load the trained model and scaler
model = load_model('stock_model.h5')
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

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
            features_scaled = scaler.transform(features)
            prediction = model.predict(features_scaled)

            print(f"Received data: {stock_data}")
            print(f"Predicted close price: {prediction[0][0]}")
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

- **Training Script**:
  - Loads historical data, preprocesses it, and trains an ANN model.
  - Saves the trained model and scaler for future use.

- **Server Script**:
  - Generates synthetic stock data at one-minute intervals and sends it to connected clients via TCP.

- **Client Script**:
  - Connects to the server, receives real-time stock data, preprocesses it, and makes predictions using the trained ANN model.

This setup enables you to train an ANN model on historical data and test it on real-time data sent from a TCP server. The client script will print the received stock data and the predicted close price.
