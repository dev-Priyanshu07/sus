To generate and send stock data using plain TCP sockets at one-minute intervals, you can use Python's `socket` module. Here's how you can set up a TCP server and client:

### Server Code

1. **Server to generate stock data and send to clients:**

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

### Client Code

2. **Client to receive and print stock data:**

```python
import socket

def start_client(host='localhost', port=65432):
    """
    Starts a TCP client to receive and print stock data from the server.
    """
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))

    try:
        while True:
            data = client_socket.recv(1024)
            if not data:
                break
            print(data.decode('utf-8').strip())
    finally:
        client_socket.close()

if __name__ == "__main__":
    start_client()
```

### Explanation

- **Server Code**:
  - **generate_stock_data**: This function generates random stock data.
  - **start_server**: This function starts a TCP server that listens for client connections. It sends generated stock data every minute to connected clients. If the connection is lost, it handles the exception and continues to listen for new connections.
  
- **Client Code**:
  - **start_client**: This function connects to the server and receives stock data. It prints the received data to the console.

### Running the Server and Client

1. **Run the server**:
   ```bash
   python server.py
   ```

2. **Run the client**:
   ```bash
   python client.py
   ```

This setup allows you to generate and send stock data at one-minute intervals using plain TCP sockets. The client receives and prints the stock data from the server.
