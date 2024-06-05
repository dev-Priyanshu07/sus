To achieve this, we can set up an inter-process communication (IPC) mechanism to send the supertrend values calculated in `supertrend.py` to `strategy.py`. There are several ways to do this, such as using sockets, shared memory, or message queues. For simplicity and robustness, we'll use a basic socket communication approach, which should work well given your existing use of sockets.

Here's a step-by-step outline of how to set this up:

1. **Modify `supertrend.py` to send supertrend values via a socket:**
    - Create a client socket in `supertrend.py`.
    - Connect this socket to a server socket in `strategy.py`.
    - Send the calculated supertrend values through this socket.

2. **Set up `strategy.py` to receive supertrend values:**
    - Create a server socket in `strategy.py`.
    - Listen for incoming connections and receive supertrend values.
    - Process these values to generate buy or sell signals.

### `supertrend.py`

Here's how you might modify `supertrend.py`:

```python
import socket
import json
import time

def calculate_supertrend(data):
    # Dummy implementation; replace with actual calculation
    supertrend_value = sum(data) / len(data)
    return supertrend_value

def send_supertrend_to_strategy(supertrend_value):
    try:
        # Create a socket object
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Define the port and host
        host = 'localhost'
        port = 12345
        # Connect to the server
        s.connect((host, port))
        # Send the supertrend value
        message = json.dumps({'supertrend': supertrend_value})
        s.sendall(message.encode('utf-8'))
        # Close the connection
        s.close()
    except Exception as e:
        print(f"Error sending data: {e}")

def main():
    while True:
        # Replace with the actual method to receive data
        data = [1, 2, 3, 4, 5]  # Dummy data
        supertrend_value = calculate_supertrend(data)
        send_supertrend_to_strategy(supertrend_value)
        time.sleep(5)  # Simulate regular interval

if __name__ == "__main__":
    main()
```

### `strategy.py`

Here's how you might set up `strategy.py`:

```python
import socket
import json

def process_supertrend(supertrend_value):
    # Dummy strategy; replace with actual strategy logic
    if supertrend_value > 50:
        return "buy"
    else:
        return "sell"

def main():
    # Create a socket object
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Define the port and host
    host = 'localhost'
    port = 12345
    # Bind the socket to the port
    server_socket.bind((host, port))
    # Listen for incoming connections
    server_socket.listen(5)

    print("Strategy server listening on port", port)

    while True:
        # Accept a connection
        client_socket, addr = server_socket.accept()
        print("Got connection from", addr)
        try:
            # Receive data from the client
            data = client_socket.recv(1024)
            if not data:
                break
            message = data.decode('utf-8')
            supertrend_value = json.loads(message)['supertrend']
            signal = process_supertrend(supertrend_value)
            print(f"Supertrend: {supertrend_value}, Signal: {signal}")
        except Exception as e:
            print(f"Error receiving data: {e}")
        finally:
            client_socket.close()

if __name__ == "__main__":
    main()
```

### Explanation:

1. **`supertrend.py`**:
    - A client socket is created to connect to `strategy.py`.
    - The calculated supertrend value is serialized to JSON and sent over the socket.

2. **`strategy.py`**:
    - A server socket is created to listen for incoming connections.
    - When a connection is accepted, the supertrend value is received, deserialized, and processed to determine a buy or sell signal.

This setup ensures that `strategy.py` can receive and process supertrend values in real-time as they are calculated by `supertrend.py`. Adjust the host and port as necessary for your environment.
