Certainly! Let's switch the key exchange mechanism from Diffie-Hellman to RSA. RSA (Rivest–Shamir–Adleman) is an asymmetric encryption algorithm that uses a pair of keys: a public key for encryption and a private key for decryption.

### Overview

In RSA, one party generates a pair of keys (public and private). The public key is shared with the other party, which uses it to encrypt a message. The private key, which is kept secret, is used to decrypt the message. This ensures secure communication.

### Updated Modules

We'll update the necessary modules to use RSA for key exchange and encryption/decryption.

### File Structure

```
project/
│
├── analysis_engine.py
├── derive.py
├── exchange.py
├── process.py
├── security.py
└── websocket_module/
    └── websocket_client.py
```

### `exchange.py`

Handles RSA key generation, serialization, and shared secret computation.

#### `exchange.py`

```python
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization, hashes

# Step 1: Generate RSA private and public keys

# Generate RSA private key
private_key = rsa.generate_private_key(
    public_exponent=65537,
    key_size=2048
)

# Extract the public key from the private key
public_key = private_key.public_key()

# Step 2: Serialize public key for sharing

# Function to serialize the public key to bytes
def get_serialized_public_key():
    return public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )

# Step 3: Deserialize received public key

# Function to deserialize a received public key
def load_peer_public_key(peer_public_key_bytes):
    return serialization.load_pem_public_key(peer_public_key_bytes)

# Step 4: Encrypt and decrypt shared secret

# Function to encrypt data with a peer's public key
def encrypt_with_peer_public_key(peer_public_key, data):
    ciphertext = peer_public_key.encrypt(
        data,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    return ciphertext

# Function to decrypt data with our private key
def decrypt_with_private_key(ciphertext):
    plaintext = private_key.decrypt(
        ciphertext,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    return plaintext
```

### `derive.py`

Since RSA encryption is directly used for sharing secrets, we no longer need a key derivation function. This module can be omitted or kept for other cryptographic operations if needed.

### `security.py`

Contains functions for AES encryption, decryption, and DataFrame encryption/decryption.

#### `security.py`

```python
import pandas as pd
from Crypto.Cipher import AES
import base64

def encrypt_data(data, key):
    cipher = AES.new(key, AES.MODE_GCM)
    nonce = cipher.nonce
    ciphertext, tag = cipher.encrypt_and_digest(data.encode('utf-8'))
    return nonce, ciphertext, tag

def decrypt_data(nonce, ciphertext, tag, key):
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    data = cipher.decrypt_and_verify(ciphertext, tag)
    return data.decode('utf-8')

def encrypt_dataframe(df, key):
    encrypted_df = df.copy()
    nonces = {}
    tags = {}

    for column in df.columns:
        encrypted_values = []
        column_nonces = []
        column_tags = []
        
        for value in df[column]:
            nonce, ciphertext, tag = encrypt_data(str(value), key)
            encrypted_values.append(base64.b64encode(ciphertext).decode('utf-8'))
            column_nonces.append(base64.b64encode(nonce).decode('utf-8'))
            column_tags.append(base64.b64encode(tag).decode('utf-8'))
        
        encrypted_df[column] = encrypted_values
        nonces[column] = column_nonces
        tags[column] = column_tags

    return encrypted_df, nonces, tags

def decrypt_dataframe(encrypted_df, nonces, tags, key):
    decrypted_df = encrypted_df.copy()

    for column in encrypted_df.columns:
        decrypted_values = []
        
        for encrypted_value, nonce, tag in zip(encrypted_df[column], nonces[column], tags[column]):
            ciphertext = base64.b64decode(encrypted_value.encode('utf-8'))
            nonce = base64.b64decode(nonce.encode('utf-8'))
            tag = base64.b64decode(tag.encode('utf-8'))
            decrypted_value = decrypt_data(nonce, ciphertext, tag, key)
            decrypted_values.append(decrypted_value)
        
        decrypted_df[column] = decrypted_values

    return decrypted_df
```

### `process.py`

This file will now use RSA for key exchange, encryption, and decryption.

#### `process.py`

```python
import asyncio
import pandas as pd
from websocket_module.websocket_client import receive_data
import exchange
import security

async def handle_websocket(uri, analysis_func):
    while True:
        data = await receive_data(uri)
        
        # Convert the received data to a DataFrame
        df = pd.read_csv(pd.compat.StringIO(data))
        
        # Simulate peer public key exchange
        peer_public_key_bytes = exchange.get_serialized_public_key()
        peer_public_key = exchange.load_peer_public_key(peer_public_key_bytes)
        
        # Encrypt the DataFrame
        encrypted_df, nonces, tags = security.encrypt_dataframe(df, peer_public_key)
        
        # Convert encrypted DataFrame and metadata to JSON for transmission
        encrypted_data = {
            "encrypted_df": encrypted_df.to_json(),
            "nonces": nonces,
            "tags": tags
        }
        
        # Send encrypted data to analysis module
        analysis_func(encrypted_data, peer_public_key)

def analysis_func(encrypted_data, key):
    import analysis_engine
    analysis_engine.process_encrypted_data(encrypted_data, key)

if __name__ == "__main__":
    uri = "wss://example.com/data"  # Replace with your WebSocket URI
    asyncio.get_event_loop().run_until_complete(handle_websocket(uri, analysis_func))
```

### `analysis_engine.py`

Handles receiving encrypted data, decrypting it, and performing analysis.

#### `analysis_engine.py`

```python
import pandas as pd
import security

def process_encrypted_data(encrypted_data, key):
    # Extract encrypted DataFrame and metadata
    encrypted_df = pd.read_json(encrypted_data["encrypted_df"])
    nonces = encrypted_data["nonces"]
    tags = encrypted_data["tags"]
    
    # Decrypt the DataFrame
    decrypted_df = security.decrypt_dataframe(encrypted_df, nonces, tags, key)
    
    # Perform further analysis
    analysis_result = analyze_data(decrypted_df)
    print("Analysis Result:", analysis_result)

def analyze_data(df):
    # Placeholder for actual analysis logic
    return f"Analyzed DataFrame: {df}"
```

### Summary

1. **`exchange.py`**: Handles RSA key generation, serialization, and encryption/decryption of the shared secret.
2. **`security.py`**: Contains functions for AES encryption, decryption, and DataFrame encryption/decryption.
3. **`process.py`**: Uses RSA for key exchange, encrypts the DataFrame, and sends encrypted data to `analysis_engine.py`.
4. **`analysis_engine.py`**: Receives encrypted DataFrame, decrypts it, and performs analysis.
5. **`websocket_module/websocket_client.py`**: Handles WebSocket connections and data reception.

With RSA, the public and private keys are used for secure communication. The public key is used to encrypt the shared secret, and the private key is used to decrypt it. This setup ensures secure data exchange while keeping the column names intact for proper database storage and retrieval.
