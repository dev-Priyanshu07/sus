Certainly! Below is the complete code organized into different Python files based on their functionality. This modular approach makes it easier to manage and understand the code.

### Folder Structure

```
encryption_module/
│
├── key_generation.py
├── key_exchange.py
├── data_encryption.py
├── data_decryption.py
└── main.py
```

### 1. `key_generation.py`

```python
# key_generation.py
from Crypto.PublicKey import RSA
from Crypto.Random import get_random_bytes

def generate_rsa_keys():
    key = RSA.generate(2048)
    private_key = key.export_key()
    public_key = key.publickey().export_key()
    return private_key, public_key

def generate_aes_key():
    return get_random_bytes(32)  # AES-256
```

### 2. `key_exchange.py`

```python
# key_exchange.py
from Crypto.Cipher import PKCS1_OAEP
from Crypto.PublicKey import RSA

def encrypt_aes_key_with_rsa(aes_key, rsa_public_key):
    rsa_key = RSA.import_key(rsa_public_key)
    cipher_rsa = PKCS1_OAEP.new(rsa_key)
    encrypted_aes_key = cipher_rsa.encrypt(aes_key)
    return encrypted_aes_key

def decrypt_aes_key_with_rsa(encrypted_aes_key, rsa_private_key):
    rsa_key = RSA.import_key(rsa_private_key)
    cipher_rsa = PKCS1_OAEP.new(rsa_key)
    aes_key = cipher_rsa.decrypt(encrypted_aes_key)
    return aes_key
```

### 3. `data_encryption.py`

```python
# data_encryption.py
import json
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad

def encrypt_json_data(data, aes_key):
    json_data = json.dumps(data).encode('utf-8')
    cipher_aes = AES.new(aes_key, AES.MODE_CBC)
    ct_bytes = cipher_aes.encrypt(pad(json_data, AES.block_size))
    return cipher_aes.iv, ct_bytes
```

### 4. `data_decryption.py`

```python
# data_decryption.py
import json
from Crypto.Cipher import AES
from Crypto.Util.Padding import unpad

def decrypt_json_data(iv, encrypted_data, aes_key):
    cipher_aes = AES.new(aes_key, AES.MODE_CBC, iv)
    decrypted_data = unpad(cipher_aes.decrypt(encrypted_data), AES.block_size)
    return json.loads(decrypted_data.decode('utf-8'))
```

### 5. `main.py`

```python
# main.py
from key_generation import generate_rsa_keys, generate_aes_key
from key_exchange import encrypt_aes_key_with_rsa, decrypt_aes_key_with_rsa
from data_encryption import encrypt_json_data
from data_decryption import decrypt_json_data

# 1. Generate RSA and AES keys
private_key, public_key = generate_rsa_keys()
aes_key = generate_aes_key()

# 2. Encrypt AES key using RSA public key
encrypted_aes_key = encrypt_aes_key_with_rsa(aes_key, public_key)

# 3. Encrypt JSON data using AES key
data = {"result": "This is a test."}
iv, encrypted_data = encrypt_json_data(data, aes_key)

# 4. Decrypt AES key using RSA private key
decrypted_aes_key = decrypt_aes_key_with_rsa(encrypted_aes_key, private_key)

# 5. Decrypt JSON data using decrypted AES key
decrypted_data = decrypt_json_data(iv, encrypted_data, decrypted_aes_key)

print(decrypted_data)
```

### Running the Code

To run the code, navigate to the `encryption_module` directory and execute the `main.py` script:

```bash
cd encryption_module
python main.py
```

This will perform the key generation, key exchange, data encryption, and data decryption, printing the decrypted JSON data at the end.
