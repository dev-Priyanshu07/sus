You're right; if the RSA public key is constant and securely known by both the sender and the receiver, there's no need to transmit it with each payload. We can streamline the process by storing and reusing the public key and only transmitting the `encrypted_aes_key` and the `encrypted_data`.

### Step-by-Step Solution

#### 1. Generate and Save RSA Keys

First, generate and save the RSA key pair. This is a one-time setup step.

##### `generate_keys.py`

```python
from encryption.key_generation import generate_rsa_keys

def save_rsa_keys(private_key_file='private_key.pem', public_key_file='public_key.pem'):
    private_key, public_key = generate_rsa_keys()

    # Save the private key
    with open(private_key_file, 'wb') as f:
        f.write(private_key)

    # Save the public key
    with open(public_key_file, 'wb') as f:
        f.write(public_key)

if __name__ == "__main__":
    save_rsa_keys()
```

Run this script once to generate and save the RSA keys:

```bash
python generate_keys.py
```

#### 2. Modify Encryption Script to Use Saved RSA Keys

Update the encryption script to load the saved RSA public key and exclude it from the output payload.

##### `encrypt.py`

```python
import json
from encryption.key_generation import generate_aes_key
from encryption.key_exchange import encrypt_aes_key_with_rsa
from encryption.data_encryption import encrypt_json_data

def load_public_key(public_key_file='public_key.pem'):
    with open(public_key_file, 'rb') as f:
        public_key = f.read()
    return public_key

def encrypt_json_values(data, aes_key):
    encrypted_data = {}
    for key, value in data.items():
        iv, encrypted_value = encrypt_json_data({key: value}, aes_key)
        encrypted_data[key] = {
            'iv': iv.hex(),
            'encrypted_value': encrypted_value.hex()
        }
    return encrypted_data

def encrypt_data(data):
    # Load the RSA public key
    public_key = load_public_key()

    # Generate AES key
    aes_key = generate_aes_key()

    # Encrypt AES key using RSA public key
    encrypted_aes_key = encrypt_aes_key_with_rsa(aes_key, public_key)

    # Encrypt only the values of the JSON data using AES key
    encrypted_values = encrypt_json_values(data, aes_key)

    # Prepare the encrypted payload
    encrypted_payload = {
        'encrypted_data': encrypted_values,
        'encrypted_aes_key': encrypted_aes_key.hex()
    }

    return encrypted_payload

if __name__ == "__main__":
    # Assume result contains the data to be encrypted
    result = {"result": "This is a test."}
    encrypted_payload = encrypt_data(result)
    print(json.dumps(encrypted_payload))
```

#### 3. Modify Decryption Script to Use Saved RSA Private Key

Update the decryption script to load the saved RSA private key.

##### `decrypt.js`

```javascript
const fs = require('fs');
const forge = require('node-forge');

function hexToBytes(hex) {
  const bytes = [];
  for (let c = 0; c < hex.length; c += 2) {
    bytes.push(parseInt(hex.substr(c, 2), 16));
  }
  return bytes;
}

process.stdin.setEncoding('utf8');

let inputData = '';

process.stdin.on('data', (chunk) => {
  inputData += chunk;
});

process.stdin.on('end', () => {
  const encryptedData = JSON.parse(inputData);

  const { encrypted_data, encrypted_aes_key } = encryptedData;

  // Load the private key from file
  const privateKeyPem = fs.readFileSync('private_key.pem', 'utf8');
  const privateKey = forge.pki.privateKeyFromPem(privateKeyPem);

  // Decrypt AES key with RSA private key
  const encryptedAesKeyBytes = hexToBytes(encrypted_aes_key);
  const aesKey = privateKey.decrypt(forge.util.binary.raw.encode(encryptedAesKeyBytes), 'RSA-OAEP');

  // Decrypt the values with the AES key
  const decryptedData = {};
  for (const key in encrypted_data) {
    const { iv, encrypted_value } = encrypted_data[key];
    const ivBytes = hexToBytes(iv);
    const encryptedValueBytes = hexToBytes(encrypted_value);

    const decipher = forge.cipher.createDecipher('AES-CBC', aesKey);
    decipher.start({ iv: forge.util.createBuffer(ivBytes) });
    decipher.update(forge.util.createBuffer(encryptedValueBytes));
    decipher.finish();

    decryptedData[key] = JSON.parse(decipher.output.toString('utf8'))[key];
  }

  // Log the decrypted data
  console.log(decryptedData);
});
```

### Running the Code

1. **Generate and Save RSA Keys (once):**

   ```bash
   python generate_keys.py
   ```

2. **Encrypt the Data in Python:**

   Run the Python script to encrypt the data. This will output the encrypted payload to the standard output.

   ```bash
   python encrypt.py > encrypted_payload.json
   ```

   If you want to pass the output directly to the Node.js script without saving it to a file, you can use a pipe:

   ```bash
   python encrypt.py | node decrypt.js
   ```

3. **Decrypt the Data in Node.js:**

   The `decrypt.js` script reads the encrypted data from the standard input, so it can be directly piped from the Python script as shown above.

By separating the key generation and usage steps, and by storing the RSA keys securely, you ensure that the same RSA public key is used for all encryption operations. This approach minimizes the overhead and ensures that only the necessary information (the encrypted AES key and the encrypted data) is transmitted each time.
