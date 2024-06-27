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









Understood. Here's an updated version of the code where the RSA keys are generated once and reused. 

### Python Code for Encryption

```python
import json
import base64
from Crypto.Cipher import AES, PKCS1_OAEP
from Crypto.PublicKey import RSA
from Crypto.Random import get_random_bytes

# RSA keys (generated once and reused)
PRIVATE_KEY = """-----BEGIN RSA PRIVATE KEY-----
MIIEowIBAAKCAQEA...
-----END RSA PRIVATE KEY-----"""
PUBLIC_KEY = """-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8A...
-----END PUBLIC KEY-----"""

# Function to encrypt data using AES and RSA
def encrypt_data(data, public_key):
    # Generate a random AES key
    aes_key = get_random_bytes(32)  # 256 bits for AES-256
    cipher_aes = AES.new(aes_key, AES.MODE_EAX)
    
    # Encrypt the data with AES
    ciphertext, tag = cipher_aes.encrypt_and_digest(json.dumps(data).encode('utf-8'))
    
    # Encrypt the AES key with the RSA public key
    rsa_public_key = RSA.import_key(public_key)
    cipher_rsa = PKCS1_OAEP.new(rsa_public_key)
    enc_aes_key = cipher_rsa.encrypt(aes_key)
    
    # Combine encrypted AES key and ciphertext
    encrypted_data = {
        'enc_aes_key': base64.b64encode(enc_aes_key).decode('utf-8'),
        'nonce': base64.b64encode(cipher_aes.nonce).decode('utf-8'),
        'tag': base64.b64encode(tag).decode('utf-8'),
        'ciphertext': base64.b64encode(ciphertext).decode('utf-8')
    }
    
    return json.dumps(encrypted_data)

# Example usage
data = {'column1': 'value1', 'column2': 'value2'}
encrypted_data = encrypt_data(data, PUBLIC_KEY)
print(encrypted_data)
```

### JavaScript Code for Decryption

```javascript
const NodeRSA = require('node-rsa');
const CryptoJS = require('crypto-js');

// RSA private key (generated once and reused)
const PRIVATE_KEY = `-----BEGIN RSA PRIVATE KEY-----
MIIEowIBAAKCAQEA...
-----END RSA PRIVATE KEY-----`;

// Function to decrypt data using AES and RSA
function decryptData(encryptedData, privateKey) {
    // Parse the encrypted data
    let data = JSON.parse(encryptedData);
    
    // Decode the base64 encoded values
    let encAESKey = Buffer.from(data.enc_aes_key, 'base64');
    let nonce = Buffer.from(data.nonce, 'base64');
    let tag = Buffer.from(data.tag, 'base64');
    let ciphertext = Buffer.from(data.ciphertext, 'base64');
    
    // Decrypt the AES key with the RSA private key
    let key = new NodeRSA(privateKey);
    let aesKey = key.decrypt(encAESKey, 'buffer');
    
    // Decrypt the data with AES
    let decrypted = CryptoJS.AES.decrypt(
        {
            ciphertext: CryptoJS.lib.WordArray.create(ciphertext),
            salt: CryptoJS.lib.WordArray.create(nonce),
            iv: CryptoJS.lib.WordArray.create(tag)
        },
        aesKey.toString('base64'),
        {
            mode: CryptoJS.mode.EAX,
            format: CryptoJS.format.OpenSSL,
            padding: CryptoJS.pad.Pkcs7
        }
    );
    
    // Convert the decrypted data to a string and parse it as JSON
    let decryptedData = CryptoJS.enc.Utf8.stringify(decrypted);
    return JSON.parse(decryptedData);
}

// Example usage
let encryptedData = `{
    "enc_aes_key": "...",
    "nonce": "...",
    "tag": "...",
    "ciphertext": "..."
}`;

let decryptedData = decryptData(encryptedData, PRIVATE_KEY);
console.log(decryptedData);
```

### Notes:
1. Replace the placeholder values (`MIIEowIBAAKCAQEA...`) in `PRIVATE_KEY` and `PUBLIC_KEY` with your actual RSA keys.
2. Ensure the private key is securely stored and not exposed to the client side in a real-world application.
3. The `pycryptodome` and `crypto-js` libraries are used for encryption and decryption.
4. Handle the AES key and data encryption details correctly as per the AES mode used.

With these updates, the RSA keys are generated once and reused for encryption and decryption, ensuring a consistent and secure key management approach.


Sure! Here’s how you can replace the `CryptoJS` library with `forge` in the JavaScript decryption function. First, you need to install the `node-forge` library:

```bash
npm install node-forge
```

Here's the updated JavaScript code using `node-forge`:

### JavaScript Code for Decryption with `node-forge`

```javascript
const NodeRSA = require('node-rsa');
const forge = require('node-forge');

// RSA private key (generated once and reused)
const PRIVATE_KEY = `-----BEGIN RSA PRIVATE KEY-----
MIIEowIBAAKCAQEA...
-----END RSA PRIVATE KEY-----`;

// Function to decrypt data using AES and RSA
function decryptData(encryptedData, privateKey) {
    // Parse the encrypted data
    let data = JSON.parse(encryptedData);
    
    // Decode the base64 encoded values
    let encAESKey = forge.util.decode64(data.enc_aes_key);
    let nonce = forge.util.decode64(data.nonce);
    let tag = forge.util.decode64(data.tag);
    let ciphertext = forge.util.decode64(data.ciphertext);
    
    // Decrypt the AES key with the RSA private key
    let key = new NodeRSA(privateKey);
    let aesKey = key.decrypt(encAESKey, 'buffer');
    
    // Convert AES key to forge format
    let aesKeyForge = forge.util.createBuffer(aesKey, 'binary');
    
    // Decrypt the data with AES
    let decipher = forge.cipher.createDecipher('AES-GCM', aesKeyForge);
    decipher.start({ iv: nonce, tag: forge.util.createBuffer(tag, 'binary') });
    decipher.update(forge.util.createBuffer(ciphertext, 'binary'));
    let success = decipher.finish();
    
    if (!success) {
        throw new Error('Decryption failed');
    }
    
    // Convert the decrypted data to a string and parse it as JSON
    let decryptedData = decipher.output.toString('utf8');
    return JSON.parse(decryptedData);
}

// Example usage
let encryptedData = `{
    "enc_aes_key": "...",
    "nonce": "...",
    "tag": "...",
    "ciphertext": "..."
}`;

try {
    let decryptedData = decryptData(encryptedData, PRIVATE_KEY);
    console.log(decryptedData);
} catch (error) {
    console.error('Error decrypting data:', error);
}
```

### Explanation

1. **Decoding Base64**:
    - The `forge.util.decode64` method is used to decode the base64 encoded values for the encrypted AES key, nonce, tag, and ciphertext.

2. **RSA Decryption**:
    - The AES key is decrypted using the RSA private key with the `node-rsa` library.

3. **AES Decryption**:
    - The decrypted AES key is used to create a decipher in `forge`.
    - The nonce and tag are used as parameters for the `AES-GCM` mode in `forge`.
    - The ciphertext is decrypted using the `forge` AES decipher.

4. **Error Handling**:
    - The decryption process checks if the decryption was successful and throws an error if it fails.

This setup replaces `CryptoJS` with `forge` for the decryption process in JavaScript, ensuring that the decryption logic matches the encryption logic in Python.
