To handle the decryption of data within a `.js` file, you'll need to use appropriate libraries for RSA and AES decryption in JavaScript. In this example, I'll use the `node-forge` library, which provides comprehensive support for cryptographic operations in Node.js.

First, ensure you have `node-forge` installed in your project:

```bash
npm install node-forge
```

### Folder Structure

```
encryption_project/
│
├── encryption/
│   ├── __init__.py
│   ├── key_generation.py
│   ├── key_exchange.py
│   ├── data_encryption.py
│   ├── data_decryption.py
│
├── encrypted_data.json
├── main.py
└── decrypt.js
```

### 1. Modify `main.py` to Save Encrypted Data to `encrypted_data.json`

```python
# main.py
import json
from encryption.key_generation import generate_rsa_keys, generate_aes_key
from encryption.key_exchange import encrypt_aes_key_with_rsa, decrypt_aes_key_with_rsa
from encryption.data_encryption import encrypt_json_data
from encryption.data_decryption import decrypt_json_data

def main():
    # 1. Generate RSA and AES keys
    private_key, public_key = generate_rsa_keys()
    aes_key = generate_aes_key()

    # 2. Encrypt AES key using RSA public key
    encrypted_aes_key = encrypt_aes_key_with_rsa(aes_key, public_key)

    # 3. Encrypt JSON data using AES key
    data = {"result": "This is a test."}
    iv, encrypted_data = encrypt_json_data(data, aes_key)

    # Save data to encrypted_data.json
    encrypted_payload = {
        'iv': iv.hex(),
        'encrypted_data': encrypted_data.hex(),
        'encrypted_aes_key': encrypted_aes_key.hex(),
        'public_key': public_key.decode('utf-8'),
        'private_key': private_key.decode('utf-8')  # Typically, you wouldn't share the private key like this
    }

    with open('encrypted_data.json', 'w') as f:
        json.dump(encrypted_payload, f)

    print("Encrypted data saved to encrypted_data.json")

if __name__ == "__main__":
    main()
```

### 2. Create `decrypt.js` for Decryption

```javascript
// decrypt.js
const fs = require('fs');
const forge = require('node-forge');

function hexToBytes(hex) {
  const bytes = [];
  for (let c = 0; c < hex.length; c += 2) {
    bytes.push(parseInt(hex.substr(c, 2), 16));
  }
  return bytes;
}

// Read the encrypted data
const encryptedData = JSON.parse(fs.readFileSync('encrypted_data.json', 'utf8'));

const { iv, encrypted_data, encrypted_aes_key, public_key, private_key } = encryptedData;

// Convert hex strings to byte arrays
const ivBytes = hexToBytes(iv);
const encryptedDataBytes = hexToBytes(encrypted_data);
const encryptedAesKeyBytes = hexToBytes(encrypted_aes_key);

// Decrypt AES key with RSA private key
const privateKeyForge = forge.pki.privateKeyFromPem(private_key);
const aesKey = privateKeyForge.decrypt(forge.util.binary.raw.encode(encryptedAesKeyBytes), 'RSA-OAEP');

// Decrypt the data with the AES key
const decipher = forge.cipher.createDecipher('AES-CBC', aesKey);
decipher.start({ iv: forge.util.createBuffer(ivBytes) });
decipher.update(forge.util.createBuffer(encryptedDataBytes));
decipher.finish();
const decryptedData = decipher.output.toString('utf8');

// Parse and log the decrypted data
console.log(JSON.parse(decryptedData));
```

### Running the Code

1. First, generate and save the encrypted data by running `main.py`:

```bash
cd encryption_project
python main.py
```

2. Then, decrypt the data by running `decrypt.js`:

```bash
node decrypt.js
```

This setup ensures that data encrypted in Python can be securely decrypted in JavaScript using Node.js, utilizing the RSA and AES algorithms for key exchange and data encryption.
