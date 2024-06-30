Sure! Let's go through each part of the code and explain it in detail.

### Importing Node-Fetch Library

```javascript
const forge = require('node-forge');
```

This line imports the `node-forge` library, which provides cryptographic functions including RSA key generation and AES encryption/decryption.

### RSA Key Pair Generation

```javascript
// Generate RSA key pair
const keyPair = forge.pki.rsa.generateKeyPair(2048);
const publicKeyPem = forge.pki.publicKeyToPem(keyPair.publicKey);
const privateKeyPem = forge.pki.privateKeyToPem(keyPair.privateKey);

console.log('Public Key:', publicKeyPem);
console.log('Private Key:', privateKeyPem);
```

1. **Generate RSA key pair**: This generates an RSA key pair (public and private keys) with a key size of 2048 bits.
2. **Convert keys to PEM format**: The keys are converted to PEM format, which is a base64 encoded format often used for storing and sharing cryptographic keys.
3. **Print keys**: The public and private keys are printed to the console. This is for demonstration purposes; in a real application, you would securely store and share these keys.

### AES Key Generation

```javascript
// Generate a random AES key
const aesKey = forge.random.getBytesSync(32); // AES-256
```

This generates a random 256-bit AES key, which will be used for AES encryption and decryption.

### AES Encryption Function

```javascript
// Function to encrypt data
function encryptData(data, key) {
  const iv = forge.random.getBytesSync(16);
  const cipher = forge.cipher.createCipher('AES-GCM', key);
  cipher.start({ iv: iv });
  cipher.update(forge.util.createBuffer(data, 'utf8'));
  cipher.finish();
  
  const encrypted = cipher.output.getBytes();
  const tag = cipher.mode.tag.getBytes();
  
  return { encryptedData: encrypted, iv: iv, tag: tag };
}
```

1. **Generate IV**: A random 16-byte initialization vector (IV) is generated.
2. **Create cipher**: An AES-GCM cipher is created using the provided key.
3. **Start cipher**: The cipher is started with the IV.
4. **Encrypt data**: The data is encrypted. It is first converted to a Forge buffer in UTF-8 format.
5. **Finish encryption**: Finalize the encryption process.
6. **Get encrypted data**: The encrypted data is extracted as bytes.
7. **Get authentication tag**: The authentication tag (used in AES-GCM for data integrity) is extracted.
8. **Return encrypted data**: The encrypted data, IV, and authentication tag are returned.

### AES Decryption Function

```javascript
// Function to decrypt data
function decryptData(encryptedData, key, iv, tag) {
  const decipher = forge.cipher.createDecipher('AES-GCM', key);
  decipher.start({ iv: iv, tag: tag });
  decipher.update(forge.util.createBuffer(encryptedData));
  const pass = decipher.finish();
  
  if (pass) {
    return decipher.output.toString('utf8');
  } else {
    throw new Error('Decryption failed');
  }
}
```

1. **Create decipher**: An AES-GCM decipher is created using the provided key.
2. **Start decipher**: The decipher is started with the IV and authentication tag.
3. **Decrypt data**: The encrypted data is decrypted. It is first converted to a Forge buffer.
4. **Finish decryption**: Finalize the decryption process.
5. **Check decryption success**: If decryption is successful, return the decrypted data as a UTF-8 string. Otherwise, throw an error.

### JSON Encryption Function

```javascript
function encryptJson(json, key, exemptFields) {
  const encryptedJson = {};

  for (const field in json) {
    if (exemptFields.includes(field)) {
      encryptedJson[field] = json[field];
    } else {
      const encrypted = encryptData(JSON.stringify(json[field]), key);
      encryptedJson[field] = encrypted;
    }
  }

  return encryptedJson;
}
```

1. **Initialize encrypted JSON**: Create an empty object to store the encrypted JSON.
2. **Iterate over fields**: Loop through each field in the input JSON.
3. **Check for exempt fields**: If the field is in the list of exempt fields, copy it as is.
4. **Encrypt non-exempt fields**: For other fields, encrypt the field's value and store the result.
5. **Return encrypted JSON**: Return the encrypted JSON object.

### JSON Decryption Function

```javascript
function decryptJson(encryptedJson, key, exemptFields) {
  const decryptedJson = {};

  for (const field in encryptedJson) {
    if (exemptFields.includes(field)) {
      decryptedJson[field] = encryptedJson[field];
    } else {
      const encrypted = encryptedJson[field];
      decryptedJson[field] = JSON.parse(decryptData(encrypted.encryptedData, key, encrypted.iv, encrypted.tag));
    }
  }

  return decryptedJson;
}
```

1. **Initialize decrypted JSON**: Create an empty object to store the decrypted JSON.
2. **Iterate over fields**: Loop through each field in the encrypted JSON.
3. **Check for exempt fields**: If the field is in the list of exempt fields, copy it as is.
4. **Decrypt non-exempt fields**: For other fields, decrypt the field's value and store the result.
5. **Return decrypted JSON**: Return the decrypted JSON object.

### Example Usage

```javascript
const exemptFields = ['company id', 'timestamp'];

const json = {
  "name": "John Doe",
  "age": 30,
  "company id": "123456",
  "timestamp": "2024-06-30T12:34:56Z"
};

// Encrypt the JSON data
const encryptedJson = encryptJson(json, aesKey, exemptFields);
console.log('Encrypted JSON:', JSON.stringify(encryptedJson, null, 2));

// Decrypt the JSON data
const decryptedJson = decryptJson(encryptedJson, aesKey, exemptFields);
console.log('Decrypted JSON:', JSON.stringify(decryptedJson, null, 2));
```

1. **Define exempt fields**: Specify the fields that should not be encrypted.
2. **Define input JSON**: Define the JSON object to be encrypted.
3. **Encrypt JSON**: Encrypt the JSON object, preserving the exempt fields.
4. **Print encrypted JSON**: Print the encrypted JSON object to the console.
5. **Decrypt JSON**: Decrypt the encrypted JSON object.
6. **Print decrypted JSON**: Print the decrypted JSON object to the console, verifying that it matches the original input.

By following these steps, you create a secure encryption and decryption system using RSA for key exchange and AES for data encryption, while ensuring specific fields are left unencrypted.
