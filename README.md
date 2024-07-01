Let's modify the code to generate the RSA key once, save it to files, and then read from these files for encryption and decryption.

### File: `keygen.js`

This file generates the RSA key pair and saves them to files.

```javascript
const forge = require('node-forge');
const fs = require('fs');

// Generate RSA key pair
const keyPair = forge.pki.rsa.generateKeyPair(2048);
const publicKeyPem = forge.pki.publicKeyToPem(keyPair.publicKey);
const privateKeyPem = forge.pki.privateKeyToPem(keyPair.privateKey);

// Save public key to file
fs.writeFileSync('publicKey.pem', publicKeyPem);
console.log('Public key saved to publicKey.pem');

// Save private key to file
fs.writeFileSync('privateKey.pem', privateKeyPem);
console.log('Private key saved to privateKey.pem');
```

### File: `encryption.js`

This file contains the encryption functions and reads the RSA public key from a file.

```javascript
const forge = require('node-forge');
const fs = require('fs');

// Read public key from file
const publicKeyPem = fs.readFileSync('publicKey.pem', 'utf8');
const publicKey = forge.pki.publicKeyFromPem(publicKeyPem);

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

// Function to encrypt JSON
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

// Function to encrypt AES key with RSA public key
function encryptAESKey(aesKey) {
  const encryptedAESKey = publicKey.encrypt(aesKey, 'RSA-OAEP', {
    md: forge.md.sha256.create(),
    mgf1: {
      md: forge.md.sha1.create()
    }
  });
  return encryptedAESKey;
}

module.exports = {
  encryptJson,
  encryptAESKey
};
```

### File: `decryption.js`

This file contains the decryption functions and reads the RSA private key from a file.

```javascript
const forge = require('node-forge');
const fs = require('fs');

// Read private key from file
const privateKeyPem = fs.readFileSync('privateKey.pem', 'utf8');
const privateKey = forge.pki.privateKeyFromPem(privateKeyPem);

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

// Function to decrypt JSON
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

// Function to decrypt AES key with RSA private key
function decryptAESKey(encryptedAESKey) {
  const aesKey = privateKey.decrypt(encryptedAESKey, 'RSA-OAEP', {
    md: forge.md.sha256.create(),
    mgf1: {
      md: forge.md.sha1.create()
    }
  });
  return aesKey;
}

module.exports = {
  decryptJson,
  decryptAESKey
};
```

### File: `main.js`

This file demonstrates the usage of the encryption and decryption functions.

```javascript
const forge = require('node-forge');
const { encryptJson, encryptAESKey } = require('./encryption');
const { decryptJson, decryptAESKey } = require('./decryption');

// Generate a random AES key
const aesKey = forge.random.getBytesSync(32); // AES-256

// Encrypt AES key with RSA public key
const encryptedAESKey = encryptAESKey(aesKey);
console.log('Encrypted AES Key:', encryptedAESKey);

// Example Usage
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

// Decrypt AES key with RSA private key
const decryptedAESKey = decryptAESKey(encryptedAESKey);
console.log('Decrypted AES Key:', decryptedAESKey);

// Decrypt the JSON data
const decryptedJson = decryptJson(encryptedJson, decryptedAESKey, exemptFields);
console.log('Decrypted JSON:', JSON.stringify(decryptedJson, null, 2));
```

### Explanation

1. **keygen.js**:
   - Generates an RSA key pair.
   - Saves the public and private keys to `publicKey.pem` and `privateKey.pem` files.

2. **encryption.js**:
   - Reads the RSA public key from the `publicKey.pem` file.
   - Contains the `encryptData`, `encryptJson`, and `encryptAESKey` functions.

3. **decryption.js**:
   - Reads the RSA private key from the `privateKey.pem` file.
   - Contains the `decryptData`, `decryptJson`, and `decryptAESKey` functions.

4. **main.js**:
   - Demonstrates the usage of the encryption and decryption functions.
   - Generates an AES key and encrypts it using the RSA public key.
   - Encrypts a JSON object, then decrypts the AES key using the RSA private key.
   - Decrypts the JSON object and prints the results.

This approach ensures that the RSA key pair is generated once and can be reused by reading the keys from files for encryption and decryption.
