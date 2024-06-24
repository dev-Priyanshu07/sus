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
