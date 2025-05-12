def text_to_binary(text):
    return ''.join(format(ord(char), '08b') for char in text)

def binary_to_text(binary):
    chars = [binary[i:i+8] for i in range(0, len(binary), 8)]
    return ''.join(chr(int(char, 2)) for char in chars)

def xor_encrypt(binary_message, secret_key):
    key_stream = (''.join(map(str, secret_key)) * ((len(binary_message) // len(secret_key)) + 1))[:len(binary_message)]
    encrypted = ''.join(str(int(b) ^ int(k)) for b, k in zip(binary_message, key_stream))
    return encrypted

