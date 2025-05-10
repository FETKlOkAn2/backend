"""
Encryption utilities for securely handling API keys and other sensitive information
"""
import os
import base64
import logging
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Setup logger
logger = logging.getLogger(__name__)

# Environment variable for encryption key
# In production, this should be securely stored and managed (e.g. with a secrets manager)
ENCRYPTION_KEY = os.environ.get('ENCRYPTION_KEY', 'THIS_IS_A_DEVELOPMENT_KEY_REPLACE_IN_PRODUCTION')
SALT = os.environ.get('ENCRYPTION_SALT', 'THIS_IS_A_DEVELOPMENT_SALT_REPLACE_IN_PRODUCTION').encode()

def generate_key(password):
    """
    Generate a Fernet key from password and salt
    
    Args:
        password: Password to derive key from
        
    Returns:
        bytes: The derived key
    """
    password = password.encode()
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=SALT,
        iterations=100000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(password))
    return key

def encrypt_api_key(api_key, user_secret=None):
    """
    Encrypt an API key or secret
    
    Args:
        api_key: The API key/secret to encrypt
        user_secret: Optional additional user-specific secret for encryption
        
    Returns:
        str: Encrypted API key as a base64 string
    """
    try:
        if not api_key:
            logger.warning("Attempting to encrypt empty API key")
            return None
            
        # Use the global encryption key with optional user-specific secret
        password = ENCRYPTION_KEY
        if user_secret:
            password = f"{ENCRYPTION_KEY}_{user_secret}"
            
        key = generate_key(password)
        f = Fernet(key)
        
        # Convert to bytes and encrypt
        encrypted_data = f.encrypt(api_key.encode())
        
        # Return as a base64 string
        return base64.b64encode(encrypted_data).decode('utf-8')
    except Exception as e:
        logger.error(f"Encryption error: {str(e)}")
        raise

def decrypt_api_key(encrypted_api_key, user_secret=None):
    """
    Decrypt an API key or secret
    
    Args:
        encrypted_api_key: The encrypted API key/secret (base64 string)
        user_secret: Optional additional user-specific secret used for encryption
        
    Returns:
        str: Decrypted API key
    """
    try:
        if not encrypted_api_key:
            logger.warning("Attempting to decrypt empty encrypted API key")
            return None
            
        # Use the global encryption key with optional user-specific secret
        password = ENCRYPTION_KEY
        if user_secret:
            password = f"{ENCRYPTION_KEY}_{user_secret}"
            
        key = generate_key(password)
        f = Fernet(key)
        
        # Decode from base64 string and decrypt
        encrypted_data = base64.b64decode(encrypted_api_key)
        decrypted_data = f.decrypt(encrypted_data)
        
        return decrypted_data.decode('utf-8')
    except Exception as e:
        logger.error(f"Decryption error: {str(e)}")
        raise