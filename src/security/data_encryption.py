"""
Data Encryption & Credential Hashing
Secure handling of sensitive data in Energy Forecast system

Security Features:
- AES-256 encryption for cached data
- SHA-256 hashing for credentials
- Secure key generation and storage
- PBKDF2 key derivation
"""

import hashlib
import secrets
import json
from pathlib import Path
from typing import Optional, Any
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import base64
import logging

logger = logging.getLogger(__name__)


class DataEncryption:
    """
    Handles encryption/decryption of sensitive data
    
    Uses Fernet (symmetric encryption with AES-256)
    """
    
    def __init__(self, key_file: str = ".keys/encryption.key"):
        self.key_file = Path(key_file)
        self.key_file.parent.mkdir(exist_ok=True, parents=True)
        self.cipher = self._load_or_create_key()
    
    def _load_or_create_key(self) -> Fernet:
        """Load existing encryption key or create new one"""
        if self.key_file.exists():
            with open(self.key_file, 'rb') as f:
                key = f.read()
            logger.info("Loaded existing encryption key")
        else:
            # Generate new key
            key = Fernet.generate_key()
            with open(self.key_file, 'wb') as f:
                f.write(key)
            # Restrictive permissions
            import os
            os.chmod(self.key_file, 0o600)
            logger.info("Generated new encryption key")
        
        return Fernet(key)
    
    def encrypt_string(self, plaintext: str) -> str:
        """
        Encrypt string data
        
        Args:
            plaintext: String to encrypt
        
        Returns:
            Base64-encoded encrypted string
        """
        encrypted = self.cipher.encrypt(plaintext.encode())
        return base64.b64encode(encrypted).decode()
    
    def decrypt_string(self, encrypted: str) -> str:
        """
        Decrypt string data
        
        Args:
            encrypted: Base64-encoded encrypted string
        
        Returns:
            Original plaintext
        """
        decoded = base64.b64decode(encrypted.encode())
        decrypted = self.cipher.decrypt(decoded)
        return decrypted.decode()
    
    def encrypt_file(self, input_path: str, output_path: Optional[str] = None):
        """
        Encrypt entire file
        
        Args:
            input_path: Path to file to encrypt
            output_path: Path for encrypted file (default: input_path + .enc)
        """
        if output_path is None:
            output_path = f"{input_path}.enc"
        
        with open(input_path, 'rb') as f:
            plaintext = f.read()
        
        encrypted = self.cipher.encrypt(plaintext)
        
        with open(output_path, 'wb') as f:
            f.write(encrypted)
        
        logger.info(f"Encrypted file: {input_path} -> {output_path}")
    
    def decrypt_file(self, input_path: str, output_path: Optional[str] = None):
        """
        Decrypt entire file
        
        Args:
            input_path: Path to encrypted file
            output_path: Path for decrypted file
        """
        if output_path is None:
            output_path = input_path.replace('.enc', '')
        
        with open(input_path, 'rb') as f:
            encrypted = f.read()
        
        decrypted = self.cipher.decrypt(encrypted)
        
        with open(output_path, 'wb') as f:
            f.write(decrypted)
        
        logger.info(f"Decrypted file: {input_path} -> {output_path}")


class CredentialHasher:
    """
    Secure credential hashing using SHA-256 and salt
    
    Best practices for password/credential storage
    """
    
    @staticmethod
    def hash_password(password: str, salt: Optional[bytes] = None) -> tuple[str, str]:
        """
        Hash password with salt using SHA-256
        
        Args:
            password: Plain text password
            salt: Optional salt (generated if not provided)
        
        Returns:
            (hash_hex, salt_hex) tuple
        """
        if salt is None:
            salt = secrets.token_bytes(32)  # 32-byte random salt
        
        # Use PBKDF2 for key derivation (100,000 iterations)
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        
        hash_bytes = kdf.derive(password.encode())
        
        logger.info("Password hashed successfully")
        return hash_bytes.hex(), salt.hex()
    
    @staticmethod
    def verify_password(password: str, hash_hex: str, salt_hex: str) -> bool:
        """
        Verify password against stored hash
        
        Args:
            password: Plain text password to verify
            hash_hex: Stored password hash (hex)
            salt_hex: Stored salt (hex)
        
        Returns:
            True if password matches, False otherwise
        """
        salt = bytes.fromhex(salt_hex)
        computed_hash, _ = CredentialHasher.hash_password(password, salt)
        
        is_valid = secrets.compare_digest(computed_hash, hash_hex)
        
        if is_valid:
            logger.info("Password verification: SUCCESS")
        else:
            logger.warning("Password verification: FAILED")
        
        return is_valid
    
    @staticmethod
    def hash_api_key(api_key: str) -> str:
        """
        Hash API key for secure storage (one-way)
        
        Args:
            api_key: API key to hash
        
        Returns:
            SHA-256 hash (hex)
        """
        return hashlib.sha256(api_key.encode()).hexdigest()


class SecureCache:
    """
    Encrypted cache for sensitive data
    
    Stores data with encryption at rest
    """
    
    def __init__(self, cache_dir: str = ".cache/secure"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        self.encryptor = DataEncryption()
    
    def save(self, key: str, data: Any):
        """
        Save data to encrypted cache
        
        Args:
            key: Cache key (filename)
            data: Data to cache (will be JSON serialized)
        """
        cache_file = self.cache_dir / f"{key}.json.enc"
        
        # Serialize to JSON
        json_str = json.dumps(data)
        
        # Encrypt
        encrypted = self.encryptor.encrypt_string(json_str)
        
        # Save
        with open(cache_file, 'w') as f:
            f.write(encrypted)
        
        logger.info(f"Saved encrypted cache: {key}")
    
    def load(self, key: str) -> Optional[Any]:
        """
        Load data from encrypted cache
        
        Args:
            key: Cache key
        
        Returns:
            Cached data or None if not found
        """
        cache_file = self.cache_dir / f"{key}.json.enc"
        
        if not cache_file.exists():
            return None
        
        # Load encrypted data
        with open(cache_file, 'r') as f:
            encrypted = f.read()
        
        # Decrypt
        json_str = self.encryptor.decrypt_string(encrypted)
        
        # Deserialize
        data = json.loads(json_str)
        
        logger.info(f"Loaded encrypted cache: {key}")
        return data


# Global instances
encryptor = DataEncryption()
hasher = CredentialHasher()
secure_cache = SecureCache()


if __name__ == "__main__":
    print("🔐 Data Encryption & Hashing Demo\n")
    
    # 1. String Encryption
    print("1️⃣ String Encryption (AES-256):")
    secret_data = "Sensitive API Response Data"
    encrypted = encryptor.encrypt_string(secret_data)
    decrypted = encryptor.decrypt_string(encrypted)
    print(f"   Original:  {secret_data}")
    print(f"   Encrypted: {encrypted[:40]}... (truncated)")
    print(f"   Decrypted: {decrypted}")
    print(f"   ✅ Match: {secret_data == decrypted}\n")
    
    # 2. Password Hashing
    print("2️⃣ Password Hashing (PBKDF2 + SHA-256):")
    password = "SecurePassword123!"
    hash_val, salt = hasher.hash_password(password)
    print(f"   Password: {password}")
    print(f"   Hash: {hash_val[:32]}... (truncated)")
    print(f"   Salt: {salt[:32]}... (truncated)")
    
    # Verify correct password
    is_valid = hasher.verify_password(password, hash_val, salt)
    print(f"   ✅ Correct password: {is_valid}")
    
    # Verify wrong password
    is_valid_wrong = hasher.verify_password("WrongPassword", hash_val, salt)
    print(f"   ❌ Wrong password: {not is_valid_wrong}\n")
    
    # 3. API Key Hashing
    print("3️⃣ API Key Hashing (SHA-256):")
    api_key = "sk_live_51abc123xyz"
    hashed_key = hasher.hash_api_key(api_key)
    print(f"   API Key: {api_key}")
    print(f"   Hashed:  {hashed_key[:32]}... (truncated)")
    print(f"   ✅ One-way hash (cannot be reversed)\n")
    
    # 4. Secure Cache
    print("4️⃣ Secure Encrypted Cache:")
    test_data = {"api_response": "sensitive_data", "timestamp": "2025-11-26"}
    secure_cache.save("test_cache", test_data)
    loaded_data = secure_cache.load("test_cache")
    print(f"   Saved:  {test_data}")
    print(f"   Loaded: {loaded_data}")
    print(f"   ✅ Match: {test_data == loaded_data}")