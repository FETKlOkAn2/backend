"""
Enhanced API Key Testing Utilities
Tests API keys against trading platforms to verify they are valid with improved security and reliability
"""
import logging
import hmac
import hashlib
import time
import base64
import requests
import json
from urllib.parse import urlencode
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)

class ApiKeyTester:
    """
    Class to test API keys for various trading platforms with improved security and error handling
    """
    
    @staticmethod
    def test_coinbase_api_keys(api_key: str, api_secret: str, passphrase: str) -> Dict[str, Any]:
        """
        Test Coinbase API keys with improved error handling
        
        Args:
            api_key: Coinbase API key
            api_secret: Coinbase API secret
            passphrase: Coinbase API passphrase
            
        Returns:
            dict: Test result with status and message
        """
        try:
            # Input validation
            if not all([api_key, api_secret, passphrase]):
                return {"status": "error", "message": "Missing required Coinbase credentials"}
            
            # Coinbase Pro API endpoint for account information - non-destructive call
            base_url = "https://api.pro.coinbase.com"
            endpoint = "/accounts"
            
            # Get timestamp for signature
            timestamp = str(int(time.time()))
            
            # Create prehash string
            method = "GET"
            request_path = endpoint
            body = ""
            
            message = timestamp + method + request_path + body
            
            # Create HMAC signature with error handling for base64 decoding
            try:
                signature = hmac.new(
                    base64.b64decode(api_secret),
                    message.encode('utf-8'),
                    digestmod=hashlib.sha256
                )
                signature_b64 = base64.b64encode(signature.digest()).decode('utf-8')
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Invalid API secret format: {str(e)}"
                }
            
            # Set headers
            headers = {
                'CB-ACCESS-KEY': api_key,
                'CB-ACCESS-SIGN': signature_b64,
                'CB-ACCESS-TIMESTAMP': timestamp,
                'CB-ACCESS-PASSPHRASE': passphrase,
                'Content-Type': 'application/json'
            }
            
            # Make request with improved error handling
            try:
                response = requests.get(base_url + endpoint, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    return {
                        "status": "success",
                        "message": "Coinbase API keys validated successfully"
                    }
                elif response.status_code == 401:
                    return {
                        "status": "error",
                        "message": "Coinbase authentication failed: Invalid API credentials",
                        "code": response.status_code
                    }
                else:
                    error_message = "Failed to validate Coinbase API keys"
                    try:
                        error_data = response.json()
                        if 'message' in error_data:
                            error_message = f"Coinbase error: {error_data['message']}"
                    except:
                        error_message = f"Coinbase error: HTTP {response.status_code}"
                    
                    return {
                        "status": "error",
                        "message": error_message,
                        "code": response.status_code
                    }
            except requests.exceptions.Timeout:
                return {
                    "status": "error",
                    "message": "Connection to Coinbase timed out. Please try again later."
                }
            except requests.exceptions.ConnectionError:
                return {
                    "status": "error",
                    "message": "Cannot connect to Coinbase API. Please check your internet connection."
                }
                
        except Exception as e:
            logger.error(f"Error testing Coinbase API keys: {str(e)}")
            return {
                "status": "error",
                "message": "Error testing API keys: Internal server error"
            }
            
    @staticmethod
    def test_kraken_api_keys(api_key: str, api_secret: str) -> Dict[str, Any]:
        """
        Test Kraken API keys with improved error handling
        
        Args:
            api_key: Kraken API key
            api_secret: Kraken API secret
            
        Returns:
            dict: Test result with status and message
        """
        try:
            # Input validation
            if not all([api_key, api_secret]):
                return {"status": "error", "message": "Missing required Kraken credentials"}
            
            # Kraken API endpoint for account balance (non-destructive, read-only)
            base_url = "https://api.kraken.com"
            endpoint = "/0/private/Balance"
            
            # Create API signature
            nonce = str(int(time.time() * 1000))
            
            # Create signature
            post_data = {
                'nonce': nonce
            }
            
            post_data_encoded = urlencode(post_data)
            encoded = (str(post_data['nonce']) + post_data_encoded).encode()
            
            try:
                message = endpoint.encode() + hashlib.sha256(encoded).digest()
                
                signature = hmac.new(
                    base64.b64decode(api_secret),
                    message,
                    hashlib.sha512
                )
                signature_digest = base64.b64encode(signature.digest()).decode()
            except Exception as e:
                return {
                    "status": "error",
                    "message": f"Invalid Kraken API secret format: {str(e)}"
                }
            
            # Set headers
            headers = {
                'API-Key': api_key,
                'API-Sign': signature_digest
            }
            
            # Make request with improved error handling
            try:
                response = requests.post(
                    base_url + endpoint,
                    headers=headers,
                    data=post_data,
                    timeout=10
                )
                
                # Parse response
                result = response.json()
                
                if response.status_code == 200 and len(result.get('error', [])) == 0:
                    return {
                        "status": "success",
                        "message": "Kraken API keys validated successfully"
                    }
                else:
                    error_message = "Failed to validate Kraken API keys"
                    if result.get('error') and len(result['error']) > 0:
                        error_message = f"Kraken error: {', '.join(result['error'])}"
                    
                    return {
                        "status": "error",
                        "message": error_message
                    }
            except requests.exceptions.Timeout:
                return {
                    "status": "error",
                    "message": "Connection to Kraken timed out. Please try again later."
                }
            except requests.exceptions.ConnectionError:
                return {
                    "status": "error",
                    "message": "Cannot connect to Kraken API. Please check your internet connection."
                }
            except json.JSONDecodeError:
                return {
                    "status": "error",
                    "message": "Received invalid response from Kraken"
                }
                
        except Exception as e:
            logger.error(f"Error testing Kraken API keys: {str(e)}")
            return {
                "status": "error",
                "message": "Error testing API keys: Internal server error"
            }
    
    @classmethod
    def test_api_connection(cls, platform: str, api_key: str, api_secret: str, passphrase: Optional[str] = None) -> Dict[str, Any]:
        """
        Test API connection for a specific platform with improved validation
        
        Args:
            platform: Trading platform name
            api_key: API key
            api_secret: API secret
            passphrase: Optional passphrase (required for some platforms)
            
        Returns:
            dict: Test result with status and message
        """
        # Input validation
        if not platform or not isinstance(platform, str):
            return {"status": "error", "message": "Invalid platform specified"}
        
        if not api_key or not api_secret:
            return {"status": "error", "message": "API key and secret are required"}
            
        platform = platform.lower().strip()
        
        # Platform-specific validation
        try:
            if platform == "coinbase":
                if not passphrase:
                    return {
                        "status": "error",
                        "message": "Passphrase is required for Coinbase API"
                    }
                return cls.test_coinbase_api_keys(api_key, api_secret, passphrase)
                
            elif platform == "kraken":
                return cls.test_kraken_api_keys(api_key, api_secret)
                
            elif platform == "robinhood":
                # This is a placeholder as noted in your original code
                # In production, implement proper Robinhood API testing
                return {
                    "status": "error",
                    "message": "Robinhood API testing is not implemented. Please add your credentials directly."
                }
                
            else:
                return {
                    "status": "error",
                    "message": f"Unsupported platform: {platform}"
                }
                
        except Exception as e:
            logger.error(f"Error in test_api_connection for {platform}: {str(e)}")
            return {
                "status": "error",
                "message": "Error testing connection: Internal server error"
            }