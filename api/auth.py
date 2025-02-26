# auth.py
from fastapi import HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from typing import Optional
from sjm_package.db.db import connection_pool
import logging
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import base64
import os
import requests

# Configure logging
logger = logging.getLogger(__name__)
# In your logging configuration
logging.basicConfig(
    level=logging.DEBUG,  # or INFO, depending on how verbose you want the logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("auth.log"),
        logging.StreamHandler()
    ]
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Get encryption key from environment variable
ENCRYPTION_KEY = base64.b64decode(os.getenv('ENCRYPTION_KEY'))

def decrypt_api_key(encrypted_key: str, iv: str, auth_tag: str) -> str:
    """
    Decrypt an API key using AES-GCM.
    """
    try:
        # Debug logging
        logger.debug(f"Decryption attempt:")
        logger.debug(f"Encrypted key (first 10 chars): {encrypted_key[:10]}...")
        logger.debug(f"IV: {iv}")
        logger.debug(f"Auth tag: {auth_tag}")

        # Convert hex strings back to bytes
        iv_bytes = bytes.fromhex(iv)
        encrypted_bytes = bytes.fromhex(encrypted_key)
        auth_tag_bytes = bytes.fromhex(auth_tag)

        logger.debug("Successfully converted hex to bytes")

        # Combine encrypted data and auth tag
        combined_data = encrypted_bytes + auth_tag_bytes
        
        logger.debug(f"Combined data length: {len(combined_data)} bytes")

        # Create AESGCM cipher
        aesgcm = AESGCM(ENCRYPTION_KEY)
        logger.debug("Created AESGCM cipher")
        
        # Decrypt data
        decrypted = aesgcm.decrypt(
            nonce=iv_bytes,
            data=combined_data,
            associated_data=None
        )
        
        result = decrypted.decode('utf-8')
        logger.debug(f"Decryption successful. Result starts with: {result[:10]}...")
        return result

    except Exception as e:
        logger.error(f"Decryption error details: {str(e)}")
        return None

async def get_api_key(api_key_header: str = Security(api_key_header)) -> Optional[str]:
    # Log the incoming API key
    logger.debug(f"Received API key: {api_key_header}")
    
    # Check if API key is provided
    if not api_key_header:
        logger.warning("No API key provided in request")
        raise HTTPException(
            status_code=401,
            detail={
                "error": "Authentication failed",
                "message": "No API key provided"
            }
        )
    
    # Extract the prefix (first two parts of the API key)
    prefix = '_'.join(api_key_header.split('_')[:2]) + '_'
    logger.debug(f"API key prefix: {prefix}")
    
    # Validate API key format
    if not api_key_header.startswith(('sjm_fr_', 'sjm_pr_', 'sjm_ent_')):
        logger.warning(f"Invalid API key format: {api_key_header[:6]}...")
        raise HTTPException(
            status_code=401,
            detail={
                "error": "Authentication failed",
                "message": "Invalid API key format"
            }
        )
    
    conn = connection_pool.get_connection()
    cursor = conn.cursor(dictionary=True)
    
    try:
        # Determine the plan type from the prefix
        plan_type = prefix[4:-1]  # Extract 'fr', 'pr', or 'ent'
        
        # First, get all potential keys for this plan type without filtering active status
        cursor.execute("""
            SELECT 
                ak.*,
                u.plan_type,
                p.request_limit
            FROM api_keys ak 
            JOIN users u ON ak.user_id = u.id
            JOIN plans p ON p.code = ak.plan_type
            WHERE ak.plan_type = %s
        """, (plan_type,))

        potential_matches = cursor.fetchall()
        logger.debug(f"Found {len(potential_matches)} potential matching keys")

        if not potential_matches:
            logger.warning(f"No API key found matching plan type: {plan_type}")
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "Authentication failed",
                    "message": "API key not found"
                }
            )

        # Try to find the exact matching key first
        for key_data in potential_matches:
            logger.debug(f"Checking key ID: {key_data['id']}")
            
            # Try to decrypt and match
            decrypted_key = decrypt_api_key(
                key_data['api_key'],
                key_data['iv'],
                key_data['auth_tag']
            )
            
            if decrypted_key is None:
                logger.error(f"Decryption failed for key ID: {key_data['id']}")
                continue
            
            # Check if this is the key we're looking for
            if decrypted_key == api_key_header:
                logger.debug("API key match found!")
                
                # Now check if the matched key is active
                if not key_data['is_active']:
                    logger.warning(f"API key matched but inactive: {key_data['id']}")
                    raise HTTPException(
                        status_code=403,
                        detail={
                            "error": "Authentication failed",
                            "message": "API key is inactive"
                        }
                    )
                
                # Key is both matching and active, update last_used_at timestamp
                cursor.execute("""
                    UPDATE api_keys 
                    SET last_used_at = CURRENT_TIMESTAMP 
                    WHERE id = %s
                """, (key_data['id'],))
                conn.commit()
                
                # Remove sensitive data before returning
                del key_data['api_key']
                del key_data['iv']
                del key_data['auth_tag']
                
                logger.info(f"Successfully authenticated API key ID: {key_data['id']}")
                # Return a comprehensive dictionary
                return {
                    'api_key': api_key_header,  # Full API key
                    'id': key_data['id'],
                    'user_id': key_data['user_id'],
                    'plan_type': key_data['plan_type'],
                    'request_limit': key_data['request_limit']
                }

        # If we get here, no matching key was found
        logger.warning("No matching API key found after trying all potential matches")
        raise HTTPException(
            status_code=401,
            detail={
                "error": "Authentication failed",
                "message": "Invalid API key"
            }
        )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in API key validation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Authentication failed",
                "message": "Internal server error during authentication"
            }
        )
    finally:
        cursor.close()
        conn.close()