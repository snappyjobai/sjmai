from fastapi import HTTPException, Request
import time
import os
import redis
import sqlite3
from pathlib import Path
import hashlib
from mysql.connector import pooling
from ..db.db import connection_pool

class RateLimit:
    def __init__(self):
        # Add more robust environment detection
        self.environment = os.getenv("ENVIRONMENT", "development").lower()
        
         # Add these class-level constants if not already defined
        self.AUTH_WINDOW = 900  # 15 minutes
        self.MAX_AUTH_ATTEMPTS = 5
        self.BLOCK_DURATION = 1800  # 30 minutes
        
        # Initialize db_path for SQLite
        self.db_path = "rate_limits.db"
        
        # Call SQLite initialization
        self._init_sqlite()
        
        # Add logging
        import logging
        self.logger = logging.getLogger(__name__)
        
        # More explicit Redis configuration
        if self.environment == "production":
            try:
                self.redis = redis.Redis(
                    host=os.getenv("REDIS_HOST", "localhost"),
                    port=int(os.getenv("REDIS_PORT", 6379)),
                    db=0,
                    decode_responses=True,
                    socket_timeout=5,  # Add timeout
                    socket_connect_timeout=5  # Add connection timeout
                )
                # Test Redis connection
                self.redis.ping()
            except Exception as e:
                self.logger.error(f"Redis connection error: {e}")
                # Fallback to SQLite if Redis fails
                self.environment = "development"
        
    def _init_sqlite(self):
        try:
            # Ensure directory exists
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Use context manager for safer DB operations
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # More robust table creation
                cursor.executescript("""
                    -- Enable foreign key support
                    PRAGMA foreign_keys = ON;
                    
                    CREATE TABLE IF NOT EXISTS rate_limits (
                        api_key TEXT,
                        month TEXT,
                        request_count INTEGER,
                        PRIMARY KEY (api_key, month)
                    ) WITHOUT ROWID;
                    
                    CREATE TABLE IF NOT EXISTS auth_attempts (
                        identifier TEXT PRIMARY KEY,
                        attempts INTEGER DEFAULT 0,
                        last_attempt TIMESTAMP,
                        blocked_until TIMESTAMP
                    ) WITHOUT ROWID;
                """)
                
                conn.commit()
        except sqlite3.Error as e:
            self.logger.error(f"SQLite initialization error: {e}")
            raise
        
    def _get_identifier(self, request: Request, api_key: str = None):
        """Creates a unique identifier for tracking auth attempts"""
        ip = request.client.host
        if api_key:
            # Include API key prefix in the identifier
            prefix = '_'.join(api_key.split('_')[:2]) + '_'
            return hashlib.sha256(f"{ip}:{prefix}".encode()).hexdigest()
        return hashlib.sha256(ip.encode()).hexdigest()

    def _check_auth_attempts(self, request: Request, api_key: str = None):
        """Check if the IP/API key combination is allowed to make attempts"""
        identifier = self._get_identifier(request, api_key)
        current_time = int(time.time())

        if self.environment == "production":
            # Redis implementation
            attempts_key = f"auth_attempts:{identifier}"
            blocked_key = f"auth_blocked:{identifier}"
            
            # Check if blocked
            if self.redis.exists(blocked_key):
                block_ttl = self.redis.ttl(blocked_key)
                raise HTTPException(
                    status_code=429,
                    detail=f"Too many authentication attempts. Try again in {block_ttl} seconds"
                )
            
            # Check and update attempts
            attempts = int(self.redis.get(attempts_key) or 0)
            if attempts >= self.MAX_AUTH_ATTEMPTS:
                self.redis.setex(blocked_key, self.BLOCK_DURATION, 1)
                self.redis.delete(attempts_key)
                raise HTTPException(
                    status_code=429,
                    detail=f"Too many authentication attempts. Try again in {self.BLOCK_DURATION} seconds"
                )
            
            self.redis.incr(attempts_key)
            self.redis.expire(attempts_key, self.AUTH_WINDOW)
        
        else:
            # SQLite implementation
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            try:
                cursor.execute("""
                    SELECT attempts, blocked_until
                    FROM auth_attempts
                    WHERE identifier = ?
                """, (identifier,))
                
                result = cursor.fetchone()
                
                if result:
                    attempts, blocked_until = result
                    blocked_until = blocked_until or 0
                    
                    # Check if blocked
                    if blocked_until > current_time:
                        remaining = blocked_until - current_time
                        raise HTTPException(
                            status_code=429,
                            detail=f"Too many authentication attempts. Try again in {remaining} seconds"
                        )
                    
                    # Reset if window expired
                    if current_time - (blocked_until or 0) > self.AUTH_WINDOW:
                        attempts = 0
                    
                    if attempts >= self.MAX_AUTH_ATTEMPTS:
                        new_blocked_until = current_time + self.BLOCK_DURATION
                        cursor.execute("""
                            UPDATE auth_attempts
                            SET attempts = 0, blocked_until = ?
                            WHERE identifier = ?
                        """, (new_blocked_until, identifier))
                        conn.commit()
                        raise HTTPException(
                            status_code=429,
                            detail=f"Too many authentication attempts. Try again in {self.BLOCK_DURATION} seconds"
                        )
                    
                    cursor.execute("""
                        UPDATE auth_attempts
                        SET attempts = attempts + 1,
                            last_attempt = ?
                        WHERE identifier = ?
                    """, (current_time, identifier))
                else:
                    cursor.execute("""
                        INSERT INTO auth_attempts (identifier, attempts, last_attempt)
                        VALUES (?, 1, ?)
                    """, (identifier, current_time))
                
                conn.commit()
            
            finally:
                conn.close()

    def _reset_auth_attempts(self, request: Request, api_key: str):
        """Reset the auth attempts counter after successful authentication"""
        identifier = self._get_identifier(request, api_key)
        
        if self.environment == "production":
            self.redis.delete(f"auth_attempts:{identifier}")
        else:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            try:
                cursor.execute("""
                    UPDATE auth_attempts
                    SET attempts = 0, blocked_until = NULL
                    WHERE identifier = ?
                """, (identifier,))
                conn.commit()
            finally:
                conn.close()

    ###
    async def check_rate_limit(self, request, api_key_data=None):
        """
        Comprehensive rate limit checking method
        
        Args:
            request (Request): The incoming request
            api_key_data (dict): Dictionary containing API key information
        
        Returns:
            dict: Rate limit information
        """
        try:
            # Validate input
            if not api_key_data or not isinstance(api_key_data, dict):
                raise ValueError("Invalid API key data")
            
            # Extract key details
            api_key = api_key_data.get('api_key')
            plan_type = api_key_data.get('plan_type')
            request_limit = api_key_data.get('request_limit')
            
            # Validate extracted details
            if not api_key or not plan_type or request_limit is None:
                raise ValueError("Incomplete API key information")
            
            # Enterprise plans have unlimited requests
            if plan_type == 'enterprise':
                return {
                    'limit': 'unlimited',
                    'remaining': 'unlimited',
                    'reset': 2592000  # 30 days
                }
            
            # Determine current month
            current_month = time.strftime("%Y-%m")
            
            # Choose environment-specific rate limit checking
            if self.environment == "production":
                return await self._check_redis_rate_limit(
                    api_key, plan_type, request_limit, current_month
                )
            else:
                return await self._check_sqlite_rate_limit(
                    api_key, plan_type, request_limit, current_month
                )
        
        except Exception as e:
            self.logger.error(f"Rate limit check error: {e}")
            raise HTTPException(
                status_code=500, 
                detail="Error checking rate limit"
            )

    async def _check_redis_rate_limit(self, api_key, plan_type, request_limit, current_month):
        """
        Check and update rate limits using Redis
        
        Args:
            api_key (str): The API key
            plan_type (str): The plan type
            request_limit (int): Maximum number of requests allowed
            current_month (str): Current month identifier
        
        Returns:
            dict: Rate limit information
        """
        try:
            # Create a unique key for tracking
            key = f"rate_limit:{api_key}:{current_month}"
            
            # Atomic increment of request count
            current_count = self.redis.incr(key)
            
            # Set expiration for the key if it's the first request
            if current_count == 1:
                next_month = time.strftime(
                    "%Y-%m-01", 
                    time.localtime(time.time() + 32*24*60*60)
                )
                expire_at = time.mktime(time.strptime(next_month, "%Y-%m-%d"))
                self.redis.expireat(key, int(expire_at))
            
            # Calculate remaining requests
            remaining = max(0, request_limit - current_count)
            
            # Check if rate limit is exceeded
            if current_count > request_limit:
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded for {plan_type} plan"
                )
            
            # Return rate limit information
            return {
                'limit': str(request_limit),
                'remaining': str(remaining),
                'reset': self.redis.ttl(key)
            }
        
        except Exception as e:
            self.logger.error(f"Redis rate limit error: {e}")
            # Fallback to SQLite if Redis fails
            return await self._check_sqlite_rate_limit(
                api_key, plan_type, request_limit, current_month
            )

    async def _check_sqlite_rate_limit(self, api_key, plan_type, request_limit, current_month):
        """
        Check and update rate limits using SQLite
        
        Args:
            api_key (str): The API key
            plan_type (str): The plan type
            request_limit (int): Maximum number of requests allowed
            current_month (str): Current month identifier
        
        Returns:
            dict: Rate limit information
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Atomically insert or update request count
            cursor.execute("""
                INSERT INTO rate_limits (api_key, month, request_count)
                VALUES (?, ?, 1)
                ON CONFLICT(api_key, month) DO UPDATE
                SET request_count = request_count + 1
                RETURNING request_count
            """, (api_key, current_month))
            
            current_count = cursor.fetchone()[0]
            conn.commit()
            
            # Calculate remaining requests
            remaining = max(0, request_limit - current_count)
            
            # Check if rate limit is exceeded
            if current_count > request_limit:
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded for {plan_type} plan"
                )
            
            # Return rate limit information
            return {
                'limit': str(request_limit),
                'remaining': str(remaining),
                'reset': 2592000  # 30 days
            }
        
        except Exception as e:
            self.logger.error(f"SQLite rate limit error: {e}")
            raise HTTPException(
                status_code=500,
                detail="Error processing rate limit"
            )
        finally:
            conn.close()

    def get_usage_stats(self, api_key_data):
        """
        Retrieve current usage statistics
        
        Args:
            api_key_data (dict or str): API key information
        
        Returns:
            dict: Current usage statistics
        """
        try:
            # Extract API key with multiple fallback methods
            if isinstance(api_key_data, dict):
                api_key = (
                    api_key_data.get('api_key') or 
                    api_key_data.get('key') or 
                    api_key_data.get('apiKey')
                )
            else:
                api_key = api_key_data
            
            # Validate API key
            if not api_key or not isinstance(api_key, str):
                raise ValueError("Invalid API key")
            
            current_month = time.strftime("%Y-%m")
            
            # Production environment (Redis)
            if self.environment == "production":
                try:
                    key = f"rate_limit:{api_key}:{current_month}"
                    count = int(self.redis.get(key) or 0)
                    ttl = self.redis.ttl(key)
                    return {
                        "current_usage": count,
                        "ttl_seconds": ttl
                    }
                except redis.RedisError as e:
                    self.logger.error(f"Redis usage stats error: {e}")
                    # Fallback to SQLite
                    self.environment = "development"
            
            # Development/Fallback environment (SQLite)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT request_count FROM rate_limits
                    WHERE api_key = ? AND month = ?
                """, (api_key, current_month))
                result = cursor.fetchone()
                return {
                    "current_usage": result[0] if result else 0,
                    "ttl_seconds": 2592000  # 30 days
                }
        
        except Exception as e:
            self.logger.error(f"Unexpected error in get_usage_stats: {e}")
            raise HTTPException(
                status_code=500, 
                detail="Error retrieving usage statistics"
            ) 
    async def deduct_request(self, api_key_data, request_successful: bool = True):
        """
        Deduct a request from the user's monthly allocation only if the request was successful
        
        Args:
            api_key_data (dict): Authenticated API key information
            request_successful (bool): Whether the request was processed successfully
        
        Returns:
            dict: Updated request usage information
        """
        if not request_successful:
            return None

        conn = None
        cursor = None
        
        try:
            # Extract key information
            api_key_id = api_key_data.get('id')
            plan_type = api_key_data.get('plan_type')
            request_limit = api_key_data.get('request_limit')
            
            # Validate input
            if not all([api_key_id, plan_type, request_limit is not None]):
                self.logger.warning("Incomplete API key data for request deduction")
                return None
            
            # Enterprise plan has unlimited requests
            if plan_type == 'enterprise' or request_limit == -1:
                return {
                    'status': 'unlimited',
                    'remaining': 'unlimited'
                }
            
            # Get database connection
            conn = connection_pool.get_connection()
            cursor = conn.cursor(dictionary=True)
            
            # Get current month
            current_month = time.strftime("%Y-%m")
            
            # Check current request count for this month
            cursor.execute("""
                SELECT COUNT(*) as request_count 
                FROM api_key_requests 
                WHERE api_key_id = %s 
                AND YEAR(request_time) = YEAR(CURRENT_DATE) 
                AND MONTH(request_time) = MONTH(CURRENT_DATE)
            """, (api_key_id,))
            
            current_count = cursor.fetchone()['request_count']
            
            # Check if request limit is exceeded
            if current_count >= request_limit:
                # Deactivate the API key if limit is reached
                cursor.execute("""
                    UPDATE api_keys 
                    SET is_active = FALSE 
                    WHERE id = %s
                """, (api_key_id,))
                
                conn.commit()
                self.logger.warning(f"API key {api_key_id} exceeded monthly limit")
                raise HTTPException(
                    status_code=429,
                    detail=f"Monthly request limit of {request_limit} exceeded"
                )
            
            # Log the current request
            cursor.execute("""
                INSERT INTO api_key_requests (
                    api_key_id, 
                    request_time, 
                    request_details
                ) VALUES (%s, CURRENT_TIMESTAMP, %s)
            """, (api_key_id, 'Successful API request'))
            
            # Update last_used_at in api_keys
            cursor.execute("""
                UPDATE api_keys 
                SET last_used_at = CURRENT_TIMESTAMP 
                WHERE id = %s
            """, (api_key_id,))
            
            # Calculate remaining requests
            remaining_requests = request_limit - (current_count + 1)
            
            conn.commit()
            
            return {
                'status': 'success',
                'current_usage': current_count + 1,
                'remaining': remaining_requests,
                'limit': request_limit
            }
        
        except Exception as e:
            if conn:
                conn.rollback()
            
            self.logger.error(f"Request deduction error: {e}")
            raise HTTPException(
                status_code=500,
                detail="Error processing request deduction"
            )
        
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
                    
    async def update_request_usage(self, api_key_data):
        """
        Update request usage for an API key in the database
        
        Args:
            api_key_data (dict): Authenticated API key data
        """
        try:
            try:
                conn = connection_pool.get_connection()
                cursor = conn.cursor(dictionary=True)
            except Exception as e:
                self.logger.error(f"Database connection error: {e}")
                raise HTTPException(
                    status_code=500,
                    detail="Database connection error"
                )
            
            try:
                # Extract key information
                api_key = api_key_data.get('api_key')
                plan_type = api_key_data.get('plan_type')
                request_limit = api_key_data.get('request_limit')
                api_key_id = api_key_data.get('id')
                
                # Validate input
                if not all([api_key, plan_type, request_limit, api_key_id]):
                    self.logger.warning("Incomplete API key data for usage update")
                    return False
                
                # For enterprise plans, no need to track limits
                if plan_type == 'enterprise' or request_limit == -1:
                    return True
                
                # Get current month
                current_month = time.strftime("%Y-%m")
                
                # Check current request count for this month
                cursor.execute("""
                    SELECT COUNT(*) as request_count 
                    FROM api_key_requests 
                    WHERE api_key_id = %s AND YEAR(request_time) = YEAR(CURRENT_DATE) 
                    AND MONTH(request_time) = MONTH(CURRENT_DATE)
                """, (api_key_id,))
                
                current_count = cursor.fetchone()['request_count']
                
                # Check if request limit is exceeded
                if current_count >= request_limit:
                    # Deactivate the API key if limit is reached
                    cursor.execute("""
                        UPDATE api_keys 
                        SET is_active = FALSE 
                        WHERE id = %s
                    """, (api_key_id,))
                    
                    self.logger.warning(f"API key {api_key} exceeded monthly limit")
                    raise HTTPException(
                        status_code=429,
                        detail=f"Monthly request limit of {request_limit} exceeded"
                    )
                
                # Log the current request
                cursor.execute("""
                    INSERT INTO api_key_requests (
                        api_key_id, 
                        request_time, 
                        request_details
                    ) VALUES (%s, CURRENT_TIMESTAMP, %s)
                """, (api_key_id, request.url.path))
                
                # Update last_used_at in api_keys
                cursor.execute("""
                    UPDATE api_keys 
                    SET last_used_at = CURRENT_TIMESTAMP 
                    WHERE id = %s
                """, (api_key_id,))
                
                conn.commit()
                return True
            
            except Exception as e:
                conn.rollback()
                self.logger.error(f"Error updating request usage: {e}")
                return False
        
        finally:
            if conn:
                cursor.close()
                conn.close()

 
    
rate_limiter = RateLimit()