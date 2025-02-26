import os
from dotenv import load_dotenv
import mysql.connector
from mysql.connector import pooling

load_dotenv()

db_config = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "sjm_db"),
    "pool_name": "mypool",
    "pool_size": 5
}

connection_pool = mysql.connector.pooling.MySQLConnectionPool(**db_config)