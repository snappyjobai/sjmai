import os
import csv
import logging
from typing import List, Dict, Optional

from .sjm import Freelancer, DataSourceConfig, TestDataGenerator, normalize_csv

from mysql.connector import pooling

logger = logging.getLogger(__name__)

class FreelancerDataSource:
    """Flexible data source for freelancer information"""
    def __init__(self, config: DataSourceConfig = None):
        self.config = config or DataSourceConfig()
        self.freelancers = []

    def load_data(self) -> List[Freelancer]:
        """Load freelancer data based on configuration"""
        if self.config.type == "database":
            return self._load_from_database()
        elif self.config.type == "csv":
            return self._load_from_csv()
        elif self.config.type == "test":
            return self._load_test_data()
        else:
            raise ValueError(f"Unsupported data source type: {self.config.type}")

    def _load_from_database(self) -> List[Freelancer]:
        """Load freelancers from MySQL database"""
        conn = connection_pool.get_connection()
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute("""
                SELECT * FROM freelancers 
                WHERE is_active = 1
            """)
            freelancers = []
            for row in cursor.fetchall():
                freelancers.append(Freelancer(
                    id=row['id'],
                    username=row['username'],
                    name=row['name'],
                    job_title=row['job_title'],
                    skills=row['skills'].split(',') if row['skills'] else [],
                    experience=row['experience'],
                    rating=float(row['rating']),
                    hourly_rate=float(row['hourly_rate']),
                    profile_url=row['profile_url'],
                    availability=bool(row['availability']),
                    total_sales=row['total_sales'],
                    description=row['description']
                ))
            return freelancers
        except Exception as e:
            logger.error(f"Database loading failed: {e}")
            # Fallback to test data if database loading fails
            return self._load_test_data()
        finally:
            cursor.close()
            conn.close()

    def _load_from_csv(self) -> List[Freelancer]:
        """Load freelancers from CSV file"""
        try:
            # Use default path if not provided
            path = self.config.path or "test_freelancers.csv"
            
            # Check if file exists
            if not os.path.exists(path):
                logger.warning(f"CSV file not found: {path}. Using test data.")
                return self._load_test_data()
            
            # Normalize CSV to Freelancer objects
            return normalize_csv(path)
        except Exception as e:
            logger.error(f"CSV loading failed: {e}")
            return self._load_test_data()

    def _load_test_data(self) -> List[Freelancer]:
        """Generate test data"""
        generator = TestDataGenerator()
        df = generator.generate_freelancers(100)
        return [
            Freelancer(
                id=row['id'],
                username=row['username'],
                name=row['name'],
                job_title=row['job_title'],
                skills=row['skills'].split(','),
                experience=row['experience'],
                rating=row['rating'],
                hourly_rate=row['hourly_rate'],
                profile_url=row['profile_url'],
                availability=row['availability'],
                total_sales=row['total_sales'],
                description=row['description']
            )
            for _, row in df.iterrows()
        ]
