import pandas as pd
import random
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class TestDataGenerator:
    def __init__(self):
        self.skills_by_category = {
            'Web Development': [
                'React.js', 'Node.js', 'TypeScript', 'Next.js', 'Vue.js',
                'Angular', 'Django', 'Flask', 'PHP', 'Laravel'
            ],
            'Mobile Development': [
                'React Native', 'Flutter', 'Swift', 'Kotlin', 'iOS',
                'Android', 'Mobile UI/UX', 'App Store Optimization'
            ],
            'Data Science': [
                'Python', 'R', 'Machine Learning', 'Data Analysis',
                'SQL', 'Tableau', 'Power BI', 'Statistics'
            ],
            'Design': [
                'UI Design', 'UX Design', 'Figma', 'Adobe XD',
                'Photoshop', 'Illustrator', 'After Effects'
            ]
        }
        
        self.experience_levels = {
            'Junior': {'years': (1, 3), 'rate': (20, 50)},
            'Mid': {'years': (3, 6), 'rate': (50, 100)},
            'Senior': {'years': (6, 15), 'rate': (100, 200)}
        }

    def generate_freelancers(self, num_freelancers: int = 500) -> pd.DataFrame:
        data = []
        for i in range(num_freelancers):
            category = random.choice(list(self.skills_by_category.keys()))
            level = random.choice(list(self.experience_levels.keys()))
            
            num_skills = random.randint(3, 7)
            skills = random.sample(self.skills_by_category[category], min(num_skills, len(self.skills_by_category[category])))
            
            exp_range = self.experience_levels[level]['years']
            rate_range = self.experience_levels[level]['rate']
            
            freelancer = {
                'id': f"f{i+1}",
                'username': f"freelancer_{i+1}",
                'name': f"{level} {category.split()[0]} Expert",
                'job_title': f"{level} {category} Specialist",
                'skills': ','.join(skills),
                'experience': random.randint(exp_range[0], exp_range[1]),
                'rating': round(random.uniform(4.0, 5.0), 1),
                'hourly_rate': round(random.uniform(rate_range[0], rate_range[1]), 2),
                'profile_url': f"https://sjm.ai/profiles/f{i+1}",
                'availability': random.choice([True, False]),
                'total_sales': random.randint(10, 1000),
                'description': f"Experienced {category} professional specializing in {', '.join(skills[:3])}."
            }
            data.append(freelancer)
        
        return pd.DataFrame(data)

    def save_test_data(self, output_path: str = "test_freelancers.csv"):
        df = self.generate_freelancers()
        df.to_csv(output_path, index=False)
        logger.info(f"Generated {len(df)} test freelancers and saved to {output_path}")
        return output_path
