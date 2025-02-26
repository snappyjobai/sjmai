import os
import re
import json
import traceback
import logging
import sys
import csv
import socket
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from difflib import SequenceMatcher

import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rake_nltk import Rake
import numpy as np

from pydantic import BaseModel

# Global logging configuration
logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.DEBUG,  # or INFO, depending on how verbose you want the logs
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("core_test/sjm.log"),
        logging.StreamHandler()
    ]
)
# Data classes
@dataclass
class Freelancer:
    id: str
    username: str
    name: str
    job_title: str
    skills: List[str]
    experience: int
    rating: float
    hourly_rate: float
    profile_url: str
    availability: bool
    total_sales: int
    description: str = ""

@dataclass
class Project:
    id: str
    description: str
    required_skills: List[str]
    budget_range: tuple
    complexity: str
    timeline: int

# Skills extraction class combining functionalities from both versions
class SkillsExtract:
    def __init__(self, claude_api_key: Optional[str] = None, openai_api_key: Optional[str] = None):
        self.claude_api_key = claude_api_key or os.getenv('CLAUDE_API_KEY')
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        stop_words = set(stopwords.words('english'))
        self.stop_words = stop_words - {'need', 'needed', 'want', 'looking', 'developer', 'designer', 'manager', 'expert', 'senior', 'junior', 'level'}
        self.tfidf_vectorizer = TfidfVectorizer(stop_words=self.stop_words)
        self.rake = Rake()
        self.manual_keywords = []
        self.job_titles = set()
        self.known_skills = set()
        logger.info("SkillsExtract initialized.")

    def load_keywords_from_database(self, freelancers: List[Freelancer]) -> None:
        for f in freelancers:
            self.known_skills.update(f.skills)
            self.job_titles.add(f.job_title)
        self.manual_keywords = list(self.known_skills | self.job_titles)
        logger.info("Loaded keywords from freelancers.")

    def clean_skill(self, skill: str) -> str:
        cleaned = re.sub(r'[\[\]"×\+\(\)]', '', skill.strip())
        tech_formats = {
            'adobe xd': 'Adobe XD',
            'blender': 'Blender',
            'figma': 'Figma',
            'color theory': 'Color Theory',
            'unreal engine': 'Unreal Engine',
            'react': 'React.js',
            'reactjs': 'React.js',
            'node': 'Node.js',
            'nodejs': 'Node.js',
            'vue': 'Vue.js',
            'vuejs': 'Vue.js',
            'typescript': 'TypeScript',
            'javascript': 'JavaScript',
            'nextjs': 'Next.js',
            'nuxtjs': 'Nuxt.js',
            'expressjs': 'Express.js',
        }
        cleaned_lower = cleaned.lower()
        if cleaned_lower in tech_formats:
            return tech_formats[cleaned_lower]
        return cleaned.capitalize()

    def extract_skills(self, text: str) -> List[str]:
        if not text:
            return []
        cleaned_text = re.sub(r'[\[\]"×\+]', '', text)
        words = word_tokenize(cleaned_text.lower())
        matched_skills = set()
        self.rake.extract_keywords_from_text(text)
        keywords = self.rake.get_ranked_phrases()
        for kw in keywords:
            kw_clean = self.clean_skill(kw)
            matched_skills.add(kw_clean)
        return sorted(list(matched_skills))

    def verify_keyword(self, keyword: str) -> Dict[str, Any]:
        if not keyword:
            return self._empty_verification_result()
        cleaned_keyword = self.clean_skill(keyword)
        found_skills = {skill for skill in self.known_skills if cleaned_keyword.lower() == skill.lower()}
        found_titles = {title for title in self.job_titles if cleaned_keyword.lower() == title.lower()}
        if found_skills or found_titles:
            return {
                'exists': True,
                'similar_terms': [],
                'type': 'skill' if found_skills else 'job_title',
                'matches': list(found_skills or found_titles),
                'skills': list(found_skills),
                'job_titles': list(found_titles)
            }
        similar_terms = self._find_database_similar_terms(cleaned_keyword)
        return {
            'exists': False,
            'similar_terms': similar_terms,
            'type': None,
            'matches': [],
            'skills': [],
            'job_titles': []
        }

    def _find_database_similar_terms(self, keyword: str) -> List[str]:
        similar = []
        keyword_lower = keyword.lower()
        all_terms = list(self.known_skills) + list(self.job_titles)
        for term in all_terms:
            similarity = SequenceMatcher(None, keyword_lower, term.lower()).ratio()
            if similarity > 0.6:
                similar.append(term)
        return sorted(similar)[:5]

    def _empty_verification_result(self) -> Dict[str, Any]:
        return {
            'exists': False,
            'similar_terms': [],
            'type': None,
            'matches': [],
            'skills': [],
            'job_titles': []
        }

# Collaborative model for hybrid matching
class CollaborativeModel:
    def __init__(self):
        self.interaction_matrix = None
        self.freelancer_data = []
        self.project_data = []

    def train(self, project_data: List[Dict], freelancer_data: List[Freelancer]):
        self.freelancer_data = freelancer_data
        self.project_data = project_data
        num_freelancers = len(freelancer_data)
        if num_freelancers == 0:
            self.interaction_matrix = np.zeros((num_freelancers, 2))
            return
        total_sales = np.array([f.total_sales for f in freelancer_data])
        ratings = np.array([f.rating for f in freelancer_data])
        if total_sales.max() - total_sales.min() != 0:
            total_sales_norm = (total_sales - total_sales.min()) / (total_sales.max() - total_sales.min())
        else:
            total_sales_norm = total_sales
        ratings_norm = ratings  # assuming rating is on a 0-5 scale already
        self.interaction_matrix = np.column_stack((total_sales_norm, ratings_norm))

    def predict(self, project_description: str, project_skills: List[str]) -> List[float]:
        if self.interaction_matrix is None or self.interaction_matrix.size == 0:
            return [0.0] * len(self.freelancer_data)
        scores = np.nanmean(self.interaction_matrix, axis=1)
        return np.nan_to_num(scores).tolist()

# Content-based model for profile matching
class ContentBasedModel:
    def __init__(self):
        self.freelancer_profiles = None
        self.vectorizer = None

    def train(self, freelancer_data: List[Freelancer]):
        """Train the model on freelancer data"""
        # Create corpus combining skills and other relevant features
        corpus = [
            f"{f.job_title} {' '.join(f.skills)} {f.description or ''}"
            for f in freelancer_data
        ]
        
        # Initialize and fit vectorizer
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.freelancer_profiles = self.vectorizer.fit_transform(corpus)
        logger.info(f"Trained content model with {len(freelancer_data)} profiles")

    def predict(self, project_description: str, project_skills: List[str]) -> List[float]:
        """Predict similarity scores for a project"""
        if self.vectorizer is None or self.freelancer_profiles is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Create project corpus in the same format as training data
        project_text = f"{' '.join(project_skills)} {project_description}"
        
        # Use the same vectorizer to transform project text
        project_vector = self.vectorizer.transform([project_text])
        
        # Calculate similarities
        similarities = cosine_similarity(project_vector, self.freelancer_profiles)
        return similarities.flatten().tolist()

# Matching engine combining collaborative and content-based methods
from .ai_interviewer import AIInterviewer
from .data_generator import TestDataGenerator

class DataSourceConfig(BaseModel):
    """Configuration for data source"""
    type: str = "csv"  # database, csv, or test
    path: Optional[str] = None  # Path for CSV or test data
    connection_params: Optional[Dict[str, str]] = None  # Custom DB connection params

class MatchingEngine:
    def __init__(self, freelancers: List[Freelancer], projects: List[Project], 
                 skill_extractor: SkillsExtract, collaborative_model: Optional[CollaborativeModel] = None,
                 claude_api_key: Optional[str] = None, openai_api_key: Optional[str] = None, data_source_config: Optional[DataSourceConfig] = None):
        from .data_source import FreelancerDataSource
        self.freelancers = freelancers
        self.projects = projects
        self.skill_extractor = skill_extractor
        self.collaborative_model = collaborative_model or CollaborativeModel()
        self.content_model = ContentBasedModel()
        self.weights = {
            'content': 0.4,
            'collaborative': 0.4,
            'experience': 0.1,
            'rating': 0.1
        }
        self.ai_interviewer = AIInterviewer(claude_api_key, openai_api_key)
        self.data_source = FreelancerDataSource(data_source_config)
        
        # Load test data if no freelancers provided
        if not freelancers:
            generator = TestDataGenerator()
            test_data_path = generator.save_test_data()
            self.freelancers = normalize_csv(test_data_path)
        else:
            self.freelancers = freelancers

    def train_models(self):
        self.skill_extractor.load_keywords_from_database(self.freelancers)
        self.content_model.train(self.freelancers)
        project_data = [vars(p) for p in self.projects]
        self.collaborative_model.train(project_data, self.freelancers)

    def _get_content_scores(self, project: Project) -> List[float]:
        """Get content-based similarity scores for a project"""
        try:
            # Get scores using the trained content model
            return self.content_model.predict(project.description, project.required_skills)
        except Exception as e:
            logger.error(f"Error getting content scores: {e}")
            # Return neutral scores if there's an error
            return [0.5] * len(self.freelancers)

    def match_freelancers(self, project: Project, weights: Optional[Dict[str, float]] = None, page: int = 1) -> List[Dict[str, Any]]:
        weights = weights or self.weights
        self.train_models()
        content_scores = self._get_content_scores(project)
        collaborative_scores = self.collaborative_model.predict(project.description, project.required_skills)
        matches = []
        for i, freelancer in enumerate(self.freelancers):
            experience_score = min(freelancer.experience / 10, 1)
            rating_score = freelancer.rating / 5 if freelancer.rating <= 5 else 1
            combined = (weights['content'] * content_scores[i] +
                        weights['collaborative'] * collaborative_scores[i] +
                        weights['experience'] * experience_score +
                        weights['rating'] * rating_score)
            matches.append({
                'freelancer': freelancer,
                'combined_score': combined
            })
        matches.sort(key=lambda x: x['combined_score'], reverse=True)
        return matches

    def refine_skill_matching(self, required_skills: List[str], freelancer_skills: List[str]) -> int:
        return len(set(required_skills) & set(freelancer_skills))

    def get_top_matches(self, project: Project, top_n: int = 5) -> List[Dict[str, Any]]:
        all_matches = self.match_freelancers(project)
        return all_matches[:top_n]

    async def conduct_interview(self, interview_config: Dict, session_id: str = None, 
                        provided_answers: List[str] = None) -> Dict[str, Any]:
        """Conduct AI-powered interview with a freelancer"""
        # Log the start of the interview process with detailed configuration
        logger.info("Starting Interview Process")
        logger.info("Interview Configuration:")
        logger.info(json.dumps(interview_config, indent=2))
        
        # Log session information if provided
        if session_id:
            logger.info(f"Using provided session ID: {session_id}")
        if provided_answers:
            logger.info(f"Received {len(provided_answers)} answers for evaluation")

        try:
            # Extract relevant data from interview_config
            freelancer_id = interview_config["freelancer_profile"]["id"]
            project_description = interview_config["project_details"]["description"]
            project_skills = interview_config["project_details"]["required_skills"]
            
            # Log freelancer identification process
            logger.info(f"Searching for Freelancer with ID: {freelancer_id}")
            
            # Find the freelancer by ID
            freelancer = next((f for f in self.freelancers if f.id == freelancer_id), None)
            if not freelancer:
                logger.error(f"Freelancer with ID {freelancer_id} not found")
                raise ValueError(f"Freelancer with ID {freelancer_id} not found")
            
            # Log found freelancer details
            logger.info("Freelancer Details:")
            logger.info(json.dumps({
                "id": freelancer.id,
                "name": freelancer.name,
                "job_title": freelancer.job_title,
                "skills": freelancer.skills,
                "experience": freelancer.experience
            }, indent=2))
            
            # Conduct the interview using AI Interviewer
            logger.info("Initiating AI Interview")
            try:
                # Pass session_id and provided_answers to the AIInterviewer
                interview_result = await self.ai_interviewer.conduct_interview(
                    interview_config,
                    session_id=session_id,
                    provided_answers=provided_answers
                )
                
                # Log the raw interview result
                logger.info("Raw Interview Result:")
                logger.info(json.dumps({
                    "keys": list(interview_result.keys()),
                    "success": interview_result.get('success', False)
                }, indent=2))

                # Detailed logging of interview result components
                logger.info("Interview Data Components:")
                interview_data = interview_result.get('interview_data', {})
                evaluation = interview_result.get('evaluation', {})
                logger.info(json.dumps({
                    "interview_data_keys": list(interview_data.keys()) if interview_data else [],
                    "evaluation_keys": list(evaluation.keys()) if evaluation else []
                }, indent=2))

                # Ensure interview_result contains the expected fields
                if 'interview_data' not in interview_result and 'questions' in interview_result:
                    logger.info("Restructuring interview result to include interview_data")
                    interview_result['interview_data'] = {'questions': interview_result['questions']}
                    del interview_result['questions']

                # Prepare the final interview result
                final_result = {
                    'freelancer_id': freelancer.id,
                    'project_description': project_description,
                    'interview_data': interview_result.get('interview_data', {}),
                    'evaluation': interview_result.get('evaluation', {}),
                    'success': interview_result.get('success', True),
                    'session_id': interview_result.get('session_id', session_id)  # Include session_id in the response
                }

                # Log the final result structure
                logger.info("Final Interview Result Structure:")
                logger.info(json.dumps({
                    "interview_data_keys": list(final_result['interview_data'].keys()),
                    "evaluation_keys": list(final_result['evaluation'].keys()) if final_result.get('evaluation') else [],
                    "success": final_result['success'],
                    "session_id_included": bool(final_result.get('session_id'))
                }, indent=2))

                return final_result
                
            except Exception as interview_error:
                logger.error(f"AI Interview Process Failed: {interview_error}")
                logger.error(f"Detailed Error: {traceback.format_exc()}")
                return {
                    'freelancer_id': freelancer.id,
                    'project_description': project_description,
                    'error': str(interview_error),
                    'success': False
                }
                
        except Exception as e:
            logger.error(f"Interview Preparation Failed: {e}")
            logger.error(f"Detailed Error: {traceback.format_exc()}")
            return {
                'freelancer_id': freelancer_id if 'freelancer_id' in locals() else 'Unknown',
                'project_description': project_description if 'project_description' in locals() else 'N/A',
                'error': str(e),
                'success': False
            }
    
# CSV normalization to create Freelancer objects
def normalize_csv(file_path: str, csv_columns: Optional[Dict[str, str]] = None) -> List[Freelancer]:
    csv_columns = csv_columns or {
        'id': 'id',
        'freelancername': 'freelancername',
        'name': 'name',
        'job_title': 'job_title',
        'skills': 'skills',
        'experience': 'experience',
        'rating': 'rating',
        'hourly_rate': 'hourly_rate',
        'profile_url': 'profile_url',
        'availability': 'availability',
        'total_sales': 'total_sales'
    }
    freelancers = []
    with open(file_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            freelancer = Freelancer(
                id=row.get(csv_columns['id'], ""),
                username=row.get(csv_columns.get('freelancername', 'name'), ""),
                name=row.get(csv_columns.get('name', 'name'), ""),
                job_title=row.get(csv_columns['job_title'], ""),
                skills=[s.strip() for s in row.get(csv_columns['skills'], "").split(',') if s.strip()],
                experience=int(row.get(csv_columns['experience'], "0")),
                rating=float(row.get(csv_columns['rating'], "0")),
                hourly_rate=float(row.get(csv_columns['hourly_rate'], "0")),
                profile_url=row.get(csv_columns['profile_url'], ""),
                availability=(row.get(csv_columns['availability'], "True") == "True"),
                total_sales=int(row.get(csv_columns['total_sales'], "0"))
            )
            freelancers.append(freelancer)
    return freelancers

# Simple server for communication use cases
class Server:
    def __init__(self, host='127.0.0.1', port=65432):
        self.host = host
        self.port = port
        self.socket = None
        self.conn = None

    def setup_server(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((self.host, self.port))
        self.socket.listen(1)
        logger.info("Server setup complete.")

    def start_server(self):
        self.setup_server()
        self.conn, addr = self.socket.accept()
        logger.info(f"Connected by {addr}")

    def send_message(self, message):
        if self.conn:
            self.conn.sendall(message.encode('utf-8'))

    def receive_message(self):
        if self.conn:
            data = self.conn.recv(4096)
            return data.decode('utf-8')
        return ""

    def close_connection(self):
        if self.conn:
            self.conn.close()
        if self.socket:
            self.socket.close()
        logger.info("Server closed.")
