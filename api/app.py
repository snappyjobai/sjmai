from fastapi import FastAPI, Depends, Request, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Union
import logging
import PyPDF2
import docx
import io
import json
from datetime import datetime, timedelta
from enum import Enum

from .auth import get_api_key
from .rate_limit import rate_limiter
from ..core.sjm import (
    SkillsExtract,
    Project,
    Freelancer,
    MatchingEngine,
    normalize_csv,
    DataSourceConfig
)

from ..core.data_source import FreelancerDataSource
from ..core.ai_interviewer import AIInterviewer
from ..core.data_generator import TestDataGenerator
from ..db.db import connection_pool

from .test_api import router as test_router

# Configure logging
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SJM.AI API",
    description="API for resume parsing, skills matching, and AI interviews",
    version="1.0.0",
    root_path="/api/v1"
)

# Initialize core components
skill_extractor = SkillsExtract()
test_data_generator = TestDataGenerator()


# Pydantic models
class ProjectRequest(BaseModel):
    description: str
    required_skills: List[str]
    budget_range: tuple
    complexity: str
    timeline: int

# In app.py

class InterviewMode(str, Enum):
    AI_FULL = "ai_full"  # AI generates questions and scores
    AI_QUESTIONS = "ai_questions"  # AI generates questions, client scores
    CUSTOM_FULL = "custom_full"  # Client provides questions and scoring
    HYBRID = "hybrid"  # Mix of AI and custom questions

class InterviewQuestion(BaseModel):
    text: str
    expected_answer: Optional[str] = None
    scoring_criteria: Optional[Dict[str, int]] = None
    follow_up: Optional[List[str]] = None

class ScoringCriteria(BaseModel):
    technical_accuracy: int = 40
    experience_level: int = 30
    communication: int = 30

class InterviewRequest(BaseModel):
    freelancer_id: str
    project_description: str
    required_skills: List[str]
    job_title: str
    mode: InterviewMode
    custom_questions: Optional[List[InterviewQuestion]] = None
    scoring_criteria: Optional[ScoringCriteria] = None
    additional_questions: Optional[List[str]] = None
    session_id: Optional[str] = None
    provided_answers: Optional[List[str]] = None


class SkillVerificationRequest(BaseModel):
    keyword: str

# flexible data source
def init_matching_engine(data_source_config: Optional[DataSourceConfig] = None) -> MatchingEngine:
    # Create data source
    data_source = FreelancerDataSource(data_source_config)

    # Load freelancers
    freelancers = data_source.load_data()

    # Initialize matching engine with loaded freelancers
    return MatchingEngine(
        freelancers=freelancers,
        projects=[],  # Projects will be added per request
        skill_extractor=skill_extractor,
        data_source_config=data_source_config
    )

# configure data source dynamically
@app.post("/configure-data-source")
async def configure_data_source(
    config: DataSourceConfig,
    api_key_data: dict = Depends(get_api_key)
):
    """Dynamically configure the data source for the matching engine"""
    try:
        # Reinitialize matching engine with new configuration
        global matching_engine
        matching_engine = init_matching_engine(config)

        return {
            "status": "success",
            "message": f"Data source configured to {config.type}",
            "freelancer_count": len(matching_engine.freelancers)
        }
    except Exception as e:
        logger.error(f"Data source configuration failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to configure data source: {str(e)}")


# Existing helper functions
def extract_pdf_text(content: bytes) -> str:
    try:
        pdf_file = io.BytesIO(content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        logger.error(f"PDF extraction failed: {str(e)}")
        raise HTTPException(status_code=400, detail="Failed to extract text from PDF")

def extract_doc_text(content: bytes) -> str:
    try:
        doc = docx.Document(io.BytesIO(content))
        return " ".join([paragraph.text for paragraph in doc.paragraphs])
    except Exception as e:
        logger.error(f"DOCX extraction failed: {str(e)}")
        raise HTTPException(status_code=400, detail="Failed to extract text from DOCX")

# Rate limiting middleware
@app.middleware("http")
async def add_rate_limit(request: Request, call_next):
    # Exclude documentation routes
    if request.url.path in ["/docs", "/openapi.json"]:
        return await call_next(request)

    try:
        # Get the API key from the request headers
        api_key = request.headers.get("X-API-Key")

        # Authenticate the API key first
        api_key_data = await get_api_key(api_key)

        try:
            # Prepare response
            response = await call_next(request)

            # Deduct request only if response is successful (status code 2xx)
            request_successful = 200 <= response.status_code < 300

            # Deduct request and get usage stats
            request_usage = await rate_limiter.deduct_request(
                api_key_data,
                request_successful
            )

            # Add rate limit headers if request was successful
            if request_usage:
                # Handle unlimited case
                if isinstance(request_usage.get('limit'), str) and request_usage['limit'] == 'unlimited':
                    response.headers["X-RateLimit-Limit"] = 'unlimited'
                    response.headers["X-RateLimit-Remaining"] = 'unlimited'
                    response.headers["X-RateLimit-Reset"] = str(2592000)  # 30 days in seconds
                else:
                    response.headers["X-RateLimit-Limit"] = str(request_usage.get('limit', 100))
                    response.headers["X-RateLimit-Remaining"] = str(request_usage.get('remaining', 0))

                    # Calculate time until next month reset
                    now = datetime.now()
                    next_month = now.replace(day=1) + timedelta(days=32)  # Go to next month
                    next_month = next_month.replace(day=1)  # Reset to first day of next month
                    time_until_reset = next_month - now

                    response.headers["X-RateLimit-Reset"] = str(int(time_until_reset.total_seconds()))

            return response

        except HTTPException as rate_limit_error:
            # Handle rate limit exceeded
            return JSONResponse(
                status_code=rate_limit_error.status_code,
                content={
                    "error": "Rate limit exceeded",
                    "detail": rate_limit_error.detail
                }
            )

    except HTTPException as auth_error:
        # Handle authentication errors
        return JSONResponse(
            status_code=auth_error.status_code,
            content={
                "error": "Authentication failed",
                "detail": auth_error.detail
            }
        )
# API Endpoints

# Test
app.include_router(test_router)


# Full
@app.post("/parse")
async def parse_resume(
    file: UploadFile = File(...),
    api_key_data: dict = Depends(get_api_key)
):
    try:
        content = await file.read()

        if len(content) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(status_code=400, detail="File too large (max 10MB)")

        filename = file.filename.lower()
        if filename.endswith('.pdf'):
            text = extract_pdf_text(content)
        elif filename.endswith(('.doc', '.docx')):
            text = extract_doc_text(content)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Please upload PDF or DOCX")

        parsed_data = {
            "skills": skill_extractor.extract_skills(text),
            "experience": [],  # Implement experience extraction
            "education": [],   # Implement education extraction
            "contact": {
                "email": None,
                "phone": None,
                "location": None
            }
        }

        return {"status": "success", "data": parsed_data}

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Resume parsing failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to parse resume")

@app.post("/match")
async def match_freelancers(
    project: ProjectRequest,
    api_key_data: dict = Depends(get_api_key)
):
    try:
        # Ensure matching engine is up to date
        matching_engine.train_models()

        project_obj = Project(
            id="temp_id",
            description=project.description,
            required_skills=project.required_skills,
            budget_range=project.budget_range,
            complexity=project.complexity,
            timeline=project.timeline
        )

        matches = matching_engine.get_top_matches(project_obj)

        return {
            "status": "success",
            "matches": [
                {
                    "freelancer": {
                        "id": match["freelancer"].id,
                        "name": match["freelancer"].name,
                        "job_title": match["freelancer"].job_title,
                        "skills": match["freelancer"].skills,
                        "experience": match["freelancer"].experience,
                        "rating": match["freelancer"].rating,
                        "hourly_rate": match["freelancer"].hourly_rate,
                        "availability": match["freelancer"].availability,
                        "total_sales": match["freelancer"].total_sales
                    },
                    "score": match["combined_score"],
                    "matching_skills": matching_engine.refine_skill_matching(
                        project.required_skills,
                        match["freelancer"].skills
                    )
                }
                for match in matches
            ]
        }
    except Exception as e:
        logger.error(f"Matching failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to match freelancers")

# Update the interview endpoint to handle the evaluation better
@app.post("/interview")
async def conduct_interview(
    interview_req: InterviewRequest,
    api_key_data: dict = Depends(get_api_key)
):
    """Conduct an interview with flexible modes"""
    try:
        # Log detailed request information for debugging
        logger.info("Interview Request Details:")
        logger.info(f"Freelancer ID: {interview_req.freelancer_id}")
        logger.info(f"Project Description: {interview_req.project_description}")
        logger.info(f"Required Skills: {interview_req.required_skills}")
        logger.info(f"Job Title: {interview_req.job_title}")
        logger.info(f"Interview Mode: {interview_req.mode}")

        # Get freelancer from test data or current data source
        freelancer = None
        for f in matching_engine.freelancers:
            if f.id == interview_req.freelancer_id:
                freelancer = f
                break

        if not freelancer:
            raise HTTPException(status_code=404, detail="Freelancer not found")

        # Log freelancer details
        logger.info("Freelancer Details:")
        logger.info(f"Name: {freelancer.name}")
        logger.info(f"Job Title: {freelancer.job_title}")
        logger.info(f"Skills: {freelancer.skills}")
        logger.info(f"Experience: {freelancer.experience} years")

        # Extract session ID if provided
        session_id = None
        provided_answers = None

        # Check for session_id in the request data
        if hasattr(interview_req, 'session_id') and interview_req.session_id:
            session_id = interview_req.session_id
            logger.info(f"Using provided session ID: {session_id}")

        # Check for answers in the request data
        if hasattr(interview_req, 'provided_answers') and interview_req.provided_answers:
            provided_answers = interview_req.provided_answers
            logger.info(f"Received {len(provided_answers)} answers for evaluation")

        # Prepare interview configuration
        interview_config = {
            "mode": interview_req.mode,
            "freelancer_profile": {
                "id": freelancer.id,
                "name": freelancer.name,
                "job_title": freelancer.job_title,
                "skills": freelancer.skills,
                "experience": freelancer.experience,
                "rating": freelancer.rating
            },
            "project_details": {
                "description": interview_req.project_description,
                "required_skills": interview_req.required_skills,
                "job_title": interview_req.job_title
            }
        }

        # Log full configuration for debugging
        logger.info("Full Interview Configuration:")
        logger.info(json.dumps(interview_config, indent=2))

        # Add mode-specific configuration
        if interview_req.mode == InterviewMode.CUSTOM_FULL:
            if not interview_req.custom_questions:
                raise HTTPException(
                    status_code=400,
                    detail="Custom questions required for CUSTOM_FULL mode"
                )
            interview_config["questions"] = [q.dict() for q in interview_req.custom_questions]
            interview_config["scoring_criteria"] = interview_req.scoring_criteria.dict() if interview_req.scoring_criteria else None

        elif interview_req.mode == InterviewMode.HYBRID:
            if interview_req.additional_questions:
                interview_config["additional_questions"] = interview_req.additional_questions

        # Conduct interview using AI Interviewer
        result = await matching_engine.conduct_interview(
            interview_config,
            session_id=session_id,
            provided_answers=provided_answers
        )

        # Log detailed result info
        logger.info("Interview Conduct Result:")
        logger.info(f"Success: {result.get('success', False)}")

        interview_data = result.get("interview_data", {})
        evaluation = result.get("evaluation", {})

        logger.info(f"Interview Data Keys: {interview_data.keys() if interview_data else 'None'}")
        logger.info(f"Evaluation Keys: {evaluation.keys() if evaluation else 'None'}")

        # Format detailed info for logging
        detailed_data = {
            "interview_data": {
                "questions_count": len(interview_data.get("questions", [])),
                "answers_count": len(interview_data.get("answers", []))
            },
            "evaluation": {
                "overall_score": evaluation.get("overall_score") if evaluation else None,
                "hiring_recommendation": evaluation.get("hiring_recommendation") if evaluation else None
            }
        }

        logger.info("Detailed Interview Data:")
        logger.info(json.dumps(detailed_data, indent=2))

        if not result["success"]:
            raise HTTPException(
                status_code=500,
                detail=result.get("error", "Interview failed")
            )

        # Structure the response properly
        response_data = {
            "status": "success",
            "data": {
                "freelancer": {
                    "id": freelancer.id,
                    "name": freelancer.name,
                    "job_title": freelancer.job_title
                },
                "mode": interview_req.mode,
                "session_id": result.get("session_id", session_id)
            }
        }

        # Include interview data if available
        if interview_data:
            response_data["data"]["interview_data"] = interview_data

        # Include evaluation if available - make sure it's a valid dictionary
        if evaluation and isinstance(evaluation, dict):
            # Ensure all fields in evaluation are JSON serializable
            cleaned_evaluation = {}

            for key, value in evaluation.items():
                # Handle potential non-serializable values
                if isinstance(value, (str, int, float, bool, type(None))):
                    cleaned_evaluation[key] = value
                elif isinstance(value, list):
                    # Ensure all list items are serializable
                    cleaned_evaluation[key] = [
                        item if isinstance(item, (str, int, float, bool, type(None)))
                        else str(item) for item in value
                    ]
                else:
                    # Convert other types to string
                    cleaned_evaluation[key] = str(value)

            response_data["data"]["evaluation"] = cleaned_evaluation

        return response_data

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Interview failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to conduct interview")

@app.post("/verify-skill")
async def verify_skill(
    skill_req: SkillVerificationRequest,
    api_key_data: dict = Depends(get_api_key)
):
    try:
        result = skill_extractor.verify_keyword(skill_req.keyword)
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"Skill verification failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to verify skill")

@app.get("/generate-test-data")
async def generate_test_data(
    num_freelancers: int = 100,
    api_key_data: dict = Depends(get_api_key)
):
    try:
        df = test_data_generator.generate_freelancers(num_freelancers)
        return {
            "status": "success",
            "data": df.to_dict(orient='records')
        }
    except Exception as e:
        logger.error(f"Test data generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate test data")

@app.get("/health")
async def health_check():
    """Check the health status of all system components"""
    try:
        health_response = {
            "status": "healthy",
            "components": {}
        }

        # Check matching engine
        try:
            # Get current data source configuration
            current_config = matching_engine.data_source.config if hasattr(matching_engine, 'data_source') else None

            if current_config:
                health_response["components"]["data_source"] = {
                    "status": "healthy",
                    "type": current_config.type,
                    "path": current_config.path or "N/A",
                }

                # Test data loading
                try:
                    freelancers = matching_engine.data_source.load_data()
                    health_response["components"]["data_source"]["freelancer_count"] = len(freelancers)
                except Exception as e:
                    health_response["components"]["data_source"]["status"] = "error"
                    health_response["components"]["data_source"]["error"] = str(e)
            else:
                health_response["components"]["data_source"] = {
                    "status": "not_initialized",
                }

        except Exception as e:
            health_response["components"]["data_source"] = {
                "status": "error",
                "error": str(e)
            }

        # Check skill extractor
        try:
            if skill_extractor:
                health_response["components"]["skill_extractor"] = {
                    "status": "ready"
                }
        except Exception as e:
            health_response["components"]["skill_extractor"] = {
                "status": "error",
                "error": str(e)
            }

        # Check database connection if applicable
        try:
            if current_config and current_config.type == "database":
                conn = connection_pool.get_connection()
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchone()
                cursor.close()
                conn.close()
                health_response["components"]["database"] = {
                    "status": "connected"
                }
        except Exception as db_error:
            health_response["components"]["database"] = {
                "status": "error",
                "error": str(db_error)
            }

        # Check if any component is unhealthy
        unhealthy_components = [
            component for component, status in health_response["components"].items()
            if status.get("status") in ["error", "not_initialized"]
        ]

        if unhealthy_components:
            health_response["status"] = "unhealthy"
            health_response["unhealthy_components"] = unhealthy_components

        return health_response

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "components": {}
        }

# Initialize matching engine and generate test data
def initialize_test_data():
    """Ensure test data is generated on application startup"""
    try:
        # Generate test data
        test_file_path = test_data_generator.save_test_data()
        logger.info(f"Test data generated at: {test_file_path}")

        # Initialize matching engine with the generated test data
        global matching_engine
        matching_engine = init_matching_engine(
            DataSourceConfig(type="csv", path=test_file_path)
        )
        logger.info(f"Matching engine initialized with {len(matching_engine.freelancers)} freelancers")

        global ai_interviewer
        ai_interviewer = AIInterviewer()
    except Exception as e:
        logger.error(f"Failed to initialize test data: {e}")
        raise

# Call the initialization function
initialize_test_data()