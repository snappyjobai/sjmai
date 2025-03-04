import logging
import os
import json
from typing import Dict, Any, List, Optional
import tempfile
import asyncio

from fastapi import APIRouter, HTTPException, Depends, Query, File, UploadFile, Path
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .auth import get_api_key  # Your existing auth module
from ..core.data_generator import TestDataGenerator

# Initialize router
router = APIRouter(
    prefix="/test",
    tags=["Testing"],
    dependencies=[Depends(get_api_key)]
)

# Configure logging
logger = logging.getLogger(__name__)

# Base class for responses
class TestResponse(BaseModel):
    status_code: int
    rate_limit: Optional[Dict[str, str]] = None
    data: Optional[Any] = None

# Initialize shared tester instance with correct API URL
API_URL = "http://localhost:9000"  # Change to your API server URL
API_KEY = "sjm_fr_07580289890cccf65be9dcdc3573e9e1bdee3dac01d83393000b75069c1c127b_5892a92a"  # Default API key

# Define request models for different test endpoints
class ConfigureSourceRequest(BaseModel):
    source_type: str = "test"
    path: Optional[str] = None

class MatchRequest(BaseModel):
    desc: Optional[str] = None
    skills: Optional[str] = None  # Comma-separated skills
    budget: Optional[str] = None  # Format: "min-max"
    complexity: Optional[str] = "medium"
    timeline: Optional[int] = 30

class InterviewRequest(BaseModel):
    freelancer_id: str = "f1"
    description: Optional[str] = None
    skills: Optional[str] = None  # Comma-separated skills
    job_title: Optional[str] = "Full Stack Developer"
    mode: Optional[str] = "ai_full"
    interactive: Optional[bool] = False
    session_id: Optional[str] = None
    provided_answers: Optional[List[str]] = None

class VerifySkillRequest(BaseModel):
    skill: str = "Python"

class RateLimitRequest(BaseModel):
    num_requests: int = 5

# Utility functions
async def make_api_request(endpoint: str, method: str = "GET", data: Dict = None, files: Dict = None, headers: Dict = None) -> Dict:
    """Make an HTTP request to the API server"""
    import httpx

    url = f"{API_URL}{endpoint}"
    logger.info(f"Sending {method} request to {url}")

    # Set default headers with API key
    request_headers = {"X-API-Key": API_KEY}
    if headers:
        request_headers.update(headers)

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.request(
                method,
                url,
                headers=request_headers,
                json=data,
                files=files
            )

            result = {
                "status_code": response.status_code,
                "rate_limit": {
                    "limit": response.headers.get("X-RateLimit-Limit"),
                    "remaining": response.headers.get("X-RateLimit-Remaining"),
                    "reset": response.headers.get("X-RateLimit-Reset")
                }
            }

            try:
                if response.content:
                    result["data"] = response.json()
            except json.JSONDecodeError:
                result["data"] = response.text

            if response.status_code >= 400:
                logger.error(f"Request failed with status code: {response.status_code}")
                logger.error(f"Error response: {result.get('data')}")
            else:
                logger.info(f"Request successful. Status code: {response.status_code}")

            return result
    except Exception as e:
        logger.error(f"API request failed: {str(e)}")
        return {"error": str(e), "status_code": 500}

# API Endpoints that match your APITester methods
@router.get("/health")
async def test_health():
    """Test health endpoint"""
    logger.info("Testing health endpoint...")
    try:
        result = await make_api_request("/health")
        logger.info("Health check completed successfully")
        return result
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/configure-data-source")
async def test_configure_data_source(config: ConfigureSourceRequest):
    """Configure data source dynamically"""
    logger.info(f"Configuring data source: type={config.source_type}, path={config.path}")

    config_data = {
        "type": config.source_type
    }
    if config.path:
        config_data["path"] = config.path

    return await make_api_request("/configure-data-source", method="POST", data=config_data)

@router.post("/parse")
async def test_parse_resume(file: UploadFile = File(...)):
    """Test resume parsing endpoint"""
    logger.info(f"Testing parse endpoint with file: {file.filename}")

    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1])
    temp_file_path = temp_file.name

    try:
        # Write uploaded content to temporary file
        content = await file.read()
        with open(temp_file_path, "wb") as f:
            f.write(content)

        # Make the API request with the file
        with open(temp_file_path, "rb") as f:
            files = {"file": (file.filename, f)}
            result = await make_api_request("/parse", method="POST", files=files)

        return result
    except Exception as e:
        logger.error(f"Resume parsing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

@router.post("/match")
async def test_match_freelancers(request: MatchRequest):
    """Test freelancer matching endpoint with flexible parameter handling"""
    logger.info("Testing match endpoint...")

    # Handle individual parameters
    desc = request.desc or "Building a modern web application with React and Node.js"
    skills = request.skills.split(',') if request.skills else ["React.js", "Node.js", "TypeScript"]

    # Parse budget range
    budget = request.budget or "5000-10000"
    if isinstance(budget, str) and '-' in budget:
        min_budget, max_budget = map(int, budget.split('-'))
    else:
        min_budget, max_budget = 5000, 10000

    project_data = {
        "description": desc,
        "required_skills": [s.strip() for s in skills],
        "budget_range": (min_budget, max_budget),
        "complexity": request.complexity,
        "timeline": request.timeline
    }

    # Configure test data first
    await test_configure_data_source(ConfigureSourceRequest(source_type="test"))

    # Log project requirements
    logger.info(f"Project Requirements:")
    logger.info(f"Description: {project_data['description']}")
    logger.info(f"Required Skills: {', '.join(project_data['required_skills'])}")
    logger.info(f"Budget Range: ${project_data['budget_range'][0]}-${project_data['budget_range'][1]}")
    logger.info(f"Complexity: {project_data['complexity']}")
    logger.info(f"Timeline: {project_data['timeline']} days")

    result = await make_api_request("/match", method="POST", data=project_data)

    # Process and display matches (for logging)
    if result.get("status_code") == 200 and "matches" in result.get("data", {}):
        matches = result["data"]["matches"]
        logger.info(f"Found {len(matches)} matching freelancers")

    return result

@router.post("/interview")
async def test_conduct_interview(request: InterviewRequest):
    """Test AI interview endpoint"""
    logger.info("Testing interview endpoint...")

    # Configure test data source first
    await test_configure_data_source(ConfigureSourceRequest(source_type="test"))

    # Format interview data
    skills = request.skills.split(',') if request.skills else ["React.js", "Node.js", "TypeScript"]

    interview_data = {
        "freelancer_id": request.freelancer_id,
        "project_description": request.description or "Build a modern web application with React and Node.js",
        "required_skills": [s.strip() for s in skills],
        "job_title": request.job_title,
        "mode": request.mode
    }

    # Add session_id and provided_answers if available
    if request.session_id:
        interview_data["session_id"] = request.session_id

    if request.provided_answers:
        interview_data["provided_answers"] = request.provided_answers

    # Step 1: Initial request
    logger.info(f"Initiating Interview:")
    logger.info(f"Project: {interview_data['project_description']}")
    logger.info(f"Required Skills: {', '.join(interview_data['required_skills'])}")
    logger.info(f"Job Title: {interview_data['job_title']}")

    # If it's interactive mode but we don't have answers yet, use ai_questions mode
    if request.interactive and not request.provided_answers:
        questions_request_data = {**interview_data, "mode": "ai_questions"}
        result = await make_api_request("/interview", method="POST", data=questions_request_data)

        # Add a note that this is interactive mode
        if result.get("status_code") == 200:
            result["interactive_mode"] = True
            # Extract session ID if available
            if "data" in result and isinstance(result["data"], dict):
                data = result["data"].get("data", {})
                session_id = data.get("session_id")
                if session_id:
                    result["session_id"] = session_id
    else:
        # Regular mode or interactive with answers
        result = await make_api_request("/interview", method="POST", data=interview_data)

    return result

@router.post("/verify-skill")
async def test_verify_skill(request: VerifySkillRequest):
    """Test skill verification endpoint"""
    logger.info(f"Testing skill verification for: {request.skill}")
    return await make_api_request("/verify-skill", method="POST", data={"keyword": request.skill})

@router.get("/generate-test-data")
async def test_generate_test_data(num_freelancers: int = Query(100, description="Number of freelancers to generate")):
    """Test test data generation endpoint"""
    logger.info(f"Testing test data generation with {num_freelancers} freelancers")
    return await make_api_request(f"/generate-test-data?num_freelancers={num_freelancers}")

@router.post("/rate-limit")
async def test_rate_limit(request: RateLimitRequest):
    """Test rate limiting by making multiple requests"""
    logger.info(f"Testing rate limiting with {request.num_requests} requests...")
    results = []
    for i in range(request.num_requests):
        result = await make_api_request("/health")
        results.append(result)
        remaining = result.get('rate_limit', {}).get('remaining', 'N/A')
        logger.info(f"Request {i+1}: Rate limit remaining = {remaining}")
    return {"results": results}

@router.post("/invalid-auth")
async def test_invalid_auth():
    """Test invalid authentication"""
    logger.info("Testing invalid authentication...")
    # Use an invalid API key
    invalid_key_headers = {"X-API-Key": "invalid_key"}
    return await make_api_request("/health", headers=invalid_key_headers)

@router.get("/comprehensive")
async def run_comprehensive_tests():
    """Run all test scenarios"""
    logger.info("Starting Comprehensive API Tests")
    results = {}

    try:
        # Basic tests
        results["health"] = await test_health()
        results["invalid_auth"] = await test_invalid_auth()
        results["config"] = await test_configure_data_source(ConfigureSourceRequest(source_type="test"))

        # Generate test data
        results["generate_test_data"] = await test_generate_test_data(10)  # Use a small number for quicker testing

        # Test skill verification
        results["verify_skill"] = await test_verify_skill(VerifySkillRequest(skill="React.js"))

        # Test matching with a simple scenario
        match_request = MatchRequest(
            desc="Frontend Development",
            skills="HTML,CSS,JavaScript",
            budget="1000-3000",
            complexity="low",
            timeline=15
        )
        results["match"] = await test_match_freelancers(match_request)

        # Test interview
        interview_request = InterviewRequest(
            freelancer_id="f1",
            description="Build a simple website",
            skills="HTML,CSS,JavaScript",
            job_title="Frontend Developer",
            mode="ai_questions"
        )
        results["interview"] = await test_conduct_interview(interview_request)

        # Test rate limiting with just 2 requests for speed
        results["rate_limit"] = await test_rate_limit(RateLimitRequest(num_requests=2))

        logger.info("All tests completed successfully")
        return {"results": results}

    except Exception as e:
        logger.error(f"Tests failed: {e}")
        return {"error": str(e)}

# Additional endpoints for easy testing with predefined scenarios
@router.get("/scenarios/frontend-project")
async def test_frontend_project_scenario():
    """Test a frontend development project scenario"""
    match_request = MatchRequest(
        desc="Frontend Development for E-commerce Website",
        skills="HTML,CSS,JavaScript,React.js",
        budget="3000-7000",
        complexity="medium",
        timeline=30
    )
    return await test_match_freelancers(match_request)

@router.get("/scenarios/backend-project")
async def test_backend_project_scenario():
    """Test a backend development project scenario"""
    match_request = MatchRequest(
        desc="Backend API Development for Mobile App",
        skills="Python,FastAPI,PostgreSQL,AWS",
        budget="8000-15000",
        complexity="high",
        timeline=60
    )
    return await test_match_freelancers(match_request)

@router.get("/scenarios/fullstack-project")
async def test_fullstack_project_scenario():
    """Test a fullstack development project scenario"""
    match_request = MatchRequest(
        desc="Full Stack Web Application Development",
        skills="React.js,Node.js,MongoDB,Express.js",
        budget="10000-20000",
        complexity="high",
        timeline=90
    )
    return await test_match_freelancers(match_request)

@router.get("/scenarios/interview-frontend-dev")
async def test_interview_frontend_dev_scenario():
    """Test an interview for a frontend developer scenario"""
    interview_request = InterviewRequest(
        freelancer_id="f1",
        description="Frontend development for a responsive e-commerce website",
        skills="HTML,CSS,JavaScript,React.js",
        job_title="Frontend Developer",
        mode="ai_questions"
    )
    return await test_conduct_interview(interview_request)

@router.get("/scenarios/interview-backend-dev")
async def test_interview_backend_dev_scenario():
    """Test an interview for a backend developer scenario"""
    interview_request = InterviewRequest(
        freelancer_id="f1",
        description="Develop an API for a mobile application",
        skills="Python,Django,PostgreSQL,AWS",
        job_title="Backend Developer",
        mode="ai_questions"
    )
    return await test_conduct_interview(interview_request)

@router.get("/scenarios/interview-fullstack-dev")
async def test_interview_fullstack_dev_scenario():
    """Test an interview for a fullstack developer scenario"""
    interview_request = InterviewRequest(
        freelancer_id="f1",
        description="Develop a complete web application for project management",
        skills="React.js,Node.js,MongoDB,Express.js",
        job_title="Full Stack Developer",
        mode="ai_questions"
    )
    return await test_conduct_interview(interview_request)