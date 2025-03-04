# SJM.AI - Snap Job Model

SJM.AI is a comprehensive API platform for resume parsing, skills matching, and AI-powered freelancer interviews. It helps you find the perfect talent match for your projects using advanced natural language processing and machine learning techniques.

## Features

- **Resume Parsing**: Extract skills, experience, and education from PDF and DOCX resumes
- **Skills Matching**: Match project requirements with freelancer skills using hybrid recommendation algorithms
- **AI Interviews**: Conduct automated interviews with potential freelancers using Claude or OpenAI models
- **Rate Limiting**: Sophisticated rate limiting based on API key plan types
- **Authentication**: Secure API key management with encryption

## Tech Stack

- **Backend**: FastAPI
- **AI Services**: Claude (Anthropic) and OpenAI
- **Data Processing**: NLTK, scikit-learn, pandas
- **Caching**: Redis (production) / SQLite (development)

## Installation

### Prerequisites

- Python 3.9+
- MySQL Server
- Node.js and PM2 (for production deployment)
- Nginx (for production deployment)
- Claude and OpenAI API keys

<!-- ### Local Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sjm_package.git
   cd sjm_package
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   # Create a .env file
   cat > .env << EOF
   DB_HOST=localhost
   DB_USER=your_db_user
   DB_PASSWORD=your_db_password
   DB_NAME=sjm_db
   CLAUDE_API_KEY=your_claude_api_key
   OPENAI_API_KEY=your_openai_api_key
   ENCRYPTION_KEY=your_encryption_key
   ENVIRONMENT=development
   EOF
   ```

5. Download NLTK data:
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   ```

6. Run the development server:
   ```bash
   python -m uvicorn sjm_package.api.app:app --reload --port 8000
   ```

### Production Deployment

1. Clone the repository on your server:
   ```bash
   git clone https://github.com/yourusername/sjm_package.git
   cd sjm_package
   ```

2. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. Create a production startup script:
   ```bash
   cat > start_server.py << EOF
   import uvicorn

   if __name__ == "__main__":
       uvicorn.run(
           "sjm_package.api.app:app",
           host="0.0.0.0",
           port=8000,
           workers=4,
           reload=False,
           access_log=True
       )
   EOF
   ```

4. Configure PM2:
   ```bash
   cat > ecosystem.config.js << EOF
   module.exports = {
     apps: [{
       name: "sjm-api",
       script: "${PWD}/venv/bin/python",
       args: "${PWD}/start_server.py",
       instances: 1,
       autorestart: true,
       watch: false,
       max_memory_restart: "1G",
       env: {
         NODE_ENV: "production",
         DB_HOST: "localhost",
         DB_USER: "your_db_user",
         DB_PASSWORD: "your_db_password",
         DB_NAME: "sjm_db",
         CLAUDE_API_KEY: "your_claude_api_key",
         OPENAI_API_KEY: "your_openai_api_key",
         ENCRYPTION_KEY: "your_encryption_key",
         ENVIRONMENT: "production"
       }
     }]
   };
   EOF
   ```

5. Start with PM2:
   ```bash
   pm2 start ecosystem.config.js
   pm2 save
   pm2 startup
   ```

6. Configure Nginx:
   ```bash
   sudo nano /etc/nginx/conf.d/sjm-api.conf
   ```

   Add:
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;

       location / {
           proxy_pass http://127.0.0.1:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
       }

       client_max_body_size 10M;
   }
   ```

7. Apply Nginx configuration:
   ```bash
   sudo nginx -t
   sudo systemctl reload nginx
   ```

8. (Optional) Set up SSL:
   ```bash
   sudo certbot --nginx -d your-domain.com
   ``` -->

## API Documentation

When the server is running, visit `http://localhost:8000/docs` or `https://your-domain.com/docs` to access the Swagger UI documentation.

### Key Endpoints

- **POST /parse**: Parse resume files
- **POST /match**: Match freelancers to project requirements
- **POST /interview**: Conduct AI interviews with freelancers
- **POST /verify-skill**: Verify if a skill exists in the database
- **GET /generate-test-data**: Generate test data for development
- **GET /health**: Check the health status of the API

## Authentication

All API endpoints require an API key passed via the `X-API-Key` header. API keys follow this format:

- `sjm_fr_*`: Freelancer plan
- `sjm_pr_*`: Professional plan
- `sjm_ent_*`: Enterprise plan

## Rate Limiting

Rate limits are enforced based on the API key plan type:

- Freelancer: Lower request limits
- Professional: Medium request limits
- Enterprise: Unlimited requests

## Database Schema

The system requires the following MySQL tables:

- `users`: User information
- `api_keys`: API key storage (encrypted)
- `api_key_requests`: Usage tracking
- `plans`: Plan definitions and limits
- `freelancers`: Freelancer profiles (optional if using external data source)

## Customization

### Data Sources

The system supports multiple data sources:

- **Database**: Connect to a MySQL database with freelancer profiles
- **CSV**: Load data from a CSV file
- **Test**: Generate test data on the fly

Configure your data source using the `/configure-data-source` endpoint.

## Contact

snappyjob.ai@gmail.com
