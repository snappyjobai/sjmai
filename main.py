import uvicorn
from sjm_package.api.app import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)