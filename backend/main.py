from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

import os
import time
import logging
from dotenv import load_dotenv

load_dotenv()

# LOGGER
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# IMPORT DB
from database.db import db

# IMPORT ROUTES
from routes.api import router as api_router

# IMPORT AI ERROR
from services.ai_service import FaceRecognitionError

# VALIDATE DATABASE
try:
    db.command("ping")
    logger.info("MongoDB verified successfully.")
except Exception as e:
    logger.error(f"MongoDB verification failed: {e}")

# FASTAPI APP
app = FastAPI(
    title="Face Recognition API",
    description="API for biometric registration and real-time face recognition using DeepFace ArcFace embeddings.",
    version="1.0.0"
)

# CUSTOM ERROR HANDLER
@app.exception_handler(FaceRecognitionError)
async def face_recognition_exception_handler(
    request: Request,
    exc: FaceRecognitionError
):

    logger.warning(
        f"Face recognition error: {exc}"
    )

    return JSONResponse(
        status_code=400,
        content={
            "detail": str(exc)
        },
    )

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv(
        "CORS_ORIGINS",
        "*"
    ).split(","),

    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# REQUEST LOGGER
@app.middleware("http")
async def add_process_time_header(
    request: Request,
    call_next
):

    start_time = time.time()

    response = await call_next(request)

    process_time = time.time() - start_time

    logger.info(
        f"{request.method} "
        f"{request.url.path} - "
        f"{response.status_code} - "
        f"{process_time:.4f}s"
    )

    response.headers["X-Process-Time"] = str(
        process_time
    )

    return response

# UPLOADS
os.makedirs("uploads", exist_ok=True)

app.mount(
    "/uploads",
    StaticFiles(directory="uploads"),
    name="uploads"
)

# ROUTES
app.include_router(
    api_router,
    prefix="/api"
)

# ROOT
@app.get("/")
def read_root():

    return {
        "message":
        "Face Recognition API Running"
    }

# HEALTH
@app.get("/health")
def health_check():

    return {
        "status": "ok",
        "version": "1.0.0"
    }

# RUN SERVER
if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )