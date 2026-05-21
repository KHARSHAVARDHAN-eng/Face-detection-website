import os
from pymongo import MongoClient

# MONGODB CONNECTION CONFIGURATION
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("MONGODB_DB", "face_recognition")

client = MongoClient(MONGODB_URI)
db = client[DATABASE_NAME]

# DATABASE DEPENDENCY FOR FASTAPI ROUTES
def get_db():
    try:
        yield db
    finally:
        pass