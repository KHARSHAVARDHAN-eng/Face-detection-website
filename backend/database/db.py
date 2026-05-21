from sqlalchemy import create_engine

from sqlalchemy.orm import (
    declarative_base,
    sessionmaker
)

# SQLITE DATABASE
DATABASE_URL = (
    "sqlite:///./face_recognition.db"
)

# ENGINE
engine = create_engine(
    DATABASE_URL,
    connect_args={
        "check_same_thread": False
    }
)

# SESSION
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# BASE MODEL
Base = declarative_base()


# DATABASE SESSION DEPENDENCY
def get_db():

    db = SessionLocal()

    try:

        yield db

    finally:

        db.close() 