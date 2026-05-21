from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Depends,
    HTTPException
)

from sqlalchemy.orm import Session

from typing import List, Optional

import os
import uuid
import base64
import json

import numpy as np

from pydantic import BaseModel

from database.db import get_db

from models.user import User

from models.schemas import (
    RegistrationResponse,
    RecognitionResponse,
    UserResponse,
    DeleteResponse
)

from services.ai_service import (
    extract_face_embedding,
    cosine_similarity,
    FaceRecognitionError
)

router = APIRouter()

UPLOAD_DIR = "uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)


# =========================================
# REQUEST MODEL
# =========================================
class RegisterRequest(BaseModel):

    name: str

    embeddings: List[List[float]]

    image_base64: Optional[str] = None


# =========================================
# PROCESS LIVE FRAME
# =========================================
@router.post("/process-frame")
async def process_frame(
    image: UploadFile = File(...)
):

    try:

        image_bytes = await image.read()

        embedding = extract_face_embedding(
            image_bytes
        )

        if (
            not embedding or
            len(embedding) != 512
        ):

            raise FaceRecognitionError(
                "No valid face detected"
            )

        return {
            "success": True,
            "embedding": embedding
        }

    except Exception as e:

        print("FRAME ERROR:", e)

        raise HTTPException(
            status_code=400,
            detail="No valid face detected"
        )


# =========================================
# REGISTER USER
# =========================================
@router.post(
    "/register",
    response_model=RegistrationResponse
)
async def register_user(
    req: RegisterRequest,
    db: Session = Depends(get_db)
):

    print("\n===== REGISTER USER =====")

    print(
        f"User: {req.name}"
    )

    print(
        f"Embeddings received: "
        f"{len(req.embeddings)}"
    )

    # VALIDATION
    if (
        not req.name or
        not req.name.strip()
    ):

        raise HTTPException(
            status_code=400,
            detail="Name required"
        )

    if (
        not req.embeddings or
        len(req.embeddings) == 0
    ):

        raise HTTPException(
            status_code=400,
            detail="No embeddings received"
        )

    # SAVE IMAGE
    saved_image_path = None

    if req.image_base64:

        try:

            if "," in req.image_base64:

                _, encoded = req.image_base64.split(
                    ",",
                    1
                )

            else:

                encoded = req.image_base64

            image_bytes = base64.b64decode(
                encoded
            )

            filename = (
                f"{uuid.uuid4().hex}.jpg"
            )

            saved_image_path = os.path.join(
                UPLOAD_DIR,
                filename
            )

            with open(
                saved_image_path,
                "wb"
            ) as f:

                f.write(image_bytes)

        except Exception as e:

            print(
                "Image save failed:",
                e
            )

    # NORMALIZE EMBEDDINGS
    normalized_embeddings = []

    for emb in req.embeddings:

        emb_np = np.array(
            emb,
            dtype=np.float32
        )

        norm = np.linalg.norm(emb_np)

        if norm > 0:

            emb_np = emb_np / norm

        normalized_embeddings.append(
            emb_np.tolist()
        )

    # MASTER EMBEDDING
    embeddings_arr = np.array(
        normalized_embeddings,
        dtype=np.float32
    )

    master_embedding = np.mean(
        embeddings_arr,
        axis=0
    )

    master_norm = np.linalg.norm(
        master_embedding
    )

    if master_norm > 0:

        master_embedding = (
            master_embedding / master_norm
        )

    # SERIALIZE
    serialized_embeddings = json.dumps(
        normalized_embeddings
    )

    serialized_master = json.dumps(
        master_embedding.tolist()
    )

    # SAVE USER
    try:

        user = User(
            name=req.name.strip(),
            embeddings=serialized_embeddings,
            master_embedding=serialized_master,
            image_path=saved_image_path
        )

        db.add(user)

        db.commit()

        db.refresh(user)

        print(
            "USER SAVED SUCCESSFULLY"
        )

        return {
            "success": True,
            "message":
            "Enrollment completed successfully"
        }

    except Exception as e:

        db.rollback()

        print(
            "DATABASE SAVE FAILED:",
            e
        )

        raise HTTPException(
            status_code=500,
            detail="Database save failed"
        )


# =========================================
# RECOGNITION
# =========================================
@router.post(
    "/recognize",
    response_model=RecognitionResponse
)
async def recognize_user(
    image: UploadFile = File(...),
    db: Session = Depends(get_db)
):

    print("\n===== RECOGNITION =====")

    # EXTRACT LIVE EMBEDDING
    try:

        image_bytes = await image.read()

        target_embedding = extract_face_embedding(
            image_bytes
        )

        if (
            not target_embedding or
            len(target_embedding) != 512
        ):

            return {
                "status": "invalid",
                "message":
                "Invalid Person. Please Register."
            }

    except Exception as e:

        print(
            "Embedding extraction failed:",
            e
        )

        return {
            "status": "invalid",
            "message":
            "Invalid Person. Please Register."
        }

    # LOAD USERS
    users = db.query(User).all()

    if len(users) == 0:

        return {
            "status": "invalid",
            "message":
            "No registered users found."
        }

    best_similarity = -1.0

    best_user = None

    print("\n===== MATCHING =====")

    # GLOBAL MATCH SEARCH
    for user in users:

        try:

            stored_embeddings = json.loads(
                user.embeddings
            )

        except Exception as e:

            print(
                f"JSON parse failed for "
                f"{user.name}: {e}"
            )

            continue

        for emb in stored_embeddings:

            try:

                if len(emb) != 512:
                    continue

                similarity = cosine_similarity(
                    target_embedding,
                    emb
                )

                print(
                    f"{user.name} -> "
                    f"{similarity:.4f}"
                )

                if similarity > best_similarity:

                    best_similarity = similarity

                    best_user = user

            except Exception as e:

                print(
                    "Similarity error:",
                    e
                )

    print(
        f"\nBEST USER: "
        f"{best_user.name if best_user else 'NONE'}"
    )

    print(
        f"BEST SCORE: "
        f"{best_similarity:.4f}"
    )

    THRESHOLD = float(os.getenv("RECOGNITION_THRESHOLD", "0.70"))

    # MATCH FOUND
    if (
        best_user and
        best_similarity >= THRESHOLD
    ):

        return {
            "status": "matched",
            "name": best_user.name,
            "confidence": round(float(best_similarity), 4),
            "threshold": THRESHOLD
        }

    # INVALID PERSON
    return {
        "status": "invalid",
        "message":
        "Invalid Person. Please Register.",
        "confidence": round(float(best_similarity), 4) if best_user else None,
        "threshold": THRESHOLD
    }


# =========================================
# GET USERS
# =========================================
@router.get(
    "/users",
    response_model=List[UserResponse]
)
def get_users(
    db: Session = Depends(get_db)
):

    users = db.query(User).all()

    return [

        {
            "id": u.id,
            "name": u.name,
            "created_at": u.created_at,
            "image_path": u.image_path
        }

        for u in users
    ]


# =========================================
# DELETE USER
# =========================================
@router.delete(
    "/user/{user_id}",
    response_model=DeleteResponse
)
def delete_user(
    user_id: int,
    db: Session = Depends(get_db)
):

    user = db.query(User).filter(
        User.id == user_id
    ).first()

    if not user:

        raise HTTPException(
            status_code=404,
            detail="User not found"
        )

    if (
        user.image_path and
        os.path.exists(user.image_path)
    ):

        os.remove(user.image_path)

    db.delete(user)

    db.commit()

    return {
        "message":
        "User deleted successfully"
    }