from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Depends,
    HTTPException
)

from typing import List, Optional

import os
import uuid
import base64
import json

import numpy as np

from pydantic import BaseModel

from database.db import get_db
from bson.objectid import ObjectId
from datetime import datetime

from models.schemas import (
    RegistrationResponse,
    RecognitionResponse,
    UserResponse,
    DeleteResponse,
    FaceRecognitionResult,
    MultiRecognitionResponse
)

from services.ai_service import (
    extract_face_embedding,
    extract_multiple_face_embeddings,
    cosine_similarity,
    FaceRecognitionError,
    estimate_image_quality_and_threshold
)
from services.anti_spoof_service import get_anti_spoof_service
import cv2

router = APIRouter()

UPLOAD_DIR = "uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)


# =========================================
# EMBEDDING CACHE
# =========================================
_users_cache = None

def get_cached_users(db, force_refresh=False):
    global _users_cache
    if _users_cache is None or force_refresh:
        print("\n[CACHE] Refreshing database embeddings cache...")
        users = list(db.users.find({}, {"_id": 1, "name": 1, "embeddings": 1}))
        _users_cache = []
        for u in users:
            embeddings_list = []
            for emb in u.get("embeddings", []):
                if len(emb) == 512:
                    embeddings_list.append(emb)
            if embeddings_list:
                embeddings_np = np.array(embeddings_list, dtype=np.float32)
            else:
                embeddings_np = np.empty((0, 512), dtype=np.float32)
            _users_cache.append({
                "id": str(u["_id"]),
                "name": u["name"],
                "embeddings": embeddings_np
            })
        print(f"[CACHE] Loaded {len(_users_cache)} users into cache.\n")
    return _users_cache



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
    db = Depends(get_db)
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

    # SAVE USER TO MONGODB
    try:
        user_doc = {
            "name": req.name.strip(),
            "embeddings": normalized_embeddings,
            "master_embedding": master_embedding.tolist(),
            "image_path": saved_image_path,
            "created_at": datetime.utcnow()
        }

        result = db.users.insert_one(user_doc)

        print(
            f"USER SAVED SUCCESSFULLY: {result.inserted_id}"
        )

        get_cached_users(db, force_refresh=True)

        return {
            "success": True,
            "message":
            "Enrollment completed successfully"
        }

    except Exception as e:

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
    response_model=MultiRecognitionResponse
)
async def recognize_user(
    image: UploadFile = File(...),
    db = Depends(get_db)
):

    print("\n===== RECOGNITION (MULTI-USER WITH ANTI-SPOOFING) =====")

    try:
        image_bytes = await image.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        faces, w_img, h_img = extract_multiple_face_embeddings(image_bytes)
    except Exception as e:
        print("Embedding extraction failed:", e)
        return {
            "success": False,
            "width": 0,
            "height": 0,
            "faces": []
        }

    if len(faces) == 0:
        return {
            "success": True,
            "width": w_img,
            "height": h_img,
            "faces": []
        }

    # Load users from cache
    users = get_cached_users(db)
    anti_spoof = get_anti_spoof_service()

    results = []

    for face in faces:
        # 1. Run Anti-Spoof / Liveness Detection first
        box = face["box"]  # Coordinates: [x1, y1, x2, y2]
        is_real_from_model, liveness_score = anti_spoof.check_liveness(img_bgr, box)

        # Apply high-confidence liveness threshold
        LIVENESS_CONFIDENCE_THRESHOLD = 0.85
        is_liveness_real = is_real_from_model and (liveness_score >= LIVENESS_CONFIDENCE_THRESHOLD)

        # 2. Extract and normalize target embedding for ArcFace identity recognition
        target_np = np.array(face["embedding"], dtype=np.float32)
        norm = np.linalg.norm(target_np)
        if norm > 0:
            target_np = target_np / norm

        best_similarity = -1.0
        best_name = None

        # Compare against cached users using optimized vector operations
        for u in users:
            if u["embeddings"].size > 0:
                try:
                    similarities = np.dot(u["embeddings"], target_np)
                    max_sim = float(np.max(similarities))
                    if max_sim > best_similarity:
                        best_similarity = max_sim
                        best_name = u["name"]
                except Exception as e:
                    print("Matrix similarity error:", e)

        # Strict minimum similarity threshold (75% match)
        threshold = 0.75

        # 3. Security decision logic (Strict Priority Order: 1. UNKNOWN USER, 2. SPOOF DETECTION, 3. LIVE VERIFIED)
        if best_similarity < threshold or not best_name:
            # Invalid/Unregistered User
            results.append({
                "name": "Invalid User",
                "confidence": round(float(best_similarity) * 100, 1) if best_name else 0.0,
                "box": box,
                "status": "unknown",
                "is_real": is_liveness_real,
                "liveness_score": round(liveness_score, 4),
                "spoof_detected": False,
                "authentication_status": "unregistered",
                "message": "Not Registered"
            })
            print(f"[RECOGNIZE] Invalid User (Similarity: {best_similarity*100:.1f}% below threshold {threshold*100:.1f}%). Auth Denied.")
        elif not is_liveness_real:
            # Spoof detected - identify closest matched person but deny authentication
            results.append({
                "name": best_name,
                "confidence": round(float(best_similarity) * 100, 1),
                "box": box,
                "status": "spoof",
                "is_real": False,
                "liveness_score": round(liveness_score, 4),
                "spoof_detected": True,
                "authentication_status": "denied",
                "message": "SPOOF / PROXY ATTEMPT DETECTED"
            })
            print(f"[RECOGNIZE] Spoof detected for matched user {best_name} (Similarity: {best_similarity*100:.1f}%). Liveness: {liveness_score:.4f} - Auth Denied.")
        else:
            # Live person verified
            results.append({
                "name": best_name,
                "confidence": round(float(best_similarity) * 100, 1),
                "box": box,
                "status": "matched",
                "is_real": True,
                "liveness_score": round(liveness_score, 4),
                "spoof_detected": False,
                "authentication_status": "verified",
                "message": "Live Person Verified"
            })
            print(f"[RECOGNIZE] Match: {best_name} ({best_similarity * 100:.1f}%) | Liveness: {liveness_score:.3f} - Auth Verified.")

    return {
        "success": True,
        "width": w_img,
        "height": h_img,
        "faces": results
    }



# =========================================
# GET USERS
# =========================================
@router.get(
    "/users",
    response_model=List[UserResponse]
)
def get_users(
    db = Depends(get_db)
):

    users = list(db.users.find({}))

    return [

        {
            "id": str(u["_id"]),
            "name": u["name"],
            "created_at": u.get("created_at", datetime.utcnow()),
            "image_path": u.get("image_path")
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
    user_id: str,
    db = Depends(get_db)
):

    try:
        obj_id = ObjectId(user_id)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid user ID format"
        )

    user = db.users.find_one({"_id": obj_id})

    if not user:

        raise HTTPException(
            status_code=404,
            detail="User not found"
        )

    if (
        user.get("image_path") and
        os.path.exists(user["image_path"])
    ):

        os.remove(user["image_path"])

    db.users.delete_one({"_id": obj_id})

    get_cached_users(db, force_refresh=True)

    return {
        "message":
        "User deleted successfully"
    }