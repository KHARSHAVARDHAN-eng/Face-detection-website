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
from services.cdcn_service import get_cdcn_service
from services.liveness_features_service import get_liveness_features_service
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
    cdcn_service = get_cdcn_service()
    liveness_features_service = get_liveness_features_service()

    results = []

    for face in faces:
        box = face["box"]  # Coordinates: [x1, y1, x2, y2]
        
        # 1. ALWAYS compute ArcFace embedding/matching first
        target_np = np.array(face["embedding"], dtype=np.float32)
        norm = np.linalg.norm(target_np)
        if norm > 0:
            target_np = target_np / norm

        best_similarity = -1.0
        best_name = None
        top_k_sim = -1.0

        # Compare against cached users using optimized vector operations
        for u in users:
            if u["embeddings"].size > 0:
                try:
                    similarities = np.dot(u["embeddings"], target_np)
                    max_sim = float(np.max(similarities))
                    
                    # Compute top-3 nearest embedding average
                    sorted_sims = np.sort(similarities)[::-1]
                    k_val = min(3, len(sorted_sims))
                    avg_top_k = float(np.mean(sorted_sims[:k_val]))
                    
                    if max_sim > best_similarity:
                        best_similarity = max_sim
                        best_name = u["name"]
                        top_k_sim = avg_top_k
                except Exception as e:
                    print("Matrix similarity error:", e)

        # 2. Run CDCN Anti-Spoofing / Liveness Detection (returns label, real_score, spoof_score)
        cdcn_label, real_score, spoof_score = cdcn_service.predict_liveness(img_bgr, box)
        
        # Run MediaPipe eye blinking, head pose tracking, and static check
        tracking_info = liveness_features_service.track_and_update(img_bgr, box)
        
        blink_detected = tracking_info["blink_detected"]
        head_movements = tracking_info["head_movements"]
        is_static = tracking_info["is_static"]
        track_id = tracking_info["track_id"]
        
        # Combine model prediction and micro-movement check
        spoof_detected = (cdcn_label == "spoof") or is_static
        
        # Reject faces with no blink activity after a long duration (e.g. >15 frames)
        if track_id is not None:
            session = liveness_features_service.sessions.get(track_id)
            if session and isinstance(session, dict) and session.get("frame_count", 0) > 15 and not session.get("blink_detected", False):
                print(f"[RECOGNIZE] Track #{track_id} rejected: no blink activity over {session.get('frame_count')} frames.")
                spoof_detected = True

        # Lowered ArcFace false negatives via tuned similarity thresholds
        KNOWN_USER_THRESHOLD = 0.65
        SPOOF_MATCH_THRESHOLD = 0.48

        # Determine if identity matches based on liveness state
        effective_threshold = SPOOF_MATCH_THRESHOLD if spoof_detected else KNOWN_USER_THRESHOLD
        is_identified = (best_name is not None) and (best_similarity >= effective_threshold)

        if is_identified:
            identity = best_name
            identity_verified = True
        else:
            identity = "Invalid User"
            identity_verified = False

        # Debug Logs: For every detected face print: identity, similarity, spoof score, face bbox, embedding confidence
        print(f"[DEBUG] ==========================================")
        print(f"[DEBUG] Face BBox: {box}")
        print(f"[DEBUG] Identity Prediction: {identity} (Verified: {identity_verified})")
        print(f"[DEBUG] Cosine Similarity: {best_similarity:.4f} (Threshold: {effective_threshold:.4f})")
        print(f"[DEBUG] Top-k Similarity: {top_k_sim:.4f}")
        print(f"[DEBUG] Spoof Score: {spoof_score:.4f} (Real Score: {real_score:.4f}, Label: {cdcn_label})")
        print(f"[DEBUG] Embedding Confidence: {float(best_similarity) * 100:.1f}%")
        print(f"[DEBUG] ==========================================")

        if spoof_detected:
            # Denied
            status = "spoof"
            is_real = False
            authentication_status = "denied"
            message = f"{identity} - Proxy Attempt Detected" if identity_verified else "Spoof / Proxy Attempt Detected"
        else:
            # Live checked
            if identity_verified:
                status = "matched"
                is_real = True
                authentication_status = "verified"
                message = "Live Person Verified"
            else:
                status = "unknown"
                is_real = True
                authentication_status = "unregistered"
                message = "Not Registered"

        results.append({
            "name": identity,
            "confidence": round(float(best_similarity) * 100, 1) if best_name else 0.0,
            "box": box,
            "status": status,
            "is_real": is_real,
            "liveness_score": round(real_score, 4),
            "spoof_detected": spoof_detected,
            "authentication_status": authentication_status,
            "message": message,
            
            # CDCN / Liveness Properties
            "face_detected": True,
            "liveness": "spoof" if spoof_detected else "real",
            "real_score": round(real_score, 4),
            "spoof_score": round(spoof_score, 4),
            "blink_detected": blink_detected,
            "head_movements": head_movements,
            "identity_verified": identity_verified,
            
            # CDCN Pipeline specific outputs requested
            "identity": identity,
            "proxy_detected": spoof_detected
        })

    # Ensure multi-face tracking consistency:
    # If same identity appears twice: one real, one spoof, mark: "Duplicate identity with spoof attempt detected"
    identity_indices = {}
    for idx, face in enumerate(results):
        name = face["identity"]
        if name and name != "Invalid User" and face["identity_verified"]:
            if name not in identity_indices:
                identity_indices[name] = []
            identity_indices[name].append(idx)
            
    for name, indices in identity_indices.items():
        if len(indices) >= 2:
            # Check if at least one is spoof
            has_spoof = any(results[idx]["spoof_detected"] for idx in indices)
            if has_spoof:
                print(f"[RECOGNIZE] Duplicate identity '{name}' detected with spoof attempt!")
                for idx in indices:
                    results[idx]["message"] = "Duplicate identity with spoof attempt detected"
                    results[idx]["authentication_status"] = "denied"
                    results[idx]["is_real"] = False
                    results[idx]["proxy_detected"] = True
                    results[idx]["spoof_detected"] = True
                    results[idx]["status"] = "spoof"

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