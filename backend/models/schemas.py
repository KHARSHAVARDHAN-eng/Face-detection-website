from pydantic import BaseModel

from typing import Optional, List

from datetime import datetime


# =========================================
# USER RESPONSE
# =========================================
class UserResponse(BaseModel):

    id: str

    name: str

    created_at: datetime

    image_path: Optional[str] = None

    class Config:

        from_attributes = True


# =========================================
# REGISTRATION RESPONSE
# =========================================
class RegistrationResponse(BaseModel):

    success: bool

    message: str


# =========================================
# RECOGNITION RESPONSE
# =========================================
class RecognitionResponse(BaseModel):

    status: str

    name: Optional[str] = None

    message: Optional[str] = None

    confidence: Optional[float] = None

    threshold: Optional[float] = None


# =========================================
# MULTI FACE RECOGNITION
# =========================================
class FaceRecognitionResult(BaseModel):
    name: str
    confidence: float
    box: List[int]
    status: str
    is_real: Optional[bool] = None
    liveness_score: Optional[float] = None
    spoof_detected: Optional[bool] = None
    authentication_status: Optional[str] = None
    message: Optional[str] = None


class MultiRecognitionResponse(BaseModel):
    success: bool
    width: int
    height: int
    faces: List[FaceRecognitionResult]


# =========================================
# DELETE RESPONSE
# =========================================
class DeleteResponse(BaseModel):

    message: str