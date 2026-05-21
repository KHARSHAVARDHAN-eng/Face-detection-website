from pydantic import BaseModel

from typing import Optional

from datetime import datetime


# =========================================
# USER RESPONSE
# =========================================
class UserResponse(BaseModel):

    id: int

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
# DELETE RESPONSE
# =========================================
class DeleteResponse(BaseModel):

    message: str