from pydantic import BaseModel
from uuid import UUID
from datetime import date, datetime
from typing import List

class LaboratoriesResponse(BaseModel):
    id: UUID
    lab_name: str

class ImgRequest(BaseModel):
    username : str 
    image_path: str
    codes: List[str]
    lab_id: UUID

class KernsResponse(BaseModel):
    id: UUID
    kern_code: str
    lab_name: str
    last_date: datetime
    user_name: str
    damage_type: str | None

class KernDetailsResponse(BaseModel):
    id: UUID
    insert_user: str
    insert_date: datetime 
    lab_name: str
    kern_code: str
    damage_type: str | None


class CommentResponse(BaseModel):
    id: UUID
    insert_date: datetime 
    insert_user: str 
    comment_text: str 
    kern_code: str
    lab_name: str