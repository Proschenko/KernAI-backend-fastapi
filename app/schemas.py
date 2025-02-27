from pydantic import BaseModel
from uuid import UUID
from datetime import date, datetime


class LaboratoriesResponse(BaseModel):
    id: UUID
    lab_name: str


class ImgRequest(BaseModel):

    username: str 
    image_path: str
    codes: list[str]
    laboratories_id: UUID
