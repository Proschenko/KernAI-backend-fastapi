from pydantic import BaseModel
from uuid import UUID
from datetime import date, datetime


class LaboratoriesResponse(BaseModel):
    id: UUID
    lab_name: str


class ImgRequest(BaseModel):
    id_img: UUID
    codes: list[str]
    laboratories_id: UUID
