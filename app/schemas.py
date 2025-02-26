from pydantic import BaseModel
from uuid import UUID
from datetime import date, datetime


class LaboratoriesRespone(BaseModel):
    id : UUID
    labname : str

class ImgRequest(BaseModel):
    id_img: UUID
    codes: list[str]
    laboratories_id = UUID
