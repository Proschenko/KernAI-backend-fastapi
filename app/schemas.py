from pydantic import BaseModel
from uuid import UUID
from datetime import date, datetime

class Laboratories(BaseModel):
    id : UUID
    labname : str
