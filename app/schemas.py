# app/schemas.py
from pydantic import BaseModel
from uuid import UUID
from datetime import date, datetime
from typing import List


class LaboratoriesResponse(BaseModel):
    id: UUID
    lab_name: str

class ImgRequest(BaseModel):
    user_name: str 
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

class CommentCreateRequest(BaseModel):
    kern_id: UUID
    comment_text: str
    lab_id: UUID

class CommentResponse(BaseModel):
    id: UUID
    insert_date: datetime 
    insert_user: str 
    comment_text: str 
    kern_code: str
    lab_name: str

class ImageProcessingResult(BaseModel):
    model_confidence: float  # Уверенность модели распознавания текста
    predicted_text: str  # Распознанный текст
    algorithm_text: str | None  # Лучшее совпадение с ведомостью | NONE
    cropped_path: str # Путь к обрезанному изображению
    rotated_path: str # Путь к повернутому изображению

class ImgResponse(BaseModel):
    user_name: str  # Имя пользователя
    codes: List[str]  # Список кодов из ведомости
    lab_id: UUID  # Идентификатор лаборатории
    insert_date: datetime  # Дата начала выполнения
    input_type: str  # Тип вставки данных
    download_date: datetime  # Время выполнения алгоритма
    processing_results: List[ImageProcessingResult]  # Список результатов обработки

