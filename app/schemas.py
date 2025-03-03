from pydantic import BaseModel
from uuid import UUID
from datetime import date, datetime
from typing import List, Dict

class LaboratoriesResponse(BaseModel):
    id: UUID
    lab_name: str

class ImgRequest(BaseModel):
    username: str 
    image_path: str
    codes: List[str]
    laboratories_id: UUID

class ImageProcessingResult(BaseModel):
    model_confidence: float  # Уверенность модели распознавания текста
    predicted_text: str  # Распознанный текст
    algorithm_text: str | None  # Лучшее совпадение с ведомостью | NONE
    cropped_path: str # Путь к обрезанному изображению
    rotated_path: str # Путь к повернутому изображению

class ImgResponse(BaseModel):
    user_name: str  # Имя пользователя
    codes: List[str]  # Список кодов из ведомости
    laboratories_id: UUID  # Идентификатор лаборатории
    insert_date: datetime  # Дата начала выполнения
    input_type: str  # Тип вставки данных
    download_date: datetime  # Время выполнения алгоритма
    processing_results: List[ImageProcessingResult]  # Список результатов обработки
