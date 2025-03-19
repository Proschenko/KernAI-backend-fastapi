# app/schemas.py
from pydantic import BaseModel, ConfigDict
from uuid import UUID
from datetime import date, datetime
from typing import List, Tuple
import numpy as np

#region inner project schemas
class ImageProcessingResult(BaseModel):
    model_confidence: float  # Уверенность модели распознавания текста
    predicted_text: str  # Распознанный текст
    algorithm_text: str | None  # Лучшее совпадение с ведомостью | NONE
    kern_code: str | None  # копия algorithm_text, которая привязывается к виджету на frontend части
    cropped_path: str # Путь к обрезанному изображению
    rotated_path: str # Путь к повернутому изображению

class OCRResult(BaseModel):
    """Схема возврата данных после распознавания текста."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    image: np.array # Изображение в формате NumPy
    bbox_ocr: List[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]]  # Координаты bbox [(top_left, top_right, bottom_right, bottom_left)]
    text_ocr: str  # Полный текст
    confidence_text_ocr: float  # Уверенность модели
    words_ocr: List[str]  # Отдельные слова
    confidence_words_ocr: List[float]  # Уверенность модели

class OCRResultSelectorAlgotitm(BaseModel):
    """Схема возврата данных после выбора наиболее похожего кода из ведомости."""
    ocr_result: OCRResult
    text_algoritm: str | None

class ImgRequestOutter(BaseModel):
    user_name: str  
    party_id : UUID
    image_path: str
    codes: List[str]
    lab_id: UUID
#endregion


class LaboratoriesResponse(BaseModel):
    id: UUID
    lab_name: str

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


class ImgRequest(BaseModel):
    party_id : UUID
    image_path: str
    codes: List[str]
    lab_id: UUID


class ImgResponse(BaseModel):
    user_name: str  # Имя пользователя
    codes: List[str]  # Список кодов из ведомости
    lab_id: UUID  # Идентификатор лаборатории
    insert_date: datetime  # Дата начала выполнения
    input_type: str  # Тип вставки данных
    download_date: datetime  # Время выполнения алгоритма
    processing_results: List[ImageProcessingResult]  # Список результатов обработки
