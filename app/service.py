# service.py
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from .schemas import LaboratoriesResponse, ImgRequest
from datetime import date
import os
import shutil
from fastapi import HTTPException
from PIL import Image

from typing import List, Dict
from utils.ImageOperation import ImageOperation
from utils.KernDetection import KernDetection
from utils.TextRecognition import TextRecognition


async def get_labs(session: AsyncSession) -> list[LaboratoriesResponse]:
    query = text("""
        SELECT id, lab_name
        FROM public.laboratories""")

    result = await session.execute(query)
    labs_data = result.fetchall()
    return [LaboratoriesResponse(**row._mapping) for row in labs_data]


async def save_image(file, username: str) -> str:
    try:
        # Проверка типа файла (например, только изображение)
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Только изображения могут быть загружены.")
        if (username is None) or (len(username) == 0):
            username = "non_auth_user"

        # Путь для сохранения файла (директория пользователя)
        user_dir = os.path.join("temp", username)
        os.makedirs(user_dir, exist_ok=True)
        file_path = os.path.join(user_dir, file.filename)

        # Сохранение файла на сервер
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return file_path
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке изображения: {str(e)}")


async def process_image(self):
    model = ImagePipelineModel(
        image_path=ImgRequest.image_path,
        output_folder="temp_output",
        model_path_kern="..models/YOLO_detect_kern.pt",
        model_path_text="..models/YOLO_detect_text.pt"
    )

    result = model.execute_pipeline()
    return result


class ImagePipelineModel:
    def __init__(self, image_path: str, output_folder: str, model_path_kern: str, model_path_text: str):
        self.image_path = image_path
        self.output_folder = output_folder
        self.model_path_kern = model_path_kern
        self.model_path_text = model_path_text
        self.image_processing = ImageOperation()
        self.kern_detection = KernDetection(model_path_kern)
        self.text_recognition = TextRecognition()

    def execute_pipeline(self) -> List[Dict]:
        """
        Выполняет весь pipeline обработки изображения.

        Возвращает:
            Список словарей с результатами обработки.
        """
        result = []

        # Шаг 1: Обрезка зерен
        cropped_images = self.kern_detection.crop_kern_with_obb_predictions(self.image_path, self.output_folder)
        
        # Шаг 2: Обработка белых углов
        processed_images = [self.image_processing.process_image_white_corners(img) for img in cropped_images]
        
        # Шаг 3: Применение CLAHE
        clahe_images = [self.image_processing.clahe_processing(img) for img in processed_images]
        
        # Шаг 4: Кластеризация
        clustered_images = [self.image_processing.process_cluster_image(img) for img in clahe_images]
        
        # Шаг 5: Поворот изображений
        rotated_images = [self.kern_detection.image_rotated_with_obb_predictions(img) for img in clustered_images]
        
        # Шаг 6: Распознавание текста
        for img in rotated_images:
            text_result = self.text_recognition.recognize_text(img)
            result.append(text_result)
        
        return result
    

if __name__ == "__main__":
    img_req = ImgRequest().image_path = "D:\я у мамы программист\Diplom\datasets\1 source images\0007.jpg"
    process_image()