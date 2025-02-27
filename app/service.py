# service.py
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from .schemas import LaboratoriesResponse, ImgRequest
from datetime import date
import os
import shutil
from fastapi import HTTPException
from tqdm import tqdm
from PIL import Image

from typing import List, Dict
from .ImageOperation import ImageOperation
from .KernDetection import KernDetection
from .TextRecognition import TextRecognition



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


def process_image(request: ImgRequest):
    model = ImagePipelineModel(
        image_path=request.image_path,
        output_folder="temp_output",
        # yolo_model_path_kern_detection="..models/YOLO_detect_kern.pt",
        # yolo_model_path_text_detection="..models/YOLO_detect_text.pt"
        yolo_model_path_kern_detection="D:/я у мамы программист/Diplom/KernAI-backend-fastapi/models/YOLO_detect_kern.pt",
        yolo_model_path_text_detection="D:/я у мамы программист/Diplom/KernAI-backend-fastapi/models/YOLO_detect_text.pt"

    )

    result = model.execute_pipeline()
    return result


class ImagePipelineModel:
    def __init__(self, image_path: str, output_folder: str, yolo_model_path_kern_detection: str, yolo_model_path_text_detection: str):
        self.image_path = image_path
        self.image = Image.open(image_path)
        self.output_folder = output_folder
        self.model_path_kern_detection = yolo_model_path_kern_detection
        self.model_path_text_detection = yolo_model_path_text_detection
        self.image_processing = ImageOperation()
        self.kern_detection = KernDetection(yolo_model_path_kern_detection)
        self.kern_text_detection = KernDetection(yolo_model_path_text_detection)
        model_langs = ['ru']
        allowlist = '0123456789феспФЕСПeECc-*_.,'
        self.text_recognition = TextRecognition(model_langs, allowlist)

    def execute_pipeline(self) -> List[Dict]:
        """
        Выполняет весь pipeline обработки изображения.

        Возвращает:
            Список словарей с результатами обработки.
        """
        result = []

        # Шаг 1: Обрезка зерен
        cropped_images = self.kern_detection.crop_kern_with_obb_predictions(self.image, self.output_folder)

        # Шаг 2: Обработка белых углов
        processed_images = [self.image_processing.process_image_white_corners(img) for img in tqdm(cropped_images, desc="Обработка белых углов")]

        # Шаг 3: Применение CLAHE
        clahe_images = [self.image_processing.clahe_processing(img) for img in tqdm(processed_images, desc="Применение CLAHE")]

        # Шаг 4: Кластеризация
        clustered_images = [self.image_processing.process_cluster_image(img) for img in tqdm(clahe_images, desc="Кластеризация")]

        # Шаг 5: Поворот изображений
        rotated_images = [self.kern_text_detection.image_rotated_with_obb_predictions(img) for img in tqdm(clustered_images, desc="Поворот изображений")]

        # Шаг 6: Распознавание текста
        for img in tqdm(rotated_images, desc="Распознавание текста"):
            text_result = self.text_recognition.recognize_text(img)
            result.append(text_result)

        return result
    

if __name__ == "__main__":
    img_req = ImgRequest(
        image_path="D:\\я у мамы программист\\Diplom\\datasets\\1 source images\\0007.jpg",
        username="user_example",
        codes=[],  # Здесь нужно передать список или оставить пустым
        laboratories_id="cd4f237c-bd89-40d2-b983-05ffcd436b60"  # Пример, если это ID лаборатории
    )
    print(process_image(img_req))