# service.py
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from datetime import date, datetime
import os
import shutil
from fastapi import HTTPException
from tqdm import tqdm
from PIL import Image
from typing import List, Dict
import uuid

import logging
from .schemas import LaboratoriesResponse, ImgRequest, ImgResponse, ImageProcessingResult
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
    logging.info("Current working directory: %s", os.getcwd())

    yolo_model_path_kern_detection = os.path.join(os.getcwd(), "models", "YOLO_detect_kern.pt")
    yolo_model_path_text_detection = os.path.join(os.getcwd(), "models", "YOLO_detect_text.pt")
    logging.info(yolo_model_path_kern_detection)
    logging.info(yolo_model_path_text_detection)
    logging.info(uuid.uuid4())


    model = ImagePipelineModel(
        request=request,
        yolo_model_path_kern_detection = os.path.join(os.getcwd(), "models", "YOLO_detect_kern.pt").replace("\\", "/"),
        yolo_model_path_text_detection = os.path.join(os.getcwd(), "models", "YOLO_detect_text.pt").replace("\\", "/")

        # yolo_model_path_kern_detection="D:/я у мамы программист/Diplom/KernAI-backend-fastapi/models/YOLO_detect_kern.pt",
        # yolo_model_path_text_detection="D:/я у мамы программист/Diplom/KernAI-backend-fastapi/models/YOLO_detect_text.pt"
    )

    result = model.execute_pipeline()
    return result


class ImagePipelineModel:
    def __init__(self, request: ImgRequest, yolo_model_path_kern_detection: str, yolo_model_path_text_detection: str):
        party_uuid = uuid.uuid4()
        party_uuid_str = str(party_uuid)
        self.request = request
        self.image = Image.open(request.image_path)
        self.output_folder = f"temp\\party_{party_uuid_str}"
        self.model_path_kern_detection = yolo_model_path_kern_detection
        self.model_path_text_detection = yolo_model_path_text_detection
        self.image_processing = ImageOperation()
        self.kern_detection = KernDetection(yolo_model_path_kern_detection)
        self.kern_text_detection = KernDetection(yolo_model_path_text_detection)
        model_langs = ['ru']
        allowlist = '0123456789феспФЕСПeECc-*_.,'
        self.text_recognition = TextRecognition(model_langs, allowlist)

    def execute_pipeline(self) -> ImgResponse:
        """
        Выполняет весь pipeline обработки изображения.

        Возвращает:
            Список словарей с результатами обработки.
        """
        start_time=datetime.now().isoformat()
        results = []
        
        # Шаг 1: Обрезка зерен
        step1_folder = os.path.join(self.output_folder, 'step1_crop_kern')
        os.makedirs(step1_folder, exist_ok=True)
        cropped_images = self.kern_detection.crop_kern_with_obb_predictions(self.image, step1_folder)
        cropped_paths = [os.path.join(self.output_folder, f"crop_{i}.png") for i in range(len(cropped_images))]
        
        # Шаг 2: Обработка белых углов
        step2_folder = os.path.join(self.output_folder, 'step2_white_corners')
        os.makedirs(step2_folder, exist_ok=True)
        processed_images = [self.image_processing.process_image_white_corners(img, step2_folder) for img in tqdm(cropped_images, desc="Обработка белых углов")]
        
        # Шаг 3: Применение CLAHE
        step3_folder = os.path.join(self.output_folder, 'step3_clahe')
        os.makedirs(step3_folder, exist_ok=True)
        clahe_images = [self.image_processing.clahe_processing(img, step3_folder) for img in tqdm(processed_images, desc="Применение CLAHE")]
        
        # Шаг 4: Кластеризация
        step4_folder = os.path.join(self.output_folder, 'step4_cluster_image')
        os.makedirs(step4_folder, exist_ok=True)
        clustered_images = [self.image_processing.process_cluster_image(img, save_folder_path=step4_folder) for img in tqdm(clahe_images, desc="Кластеризация")]
        
        # Шаг 5: Поворот изображений
        step5_folder = os.path.join(self.output_folder, 'step5_rotate_image')
        os.makedirs(step5_folder, exist_ok=True)
        rotated_images = [self.kern_text_detection.image_rotated_with_obb_predictions(img, step5_folder) for img in tqdm(clustered_images, desc="Поворот изображений")]
        rotated_paths = [os.path.join(self.output_folder, f"rotated_{i}.png") for i in range(len(rotated_images))]
        
        # Шаг 6: Распознавание текста
        step6_folder = os.path.join(self.output_folder, 'step6_recognize_text')
        os.makedirs(step6_folder, exist_ok=True)
        for idx, img in enumerate(tqdm(rotated_images, desc="Распознавание текста")):
            recognize_result = self.text_recognition.recognize_text(img, return_both_results=True, save_folder_path=step6_folder)

            ocr_confidence = recognize_result.get("ocr_confidence", 0.0)
            ocr_predicted_text = recognize_result.get("ocr_result", "")
            ocr_confidence_180 = recognize_result.get("ocr_confidence_180", 0.0)
            ocr_predicted_text_180 = recognize_result.get("ocr_result_180", "")

            results.append(ImageProcessingResult(
                model_confidence=ocr_confidence,
                predicted_text=ocr_predicted_text,
                algorithm_text=None,
                cropped_path=cropped_paths[idx] if idx < len(cropped_paths) else "",
                rotated_path=rotated_paths[idx] if idx < len(rotated_paths) else ""
            ))

        return ImgResponse(
            user_name=self.request.username,
            codes=self.request.codes,
            laboratories_id=self.request.laboratories_id,  # Заглушка, замени на реальный ID
            insert_date=start_time,
            input_type = "Изображение" if not self.request.codes else "Изображение + ведомость",
            download_date=datetime.now().isoformat(),
            processing_results=results
        )

    

if __name__ == "__main__":
    img_req = ImgRequest(
        image_path="D:\\я у мамы программист\\Diplom\\datasets\\1 source images\\0007.jpg",
        username="user_example",
        codes=[],  # Здесь нужно передать список или оставить пустым
        laboratories_id="cd4f237c-bd89-40d2-b983-05ffcd436b60"  # Пример, если это ID лаборатории
    )
    print(process_image(img_req))