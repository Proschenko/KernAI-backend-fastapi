# app/service.py
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from datetime import date, datetime
import os
import shutil
from fastapi import HTTPException
from typing import List
from tqdm import tqdm
from PIL import Image
import uuid
import logging
from .utils.celary.redis_config import redis_client
from uuid import UUID
from .schemas import (LaboratoriesResponse, KernsResponse, KernDetailsResponse, CommentResponse,
                       ImgRequest, ImgResponse, ImageProcessingResult,CommentCreateRequest, ImgRequestOutter)
from .utils.ImageOperation import ImageOperation
from .utils.KernDetection import KernDetection
from .utils.TextRecognition import EasyOCRTextRecognition, OCRResultSelector, draw_predictions


async def check_and_add_user(session: AsyncSession, username: str) -> UUID:
    """Проверяет, есть ли пользователь в БД, и возвращает его ID. Если нет — добавляет."""
    query = text("SELECT id FROM users WHERE user_name = :user_name")
    result = await session.execute(query, {"user_name": username})
    user = result.fetchone()

    if user:
        return user[0]  # Возвращаем найденный ID

    insert_query = text("INSERT INTO users (user_name) VALUES (:user_name) RETURNING id")
    result = await session.execute(insert_query, {"user_name": username})
    await session.commit()
    
    new_user_id = result.fetchone()[0]
    print(f"Добавлен новый пользователь: {username} (ID: {new_user_id})")
    
    return new_user_id  # Возвращаем ID нового пользователя

async def get_labs(session: AsyncSession) -> List[LaboratoriesResponse]:
    query = text("""
        SELECT id, lab_name
        FROM public.laboratories""")

    result = await session.execute(query)
    labs_data = result.fetchall()
    return [LaboratoriesResponse(**row._mapping) for row in labs_data]

async def get_lab_id_by_name(lab_name: str, session: AsyncSession) -> UUID:
    query = text("""
        SELECT id
        FROM public.laboratories
        WHERE lab_name = :lab_name
    """)
    result = await session.execute(query, {"lab_name": lab_name})
    lab_data = result.fetchone()
    if lab_data:
        return lab_data.id
    return None

async def save_image(file, username: str) -> dict:
    try:
        # Проверка типа файла (только изображения)
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Только изображения могут быть загружены.")
        
        if not username:
            raise HTTPException(status_code=400, detail="Имя пользователя не может быть пустым.")

        # Генерация уникального идентификатора
        party_id = uuid.uuid4()
        party_id_str = str(party_id)

        # Определение структуры папок
        base_dir = os.path.join("temp", username, f"party_{party_id_str}", "inner_img")
        os.makedirs(base_dir, exist_ok=True)

        # Получение расширения оригинального файла
        _, ext = os.path.splitext(file.filename)
        new_filename = f"{party_id_str}{ext}"

        # Полный путь для сохранения
        file_path = os.path.join(base_dir, new_filename)

        # Сохранение файла
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return {"file_path": file_path, "party_id": party_id_str}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке изображения: {str(e)}")

async def get_kerns(session: AsyncSession) -> List[KernsResponse]:
    query = text("""
        SELECT DISTINCT ON (k.kern_code) 
            k.id,
            k.kern_code, 
            l.lab_name, 
            kd.insert_date AS last_date, 
            u.user_name,
            d.damage_type 
        FROM kerns k
        JOIN kern_data kd ON k.id = kd.kern_id
        JOIN laboratories l ON kd.lab_id = l.id
        JOIN users u ON kd.user_id = u.id
        LEFT JOIN damages d ON kd.damage_id = d.id
        ORDER BY k.kern_code, kd.insert_date DESC;
    """)

    result = await session.execute(query)
    kerns_data = result.fetchall()

    return [KernsResponse(**row._mapping) for row in kerns_data]

async def get_kern_details(session: AsyncSession, kern_id: str) -> List[KernDetailsResponse]:
    query = text("""
        SELECT 
            kd.id,
            u.user_name AS insert_user,
            kd.insert_date,
            l.lab_name,
            k.kern_code,
            d.damage_type
        FROM kern_data kd
        JOIN users u ON kd.user_id = u.id
        JOIN laboratories l ON kd.lab_id = l.id
        JOIN kerns k ON kd.kern_id = k.id
        LEFT JOIN damages d ON kd.damage_id = d.id
        WHERE k.id = :kern_id
    """)
    result = await session.execute(query, {"kern_id": kern_id})
    kern_data = result.fetchall()
    if not kern_data:
        raise ValueError("Kern not found")
    return [KernDetailsResponse(**row._mapping) for row in kern_data]

async def get_kern_comments(session: AsyncSession, kern_id: str) -> List[CommentResponse]:
    query = text("""
        SELECT c.id,
               c.insert_date,
               u.user_name as insert_user,
               c.comment_text, 
               k.kern_code,
               l.lab_name
        FROM comments c
        JOIN kerns k ON k.id = c.kern_id
        JOIN laboratories l ON l.id = c.lab_id
        JOIN users u on u.id =c.user_id 
        WHERE c.kern_id = :kern_id
    """)
    result = await session.execute(query, {"kern_id": kern_id})
    comments = result.fetchall()
    return [CommentResponse(**row._mapping) for row in comments]

async def add_kern_comment(
    session: AsyncSession, 
    comment: CommentCreateRequest, 
    user_id: UUID, 
    username: str
) -> CommentResponse:
    """Добавляет комментарий в БД, используя user_id из токена."""

    # Вставляем комментарий в базу
    query = text("""
        INSERT INTO comments (id, insert_date, user_id, kern_id, lab_id, comment_text)
        VALUES (gen_random_uuid(), :insert_date, :user_id, :kern_id, :lab_id, :comment_text)
        RETURNING id, insert_date, comment_text, kern_id, lab_id, user_id
    """)

    params = {
        "insert_date": datetime.now(),
        "user_id": user_id,
        "kern_id": comment.kern_id,
        "lab_id": comment.lab_id,
        "comment_text": comment.comment_text
    }

    result = await session.execute(query, params)
    new_comment = result.fetchone()
    await session.commit()

    # Получаем дополнительные данные (имя пользователя, код керна, название лаборатории)
    query_details = text("""
        SELECT c.id,
               c.insert_date,
               :username as insert_user,  -- Имя передаем напрямую
               c.comment_text,
               k.kern_code,
               l.lab_name
        FROM comments c
        JOIN kerns k ON k.id = c.kern_id
        JOIN laboratories l ON l.id = c.lab_id
        WHERE c.id = :comment_id
    """)
    
    result_details = await session.execute(query_details, {"comment_id": new_comment.id, "username": username})
    comment_data = result_details.fetchone()

    return CommentResponse(**comment_data._mapping)

def process_image(request_data: dict):
    """
    Выполняет обработку изображения через ImagePipelineModel.
    
    :param request_data: Данные запроса в формате словаря (так как Celery не поддерживает Pydantic-объекты)
    :return: Результат обработки в виде JSON-словаря
    """
    request = ImgRequestOutter(**request_data) # Преобразуем словарь в объект Pydantic

    model = ImagePipelineModel(
        request=request,
        yolo_model_path_kern_detection=os.path.join(os.getcwd(), "models", "YOLO_detect_kern.pt").replace("\\", "/"),
        yolo_model_path_text_detection=os.path.join(os.getcwd(), "models", "YOLO_detect_text.pt").replace("\\", "/")
    )

    result = model.execute_pipeline()
    return result.model_dump()  # Возвращаем JSON-словарь для корректной работы с Celery

async def get_queue_size():
    return redis_client.llen("celery")  # Возвращает количество задач в очереди

class ImagePipelineModel:
    def __init__(self, request: ImgRequestOutter, yolo_model_path_kern_detection: str, yolo_model_path_text_detection: str):
        self.request = request
        self.image = Image.open(request.image_path)
        self.output_folder = f"temp\\{str(request.user_name)}\\party_{str(request.party_id)}"
        self.model_path_kern_detection = yolo_model_path_kern_detection
        self.model_path_text_detection = yolo_model_path_text_detection
        self.image_processing = ImageOperation()
        self.kern_detection = KernDetection(yolo_model_path_kern_detection)
        self.kern_text_detection = KernDetection(yolo_model_path_text_detection)
        model_langs = ['ru']
        allowlist = '0123456789феспФЕСПeECc-*_.,'
        self.text_recognition = EasyOCRTextRecognition(model_langs, allowlist)

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
        cropped_paths = [os.path.join(step1_folder, f"kern_{i+1}.png") for i in range(len(cropped_images))]
        
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
        rotated_paths = [os.path.join(step5_folder, f"process_image_{i+1}.png") for i in range(len(rotated_images))]
        
        # Шаг 6: Распознавание текста
        step6_folder = os.path.join(self.output_folder, 'step6_recognize_text')
        os.makedirs(step6_folder, exist_ok=True)

        ocr_selector = OCRResultSelector(self.request.codes)

        for idx, img in enumerate(tqdm(rotated_images, desc="Распознавание текста")):
            # Получаем два варианта результата OCR
            ocr_result_1, ocr_result_2 = self.text_recognition.recognize_text(img)

            draw_predictions((ocr_result_1, ocr_result_2), step6_folder)

            # Выбираем наилучший вариант
            best_result = ocr_selector.select_best_text(ocr_result_1, ocr_result_2)

            draw_predictions(best_result, step6_folder)

            results.append(ImageProcessingResult(
                model_confidence=best_result.ocr_result.confidence_text_ocr, # тут уверенность модели
                predicted_text=best_result.ocr_result.text_ocr, # тут текст, который предсказала модель
                algorithm_text=best_result.text_algoritm, # тут наиболее похожий текст по мнению алгоритма
                kern_code=best_result.text_algoritm, # копия algorithm_text, которая привязывается к виджету на frontend части
                cropped_path=cropped_paths[idx] if idx < len(cropped_paths) else "",
                rotated_path=rotated_paths[idx] if idx < len(rotated_paths) else ""
            ))

        return ImgResponse(
            user_name="test-test",
            codes=self.request.codes,
            lab_id=self.request.lab_id, 
            insert_date=start_time,
            input_type="Изображение" if not self.request.codes else "Изображение + ведомость", #TODO ПЕРЕДЕЛАТЬ НА ТО ЧТО НА ФРОНТЕ ДАЕТСЯ
            download_date=datetime.now().isoformat(),
            processing_results=results
        )
