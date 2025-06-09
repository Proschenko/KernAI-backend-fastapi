# app/service.py
import os
import uuid
import shutil
import logging
from uuid import UUID
from typing import List
from sqlalchemy import text
from fastapi import HTTPException
from datetime import date, datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError

from .utils.celary.redis_config import redis_client
from .schemas import *
from .utils.ImagePipelineModel import ImagePipelineModel


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

async def add_lab(session: AsyncSession, lab: LaboratoriesCreate) -> LaboratoriesResponse:
    query = text("""
        INSERT INTO laboratories (lab_name)
        VALUES (:lab_name)
        RETURNING id, lab_name
    """)
    result = await session.execute(query, {"lab_name": lab.lab_name})
    await session.commit()
    lab_data = result.fetchone()
    return LaboratoriesResponse(id=lab_data.id, lab_name=lab_data.lab_name)

async def update_lab(session: AsyncSession, lab_id: UUID, lab: LaboratoriesCreate) -> LaboratoriesResponse:
    """Обновление лаборатории"""
    query = text("""
        UPDATE laboratories
        SET lab_name = :lab_name
        WHERE id = :lab_id
        RETURNING id, lab_name
    """)
    result = await session.execute(query, {"lab_name": lab.lab_name, "lab_id": lab_id})
    await session.commit()
    lab_data = result.fetchone()
    if not lab_data:
        raise HTTPException(status_code=404, detail="Laboratory not found")
    return LaboratoriesResponse(id=lab_data.id, lab_name=lab_data.lab_name)

async def delete_lab(session: AsyncSession, lab_id: UUID):
    query = text("DELETE FROM laboratories WHERE id = :lab_id RETURNING id")
    result = await session.execute(query, {"lab_id": lab_id})
    await session.commit()
    if not result.fetchone():
        raise HTTPException(status_code=404, detail="Laboratory not found")

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
        yolo_model_path_text_detection=os.path.join(os.getcwd(), "models", "YOLO_detect_text_v.4.pt").replace("\\", "/")
    )

    result = model.execute_pipeline()
    return result.model_dump()  # Возвращаем JSON-словарь для корректной работы с Celery

async def get_queue_size():
    return redis_client.llen("celery")  # Возвращает количество задач в очереди

async def get_damages(session: AsyncSession) -> List[DamageResponse]:
    query = text("""
        SELECT id, damage_type
        FROM public.damages
    """)

    result = await session.execute(query)
    damages_data = result.fetchall()
    return [DamageResponse(**row._mapping) for row in damages_data]

async def add_damage(session: AsyncSession, damage: DamageCreate) -> DamageResponse:
    query = text("""
        INSERT INTO damages (damage_type)
        VALUES (:damage_type)
        RETURNING id,damage_type
    """)
    result = await session.execute(query, {
        "damage_type": damage.damage_type,
    })
    await session.commit()
    damage_data = result.fetchone()
    return DamageResponse(id=damage_data.id, damage_type=damage_data.damage_type)

async def update_damage(session: AsyncSession, damage_id: UUID, damage: DamageCreate) -> DamageResponse:
    """Обновление повреждения"""
    query = text("""
        UPDATE damages
        SET damage_type = :damage_type
        WHERE id = :damage_id
        RETURNING id, damage_type
    """)
    result = await session.execute(query, {"damage_type": damage.damage_type, "damage_id": damage_id})
    await session.commit()
    damage_data = result.fetchone()
    if not damage_data:
        raise HTTPException(status_code=404, detail="Damage not found")
    return DamageResponse(id=damage_data.id, damage_type=damage_data.damage_type)

async def delete_damage(session: AsyncSession, damage_id: UUID):
    query = text("DELETE FROM damages WHERE id = :damage_id RETURNING id")
    result = await session.execute(query, {"damage_id": damage_id})
    await session.commit()
    if not result.fetchone():
        raise HTTPException(status_code=404, detail="Damage not found")

#region insert_data
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

async def insert_all_data(session: AsyncSession, data, user_id: str):
    """
    Вставка данных в БД, используя новую структуру таблиц.
    """
    try:
        # 1. Проверяем, существует ли уже партия
        query = text("SELECT id FROM public.kern_party WHERE party_id = :party_id")
        result = await session.execute(query, {"party_id": data.id_party})
        party = result.fetchone()
        
        if party:
            logger.warning("Партия %s уже загружена", data.id_party)
            raise HTTPException(status_code=400, detail="Данные по данной партии уже загружены")
        
        # 2. Вставляем новую партию
        query = text("""
            INSERT INTO public.kern_party (party_id)
            VALUES (:party_id)
            RETURNING id
        """)
        result = await session.execute(query, {"party_id": data.id_party})
        id_party_outter_key = result.fetchone()[0]
        await session.flush()
        
        # 3. Вставляем коды ведомости
        for code in data.kern_party_statements:
            query = text("""
                INSERT INTO public.kern_party_statements (party_id, kern_code_from_statement)
                VALUES (:party_id, :kern_code_from_statement)
            """)
            await session.execute(query, {"party_id": id_party_outter_key, "kern_code_from_statement": code})
        
        # 4. Обрабатываем каждую строку данных
        for row in data.rows:
            # 4.1. Проверяем, существует ли kern_code
            query = text("SELECT id FROM public.kerns WHERE kern_code = :kern_code LIMIT 1")
            result = await session.execute(query, {"kern_code": row.kern_code})
            kern = result.fetchone()
            
            if kern:
                kern_id = kern[0]
            else:
                query = text("""
                    INSERT INTO public.kerns (kern_code)
                    VALUES (:kern_code)
                    RETURNING id
                """)
                result = await session.execute(query, {"kern_code": row.kern_code})
                kern_id = result.fetchone()[0]
            
            # 4.2. Вставляем данные аналитики
            query = text("""
                INSERT INTO public.kern_data_analytics (
                    confidence_model, code_model, code_algorithm, input_type, download_date, validation_date
                ) VALUES (
                    :confidence_model, :code_model, :code_algorithm, :input_type, :download_date, :validation_date
                )
                RETURNING id
            """)
            result = await session.execute(query, {
                "confidence_model": row.confidence_model,
                "code_model": row.code_model,
                "code_algorithm": row.code_algorithm,
                "input_type": data.input_type,
                "download_date": data.download_date,
                "validation_date": data.validation_date
            })
            analytic_id = result.fetchone()[0]
            
            # 4.3. Вставляем данные кернов
            query = text("""
                INSERT INTO public.kern_data (
                    id_party, user_id, insert_date, lab_id, kern_id, damage_id, analytic_id
                ) VALUES (
                    :id_party, :user_id, :insert_date, :lab_id, :kern_id, :damage_id, :analytic_id
                )
            """)
            await session.execute(query, {
                "id_party": id_party_outter_key,
                "user_id": user_id,
                "insert_date": data.insert_date,
                "lab_id": data.lab_id,
                "kern_id": kern_id,
                "damage_id": row.damage_id,
                "analytic_id": analytic_id
            })
        
        await session.commit()
        logger.info("Все данные успешно загружены")
        return {"detail": "Данные успешно загружены"}
    except SQLAlchemyError as e:
        await session.rollback()
        logger.error("Ошибка при вставке данных: %s", str(e))
        raise HTTPException(status_code=500, detail="Ошибка при вставке данных")

