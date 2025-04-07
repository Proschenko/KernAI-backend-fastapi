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
        yolo_model_path_text_detection=os.path.join(os.getcwd(), "models", "YOLO_detect_text.pt").replace("\\", "/")
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
# Настройка логгера
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

async def insert_party_info(session: AsyncSession, data: InsertDataRequest):
    # Функция вставки партии и кодов из ведомости в базу данных
    logger.info("Начало вставки данных для партии с ID: %s", data.id_party)

    query = text("""SELECT id, id_party
                FROM public.kern_party where id_party = :id_party""")
    result = await session.execute(query, {"id_party": data.id_party}) 
    party = result.fetchone()
    if party:
        logger.warning("Данные по партии с ID %s уже существуют", data.id_party)
        raise HTTPException(status_code=400, detail="Данные по данной партии уже загружены")
    else:
        logger.info("Партия с ID %s не найдена, выполняется вставка", data.id_party)
        query = text("""INSERT INTO public.kern_party
            (id_party)
            VALUES (:id_party)
            RETURNING id""")
        result = await session.execute(query, {
            "id_party": data.id_party
        })
        id_party_outter_key = result.fetchone()[0]
        logger.info("Партия с ID %s успешно добавлена в базу данных", data.id_party)

    # Вставляем коды из data.codes в таблицу kern_party_statements
    if data.kern_party_statements:
        logger.info("Начало вставки кодов для партии с ID %s", data.id_party)
        for code in data.kern_party_statements:
            query = text("""
                INSERT INTO public.kern_party_statements (party_id, kern_code_from_statement)
                VALUES (:party_id, :kern_code_from_statement)
            """)
            await session.execute(query, {
                "party_id": id_party_outter_key,
                "kern_code_from_statement": code,
            })
            logger.info("Код %s успешно добавлен для партии с ID %s", code, data.id_party)

    # Фиксируем изменения в базе данных
    await session.commit()
    logger.info("Все изменения для партии с ID %s успешно зафиксированы", data.id_party)
    return {"detail": "Данные успешно загружены"}
        
async def insert_data(session: AsyncSession, data: InsertDataRequest, user_id: str):

    return  insert_party_info(session, data)
    #return {"detail": "Данные успешно загружены", "status_code": 200}


#endregion
