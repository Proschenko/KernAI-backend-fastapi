# service.py
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from .schemas import LaboratoriesResponse, KernsResponse, KernDetailsResponse,CommentResponse
from datetime import date
import os
import shutil
from fastapi import HTTPException
from typing import List


async def get_labs(session: AsyncSession) -> List[LaboratoriesResponse]:
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


async def get_kerns(session: AsyncSession) -> List[KernsResponse]:
    query = text("""
        SELECT DISTINCT ON (k.kern_code) 
            k.id,
            k.kern_code, 
            l.lab_name, 
            kd.download_date AS last_date, 
            u.user_name,
            d.damage_type 
        FROM kerns k
        JOIN kern_data kd ON k.id = kd.kern_id
        JOIN laboratories l ON kd.lab_id = l.id
        JOIN users u ON kd.user_id = u.id
        LEFT JOIN damages d ON kd.damage_id = d.id
        ORDER BY k.kern_code, kd.download_date DESC;
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