#service.py
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from .schemas import LaboratoriesRespone
from datetime import date
import os
import shutil
from fastapi import HTTPException

async def get_labs(session: AsyncSession)-> list[LaboratoriesRespone]:
    query=text("""
        SELECT id, labname
        FROM public.laboratories""")
    
    result = await session.execute(query)
    labs_data =  result.fetchall()
    return [LaboratoriesRespone(**row._mapping) for row in labs_data]


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
