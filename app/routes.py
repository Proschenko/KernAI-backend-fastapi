# app/routes.py
import logging
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from .database import get_session
from . import schemas as schemas
from . import service as serv
from typing import List
from app.redis_config import celery_app
from celery.result import AsyncResult
from app.utils.auth import decode_token, check_role
from uuid import UUID

router = APIRouter()

#region dev
#TODO Удалить после тестирования а также защитить все API
# 🔒 Защищенный эндпоинт — доступен только авторизованным пользователям
@router.get("/protected")
async def protected_endpoint(user=Depends(decode_token)):
    return {"message": f"Hello, {user['preferred_username']}"}

# 🔒 Доступ только для пользователей с ролью "admin"
@router.get("/admin")
async def admin_endpoint(user=Depends(check_role("admin"))):  # <-- ВАЖНО: Без вызова (без скобок)
    return {"message": "Welcome, Admin!"}

@router.get("/user-info")
async def user_info(user=Depends(decode_token)):
    return {"username": user["username"], "email": user["email"]}
#endregion


@router.get("/labs", response_model=List[schemas.LaboratoriesResponse])
async def get_laboratories(session: AsyncSession = Depends(get_session)):
    try:
        # Вызов функции из service.py для получения списка лабораторий 
        labs_data = await serv.get_labs(session)
        return labs_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
    
@router.get("/lab_id/{lab_name}", response_model=UUID)
async def get_lab_id_by_name(lab_name: str, session: AsyncSession = Depends(get_session)):
    try:
        # Вызов функции из service.py для получения id лаборатории по имени
        lab_id = await serv.get_lab_id_by_name(lab_name, session)
        if not lab_id:
            raise HTTPException(status_code=404, detail="Laboratory not found")
        return lab_id
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")



@router.get("/kerns", response_model=List[schemas.KernsResponse])
async def get_kerns_data(session: AsyncSession = Depends(get_session)):
    try:
        # Вызов функции из service.py для получения данных
        kerns_data = await serv.get_kerns(session)
        return kerns_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@router.get("/kern/{kern_id}", response_model=List[schemas.KernDetailsResponse])
async def get_kern_details(kern_id: str, session: AsyncSession = Depends(get_session)):
    try:
        kern_details = await serv.get_kern_details(session, kern_id)
        return kern_details
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@router.get("/kern/{kern_id}/comments", response_model=List[schemas.CommentResponse])
async def get_kern_comments(kern_id: str, session: AsyncSession = Depends(get_session)):
    try:
        comments = await serv.get_kern_comments(session, kern_id)
        return comments
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
    
@router.post("/kern/comments", response_model=schemas.CommentResponse)
async def add_kern_comment(
    comment: schemas.CommentCreateRequest,
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(decode_token)  # Получаем ID пользователя из токена
):
    try:
        new_comment = await serv.add_kern_comment(session, comment, user["id"], user["username"])
        return new_comment
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


@router.post("/upload_img")
async def upload_image(file: UploadFile = File(...), username: str = ""):
    try:
        # Передаем файл в сервисный слой с именем пользователя для сохранения изображения
        file_path = await serv.save_image(file, username)
        return {"filename": file.filename, "file_path": file_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при загрузке изображения: {str(e)}")


# @router.post("/analyze_img")
# async def analyze_image(request: schemas.ImgRequest, bg_task: BackgroundTasks, session: AsyncSession = Depends(get_session)):
#     # logging.info(request)
#     try:
#         results = bg_task.add_task(serv.process_image, request)
#         return {"results": results}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Ошибка при анализе изображения: {str(e)}")



@router.post("/analyze_img")
async def analyze_image(request: schemas.ImgRequest, session: AsyncSession = Depends(get_session)):
    """
    Отправляет изображение на фоновую обработку.
    """
    try:
        task = celery_app.send_task("app.tasks.process_image_task", args=[request.model_dump()])
        return {"task_id": task.id}
    except Exception as e:
        return {"error": str(e)}


@router.get("/task_status/{task_id}")
async def get_task_status(task_id: str):
    """
    Проверяет статус задачи Celery и возвращает результат, если он готов.
    """
    task_result = AsyncResult(task_id, app=celery_app)

    if task_result.status == "SUCCESS":
        return {"task_id": task_id, "status": task_result.status, "result": task_result.result}
    elif task_result.status == "FAILURE":
        return {"task_id": task_id, "status": task_result.status, "error": str(task_result.result)}
    else:
        return {"task_id": task_id, "status": task_result.status}


@router.get("/queue_size")
async def get_queue_size():
    try:
        # Вызов функции из service.py для получения данных
        queue_size = await serv.get_queue_size()
        return {"queue_size": queue_size}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
    