# app/routes.py
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from .database import get_session
from . import schemas as schemas
from . import service as serv
from typing import List
from app.utils.celary.redis_config import celery_app
from celery.result import AsyncResult
from app.utils.auth import decode_token, check_role
from uuid import UUID
import os

router = APIRouter()
BASE_IMAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

#TODO Удалить после тестирования а также защитить все API
@router.get("/admin")
async def admin_endpoint(user=Depends(check_role("admin"))):  
    return {"message": f"Welcome, {user['username']}! You have admin access."}

@router.get("/roles")
async def get_user_roles(user=Depends(decode_token)):
    return {
        "message": f"Your token_decoded: {user['token_decoded']}"
    }

#region labs
@router.get("/get_labs", response_model=List[schemas.LaboratoriesResponse], tags=["labs"])
async def get_laboratories(session: AsyncSession = Depends(get_session)):
    try:
        # Вызов функции из service.py для получения списка лабораторий 
        labs_data = await serv.get_labs(session)
        return labs_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
    
@router.get("/get_lab_id/{lab_name}", response_model=UUID, tags=["labs"])
async def get_lab_id_by_name(lab_name: str, session: AsyncSession = Depends(get_session)):
    try:
        # Вызов функции из service.py для получения id лаборатории по имени
        lab_id = await serv.get_lab_id_by_name(lab_name, session)
        if not lab_id:
            raise HTTPException(status_code=404, detail="Laboratory not found")
        return lab_id
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
    
@router.post("/add_lab", response_model=schemas.LaboratoriesResponse, tags=["labs"])
async def add_laboratory(
    lab: schemas.LaboratoriesCreate,
    session: AsyncSession = Depends(get_session),
    user=Depends(check_role("admin"))
):
    try:
        new_lab = await serv.add_lab(session, lab)
        return new_lab
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@router.put("/update_lab/{lab_id}", response_model=schemas.LaboratoriesResponse, tags=["labs"])
async def update_laboratory(
    lab_id: UUID,
    lab: schemas.LaboratoriesCreate,
    session: AsyncSession = Depends(get_session),
    user=Depends(check_role("admin"))
):
    """Обновление информации о лаборатории"""
    try:
        updated_lab = await serv.update_lab(session, lab_id, lab)
        return updated_lab
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@router.delete("/delete_lab/{lab_id}", tags=["labs"])
async def delete_laboratory(
    lab_id: UUID,
    session: AsyncSession = Depends(get_session),
    user=Depends(check_role("admin"))
):
    try:
        await serv.delete_lab(session, lab_id)
        return {"detail": "Laboratory deleted successfully"}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
#endregion

#region kerns
@router.get("/kerns", response_model=List[schemas.KernsResponse], tags=["kerns"])
async def get_kerns_data(session: AsyncSession = Depends(get_session)):
    try:
        # Вызов функции из service.py для получения данных
        kerns_data = await serv.get_kerns(session)
        return kerns_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@router.get("/kern/{kern_id}", response_model=List[schemas.KernDetailsResponse], tags=["kerns"])
async def get_kern_details(kern_id: str, session: AsyncSession = Depends(get_session)):
    try:
        kern_details = await serv.get_kern_details(session, kern_id)
        return kern_details
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@router.get("/kern/{kern_id}/comments", response_model=List[schemas.CommentResponse], tags=["kerns"])
async def get_kern_comments(kern_id: str, session: AsyncSession = Depends(get_session)):
    try:
        comments = await serv.get_kern_comments(session, kern_id)
        return comments
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
    
@router.post("/kern/comments", response_model=schemas.CommentResponse, tags=["kerns"])
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
#endregion

#region work-with-images
@router.post("/upload_img", tags=["work-with-images"])
async def upload_image(file: UploadFile = File(...), user: dict = Depends(decode_token)):
    try:
        # Передаем файл в сервисный слой с именем пользователя для сохранения изображения
        results_dict = await serv.save_image(file, user["username"])
        return {"filename": file.filename, "file_path": results_dict["file_path"], "party_id": results_dict["party_id"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при загрузке изображения: {str(e)}")


@router.post("/analyze_img", tags=["work-with-images"])
async def analyze_image(request: schemas.ImgRequest, user: dict = Depends(decode_token) ):

    """
    Отправляет изображение на фоновую обработку.
    """ 
    try:
        request_data = request.model_dump()
        request_data["user_name"] = user["username"]  # Добавляем username

        task = celery_app.send_task("app.tasks.process_image_task", args=[request_data])
        return {"task_id": task.id}
    except Exception as e:
        return {"error": str(e)}

@router.get("/task_status/{task_id}", tags=["work-with-images"])
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

@router.get("/queue_size", tags=["work-with-images"])
async def get_queue_size():
    try:
        # Вызов функции из service.py для получения данных
        queue_size = await serv.get_queue_size()
        return {"queue_size": queue_size}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@router.get("/get_image", tags=["work-with-images"])
async def get_image(path: str, user: dict = Depends(decode_token)):
    file_path = os.path.join(BASE_IMAGE_DIR, path)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Файл не найден {file_path}")
    return FileResponse(file_path)
#endregion

#region damages
@router.get("/get_damages", response_model=List[schemas.DamageResponse], tags=["damages"])
async def get_damages(session: AsyncSession = Depends(get_session)):
    try:
        damages_data = await serv.get_damages(session)
        return damages_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@router.post("/add_damage", response_model=schemas.DamageResponse, tags=["damages"])
async def add_damage(
    damage: schemas.DamageCreate,
    session: AsyncSession = Depends(get_session),
    user=Depends(check_role("admin"))
):
    try:
        new_damage = await serv.add_damage(session, damage)
        return new_damage
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@router.put("/update_damage/{damage_id}", response_model=schemas.DamageResponse, tags=["damages"])
async def update_damage(
    damage_id: UUID,
    damage: schemas.DamageCreate,
    session: AsyncSession = Depends(get_session),
    user=Depends(check_role("admin"))
):
    """Обновление информации о повреждении"""
    try:
        updated_damage = await serv.update_damage(session, damage_id, damage)
        return updated_damage
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@router.delete("/delete_damage/{damage_id}", tags=["damages"])
async def delete_damage(
    damage_id: UUID,
    session: AsyncSession = Depends(get_session),
    user=Depends(check_role("admin"))
):
    try:
        await serv.delete_damage(session, damage_id)
        return {"detail": "Damage deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
#endregion

#region data
@router.post("/insert_data", tags=["data"])
async def insert_data(
    data: schemas.InsertDataRequest,
    session: AsyncSession = Depends(get_session),
    user: dict = Depends(decode_token)):
    try:
        result_details = await serv.insert_party_info(session, data)
        return  result_details #{"detail": "Данные успешно загружены", "status_code": 200}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
#endregion
