# routers.py
import logging
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from .database import get_session
from . import schemas as schemas
from . import service as serv
from typing import List

router = APIRouter()


@router.get("/labs", response_model=List[schemas.LaboratoriesResponse])
async def get_organization_wells(session: AsyncSession = Depends(get_session)):
    try:
        # Вызов функции из service.py для получения списка лабораторий 
        labs_data = await serv.get_labs(session)
        return labs_data
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


@router.post("/upload_img")
async def upload_image(file: UploadFile = File(...), username: str = ""):
    try:
        # Передаем файл в сервисный слой с именем пользователя для сохранения изображения
        file_path = await serv.save_image(file, username)
        return {"filename": file.filename, "file_path": file_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при загрузке изображения: {str(e)}")




@router.post("/analyze_img")
async def analyze_image(request: schemas.ImgRequest, session: AsyncSession = Depends(get_session)):
    logging.info(request)
    try:
        results = serv.process_image(request)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при анализе изображения: {str(e)}")


