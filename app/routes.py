#routers.py
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from .database import get_session
from . import schemas as schemas
from .import service as serv 

router = APIRouter()

@router.get("/labs", response_model=list[schemas.LaboratoriesRespone])
async def get_organization_wells(session: AsyncSession = Depends(get_session)):
    try:
        # Вызов функции из service.py для получения списка лабораторий 
        labs_data = await serv.get_labs(session)
        return labs_data
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


@router.post("analyze_img")  #TODO MAGMUMS
async def analyze_img1(request: schemas.ImgRequest, session: AsyncSession = Depends(get_session)):
    try:
        pass
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")