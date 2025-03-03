# routers.py
import logging
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from .database import get_session
from . import schemas as schemas
from . import service as serv

router = APIRouter()


@router.get("/labs", response_model=list[schemas.LaboratoriesResponse])
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


@router.post("/analyze_img")
async def analyze_image(request: schemas.ImgRequest, session: AsyncSession = Depends(get_session)):
    # logging.info(request)
    try:
        results = serv.process_image(request)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при анализе изображения: {str(e)}")


if __name__ == "__main__":
    img_req = schemas.ImgRequest()
    img_req.image_path = "D:\\я у мамы программист\\Diplom\\datasets\\1 source images\\0007.jpg"
    analyze_image(request=img_req, session=get_session())
