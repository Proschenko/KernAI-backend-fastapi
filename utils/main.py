from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from typing import List, Optional
import os
import shutil
import Pipeline
import logging
from pydantic import BaseModel
import json


class ExcelDataItem(BaseModel):
    data: List[str]


class ExcelData(BaseModel):
    excelData: Optional[List[ExcelDataItem]] = None


# Настройка основного логгера
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

# Настройка логгера для ошибок
error_logger = logging.getLogger('error_logger')
error_logger.setLevel(logging.ERROR)
error_handler = logging.FileHandler('app_error.log')
error_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
error_logger.addHandler(error_handler)

app = FastAPI()

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Разрешенные источники
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/analyze_image")
async def analyze_image(image: UploadFile = File(...), excelData: Optional[str] = Form(None)):
    """
    Обрабатывает загруженное изображение и возвращает результаты распознавания текста.

    Аргументы:
        image: Загруженное изображение.
        excelData: Данные из Excel для сравнения результатов OCR (необязательно).

    Возвращает:
        JSONResponse: Результаты распознавания текста.
    """
    try:
        # Сохраняем загруженное изображение
        file_location = f"./tmp_data/inner_image/{image.filename}"
        os.makedirs(os.path.dirname(file_location), exist_ok=True)
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # Парсим excelData, если оно передано
        if excelData:
            excel_data_list = json.loads(excelData)
            excel_data = [[item] for sublist in excel_data_list for item in sublist]
            logging.info(f"Parsed excelData: {excel_data}")
        else:
            excel_data = None

        # Возвращаем массив изображений с атрибутами
        image_path = file_location
        output_folder = r'./tmp_data/tmp'
        model_path_kern = r"./models/YOLO_detect_kern.pt"
        model_path_text = r"./models/YOLO_detect_text.pt"

        result_array = await Pipeline.main(image_path, output_folder, model_path_kern, model_path_text,
                                           excel_data=excel_data)

        results = []
        logging.info(f"count kern: {len(result_array)}")
        for json_item in result_array:
            filename1 = json_item['path']

            results.append({
                'filename': filename1,
                'ocr_result1': str(json_item['ocr_reuslt']),
                'ocr_result2': str(json_item['ocr_result_180']),
                'best_matches_ocr_reuslt': str(json_item['best_matches_ocr_reuslt']),
                'best_matches_ocr_result2': str(json_item['best_matches_ocr_result_180']),
            })

        return JSONResponse(content=results)
    except Exception as e:
        logging.error(f'Ошибка: {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))


async def run_pipeline(image_path, output_folder, model_path_kern, model_path_text):
    """
    Запускает конвейер обработки изображений.

    Аргументы:
        image_path: Путь к исходному изображению.
        output_folder: Путь к папке для сохранения результатов.
        model_path_kern: Путь к модели для обнаружения зерен.
        model_path_text: Путь к модели для обнаружения текста.

    Возвращает:
        list: Список результатов распознавания текста.
    """
    result_array = await Pipeline.main(image_path, output_folder, model_path_kern, model_path_text)
    return result_array


@app.get("/api/get_image")
async def get_image(filename: str):
    """
    Возвращает изображение по заданному имени файла.

    Аргументы:
        filename: Имя файла изображения.

    Возвращает:
        FileResponse: Изображение.
    """
    try:
        # Проверяем, существует ли файл
        file_path = f"{filename}"
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        # Возвращаем изображение
        return FileResponse(file_path)
    except Exception as e:
        error_logger.error(f'Ошибка: {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))


# Запуск приложения
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
    # uvicorn main:app --host 0.0.0.0 --port 8000 --reload
