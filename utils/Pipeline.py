import os
from PIL import Image
from tqdm import tqdm
import asyncio
import logging

from FolderOperation import FolderOperation
from ImageOperation import ImageOperation
from KernDetection import KernDetection
from TextRecognition import TextRecognition


async def main(image_path, output_folder, model_path_kern_detection, model_path_text_detection, n_clusters=5,
               conf_threshold=90, excel_data=None):
    """
    Основная функция конвейера для обработки изображений.

    Аргументы:
        image_path: Путь к исходному изображению.
        output_folder: Путь к папке для сохранения результатов.
        model_path_kern_detection: Путь к модели для обнаружения зерен.
        model_path_text_detection: Путь к модели для обнаружения текста.
        n_clusters: Количество кластеров для кластеризации изображений.
        conf_threshold: Порог уверенности для обрезки объектов.
        excel_data: Данные из Excel для сравнения результатов OCR.

    Возвращает:
        list: Список результатов распознавания текста.
    """
    FolderOperation().delete_files_in_folder(output_folder)

    # Инициализация моделей
    kern_detection = KernDetection(model_path_kern_detection)
    text_detection = KernDetection(model_path_text_detection)

    image_operation = ImageOperation()
    model_langs = ['ru']
    allowlist = '0123456789феспФЕСПeECc-*_.,'
    text_recognition = TextRecognition(model_langs, allowlist)

    # Загрузка изображения
    image = Image.open(image_path)

    # Шаг 1: Обрезка зерен с предсказаниями OBB
    logging.info("Начало Шага 1: Обрезка зерен с предсказаниями OBB")
    step1_folder = os.path.join(output_folder, 'step1_crop_kern')
    os.makedirs(step1_folder, exist_ok=True)
    cropped_images = kern_detection.crop_kern_with_obb_predictions(image, save_folder_path=step1_folder,
                                                                   conf_threshold=conf_threshold)
    logging.info("Шаг 1 завершен: Обрезка зерен с предсказаниями OBB")

    # Шаг 2: Обработка изображений с белыми углами
    logging.info("Начало Шага 2: Обработка изображений с белыми углами")
    step2_folder = os.path.join(output_folder, 'step2_white_corners')
    os.makedirs(step2_folder, exist_ok=True)
    processed_images = []
    for img in tqdm(cropped_images, desc="Обработка белых углов"):
        processed_image = image_operation.process_image_white_corners(img, save_folder_path=step2_folder)
        processed_images.append(processed_image)
    logging.info("Шаг 2 завершен: Обработка изображений с белыми углами")

    # Шаг 3: Применение CLAHE
    logging.info("Начало Шага 3: Применение CLAHE")
    step3_folder = os.path.join(output_folder, 'step3_clahe')
    os.makedirs(step3_folder, exist_ok=True)
    clahe_images = []
    for img in tqdm(processed_images, desc="Применение CLAHE"):
        clahe_image = image_operation.clahe_processing(img, save_folder_path=step3_folder)
        clahe_images.append(clahe_image)
    logging.info("Шаг 3 завершен: Применение CLAHE")

    # Шаг 4: Кластеризация изображений
    logging.info("Начало Шага 4: Кластеризация изображений")
    step4_folder = os.path.join(output_folder, 'step4_cluster_image')
    os.makedirs(step4_folder, exist_ok=True)
    clustered_images = []
    for img in tqdm(clahe_images, desc="Кластеризация изображений"):
        clustered_image = image_operation.process_cluster_image(img, n_clusters=n_clusters,
                                                                save_folder_path=step4_folder)
        clustered_images.append(clustered_image)
    logging.info("Шаг 4 завершен: Кластеризация изображений")

    # Шаг 5: Поворот изображений с предсказаниями OBB
    logging.info("Начало Шага 5: Поворот изображений с предсказаниями OBB")
    step5_folder = os.path.join(output_folder, 'step5_rotate_image')
    os.makedirs(step5_folder, exist_ok=True)
    rotated_images = []
    for img in tqdm(clustered_images, desc="Поворот изображений"):
        rotated_image = text_detection.image_rotated_with_obb_predictions(img, save_folder_path=step5_folder)
        rotated_images.append(rotated_image)
    logging.info("Шаг 5 завершен: Поворот изображений с предсказаниями OBB")

    # Шаг 6: Распознавание текста
    logging.info("Начало Шага 6: Распознавание текста")
    result_array = []
    step6_folder = os.path.join(output_folder, 'step6_recognize_text')
    os.makedirs(step6_folder, exist_ok=True)

    for img in tqdm(rotated_images, desc="Распознавание текста"):
        json_out = text_recognition.recognize_text(img, save_folder_path=step6_folder, return_both_results=True)

        if excel_data:
            top_n_predictions = text_recognition.process_ocr_results(json_out, excel_data, n=3)
            # Объединяем json_out и top_n_predictions
            json_out.update(top_n_predictions)

        result_array.append(json_out)
    logging.info("Шаг 6 завершен: Распознавание текста")

    logging.info("Конвейер завершен успешно.")

    return result_array


if __name__ == "__main__":
    image_path = r"D:\я у мамы программист\Diplom\datasets\1 source images\0007.jpg"
    output_folder = r'.\tmp_data\tmp'
    model_path_kern = r".\models\YOLO_detect_kern.pt"
    model_path_text = r".\models\YOLO_detect_text.pt"
    asyncio.run(main(image_path, output_folder, model_path_kern, model_path_text))
