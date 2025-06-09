# app/utils/ImagePipelineModel.py
import os
from PIL import Image
from tqdm import tqdm
from datetime import date, datetime
# from celery import shared_task # TODO: ckeck progress recorder
# from celery_progress.backend import ProgressRecorder

from ..schemas import ImgRequestOutter, ImgResponse, ImageProcessingResult
from .KernDetection import YOLODetection
from .ImageOperation import ImageOperation
from .TextRecognition import EasyOCRTextRecognition, OCRResultSelector, draw_predictions

class ImagePipelineModel:
    def __init__(self, request: ImgRequestOutter, yolo_model_path_kern_detection: str, yolo_model_path_text_detection: str):
        self.request = request
        self.image = Image.open(request.image_path)
        self.output_folder = f"temp/{str(request.user_name)}/party_{str(request.party_id)}"
        self.model_path_kern_detection = yolo_model_path_kern_detection
        self.model_path_text_detection = yolo_model_path_text_detection
        self.image_processing = ImageOperation()
        self.kern_detection = YOLODetection(yolo_model_path_kern_detection)
        self.kern_text_detection = YOLODetection(yolo_model_path_text_detection)
        model_langs = ['ru']
        allowlist = '0123456789феспФЕСПeECc-*_.,'
        self.text_recognition = EasyOCRTextRecognition(model_langs, allowlist)

    def execute_pipeline(self) -> ImgResponse:
        """
        Выполняет весь pipeline обработки изображения.

        Возвращает:
            Список словарей с результатами обработки.
        """
        start_time=datetime.now().isoformat()
        results = []
        # progress_recorder = ProgressRecorder(self)
        
        # Шаг 1: Детекция керна
        step1_folder = os.path.join(self.output_folder, 'step1_crop_kern')
        os.makedirs(step1_folder, exist_ok=True)
        cropped_images, _ = self.kern_detection.crop_object_with_obb_predictions(self.image, step1_folder)
        cropped_paths = [os.path.join(step1_folder, f"kern_{i+1}.png") for i in range(len(cropped_images))]
        
        # Шаг 2: Фильтрация шума по вписанной окружности
        step2_folder = os.path.join(self.output_folder, 'step2_white_corners')
        os.makedirs(step2_folder, exist_ok=True)
        processed_images = [self.image_processing.process_image_white_corners(img, step2_folder) for img in tqdm(cropped_images, desc="Обработка белых углов")]
        
        # Шаг 3: Применение CLAHE
        step3_folder = os.path.join(self.output_folder, 'step3_clahe')
        os.makedirs(step3_folder, exist_ok=True)
        clahe_images = [self.image_processing.clahe_processing(img, step3_folder) for img in tqdm(processed_images, desc="Применение CLAHE")]
        
        # Шаг 4: Кластеризация
        step4_folder = os.path.join(self.output_folder, 'step4_cluster_image')
        os.makedirs(step4_folder, exist_ok=True)
        clustered_images = [self.image_processing.process_cluster_image(img, save_folder_path=step4_folder) for img in tqdm(clahe_images, desc="Кластеризация")]
        
        # Шаг 5: Поворот изображений
        step5_folder = os.path.join(self.output_folder, 'step5_rotate_image')
        os.makedirs(step5_folder, exist_ok=True)
        rotated_images = [self.kern_text_detection.image_rotated_with_obb_predictions(img, step5_folder) for img in tqdm(clustered_images, desc="Поворот изображений")]
        
        # Шаг 6: Распознавание текста
        step6_folder = os.path.join(self.output_folder, 'step6_recognize_text')
        os.makedirs(step6_folder, exist_ok=True)

        ocr_selector = OCRResultSelector(self.request.codes)
        rotated_paths = [os.path.join(step6_folder, "one_image", f"rotated_text_image_{i+1}.png") for i in range(len(rotated_images))]

        for idx, img in enumerate(tqdm(rotated_images, desc="Распознавание текста")):
            # Получаем два варианта результата OCR
            ocr_result_1, ocr_result_2 = self.text_recognition.recognize_text(img, text_detection=self.kern_text_detection, output_folder=step6_folder)

            draw_predictions((ocr_result_1, ocr_result_2), step6_folder)

            # Выбираем наилучший вариант
            best_result = ocr_selector.select_best_text_ensemble(ocr_result_1, ocr_result_2)

            draw_predictions(best_result, step6_folder)

            results.append(ImageProcessingResult(
                model_confidence=best_result.ocr_result.confidence_text_ocr, # тут уверенность модели
                predicted_text=best_result.ocr_result.text_ocr, # тут текст, который предсказала модель
                algorithm_text=best_result.text_algoritm, # тут наиболее похожий текст по мнению алгоритма
                kern_code=best_result.text_algoritm, # копия algorithm_text, которая привязывается к виджету на frontend части
                cropped_path=cropped_paths[idx] if idx < len(cropped_paths) else "",
                rotated_path=rotated_paths[idx] if idx < len(rotated_paths) else ""
            ))

        return ImgResponse(
            user_name="test-test",
            codes=self.request.codes,
            lab_id=self.request.lab_id, 
            insert_date=start_time,
            input_type="Изображение" if not self.request.codes else "Изображение + ведомость",
            download_date=datetime.now().isoformat(),
            processing_results=results
        )
