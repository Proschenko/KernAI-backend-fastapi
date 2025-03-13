import os
import cv2
import easyocr
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import List, Tuple, Union
from ..schemas import OCRResult, OCRResultSelectorAlgotitm
import logging

class TextRecognitionBase(ABC):
    """Абстрактный класс для распознавания текста."""
    @abstractmethod
    def __init__(self, model_langs: List[str], allowlist: str):
        pass

    @abstractmethod
    def recognize_text(self, image: Image.Image) -> Tuple[OCRResult, OCRResult]:
        """Распознает текст и возвращает два возможных результата."""
        pass


class EasyOCRTextRecognition:
    def __init__(self, model_langs: List[str], allowlist: str):
        self.reader = easyocr.Reader(model_langs, gpu=False)
        self.allowlist = allowlist

    def recognize_text(self, image: Image.Image) -> Tuple[OCRResult, OCRResult]:
        """Распознает текст и возвращает два возможных результата."""

        # Первый проход OCR
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        ocr_result = self.reader.readtext(image_cv, allowlist=self.allowlist)
        bbox = [result[0] for result in ocr_result]
        words = [result[1] for result in ocr_result]
        confidences_words = [float(result[2]) for result in ocr_result]
        confidence_text = sum(confidences_words) / len(confidences_words) if confidences_words else 0.0

        # Второй проход (развернутое изображение)
        rotated_image_180_cv = cv2.rotate(image_cv, cv2.ROTATE_180)
        ocr_result_180 = self.reader.readtext(rotated_image_180_cv, allowlist=self.allowlist)
        bbox_180 = [result[0] for result in ocr_result_180]
        words_180 = [result[1] for result in ocr_result_180]
        confidences_words_180 = [float(result[2]) for result in ocr_result_180]
        confidence_text_180 = sum(confidences_words_180) / len(confidences_words_180) if confidences_words_180 else 0.0

        return (
            OCRResult(
                image=image_cv,
                bbox_ocr=bbox,
                text_ocr=" ".join(words),
                confidence_text_ocr=confidence_text,
                words_ocr=words,
                confidence_words_ocr=confidences_words
            ),
            OCRResult(
                image=rotated_image_180_cv,
                bbox_ocr=bbox_180,
                text_ocr=" ".join(words_180),
                confidence_text_ocr=confidence_text_180,
                words_ocr=words_180,
                confidence_words_ocr=confidences_words_180
            )
        )


class OCRResultSelector:
    def __init__(self, reference_data: List[str]):
        self.reference_data = reference_data
        self.algorithm_name = 'levenshtein_distance'

    def select_best_text(self, result1: OCRResult, result2: OCRResult) -> OCRResultSelectorAlgotitm:
        """Выбирает лучший текст на основе сравнения с эталоном или уверенности модели."""  

        if not self.reference_data:
            # Если reference_data пуст, выбираем по уверенности
            best_result = result1 if result1.confidence_text_ocr >= result2.confidence_text_ocr else result2
            return OCRResultSelectorAlgotitm(ocr_result=best_result, text_algoritm=None)

        # Находим наиболее похожие эталонные тексты для обоих вариантов OCR
        best_match1_text, best_match1_distance = self.find_best_match(result1.text_ocr)
        best_match2_text, best_match2_distance = self.find_best_match(result2.text_ocr)

        if best_match1_distance < best_match2_distance:
            best_result = result1
            best_match_text = best_match1_text
        elif best_match1_distance > best_match2_distance:
            best_result = result2
            best_match_text = best_match2_text
        else:
            # Если расстояния равны, выбираем по уверенности
            best_result = result1 if result1.confidence_text_ocr >= result2.confidence_text_ocr else result2
            best_match_text = best_match1_text  # Можно взять любой, так как расстояния равны

        return OCRResultSelectorAlgotitm(ocr_result=best_result, text_algoritm=best_match_text)

    def find_best_match(self, text: str) -> Tuple[str, int]:
        """Находит наиболее похожий текст из базы данных по Левенштейну."""
        if not text:
            return "", float('inf')  # Если пустой текст, расстояние максимально большое
        matches = [(ref, self.levenshtein_distance(text, ref)) for ref in self.reference_data]
        return min(matches, key=lambda x: x[1])  # Выбираем с минимальным расстоянием

    def levenshtein_distance(self, s1: str, s2: str) -> int:
        """Расчет расстояния Левенштейна между двумя строками."""
        if len(s1) < len(s2):
            s1, s2 = s2, s1  # Гарантируем, что s1 длиннее

        previous_row = np.zeros(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = previous_row + 1
            current_row[1:] = np.minimum(
                current_row[1:], np.add(previous_row[:-1], [c1 != c2 for c2 in s2])
            )
            current_row[1:] = np.minimum(
                current_row[1:], current_row[:-1] + 1
            )
            previous_row = current_row

        return int(previous_row[-1])


def draw_predictions(
    ocr_results: Union[OCRResultSelectorAlgotitm, Tuple[OCRResult, OCRResult]],
    save_folder_path: str
) -> Tuple[np.ndarray, str]:
    """
    Рисует предсказания на изображении и сохраняет результат.

    Аргументы:
        image (Image.Image): Исходное изображение.
        ocr_results (OCRResult | (OCRResult, OCRResult)): Один или два результата OCR.
        save_folder_path (str): Путь к папке для сохранения.

    Возвращает:
        Tuple[np.ndarray, str]: Изображение с разметкой и путь к сохраненному файлу.
    """

    # Определяем, один или два результата
    if isinstance(ocr_results, tuple):
        result_list = list(ocr_results)
        images = [ocr_results[0].image, ocr_results[1].image]
        text_algoritm = None
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # Два изображения
        sub_folder = 'two_images'
    else:
        result_list = [ocr_results.ocr_result]
        images = [ocr_results.ocr_result.image]
        text_algoritm = ocr_results.text_algoritm
        fig, axs = plt.subplots(1, 1, figsize=(6, 6))  # Одно изображение
        axs = [axs]
        sub_folder = 'one_image' 

    # Создаем подкаталог в зависимости от количества изображений
    save_folder = os.path.join(save_folder_path, sub_folder)
    os.makedirs(save_folder, exist_ok=True)

    # Для каждого результата OCR
    for i, (ocr_result, img) in enumerate(zip(result_list, images)):
        annotated_image = img.copy()

        # Отрисовка предсказаний модели (зеленый)
        for word, bbox, confidence in zip(ocr_result.words_ocr, ocr_result.bbox_ocr, ocr_result.confidence_words_ocr):
            if bbox:
                (top_left, _, bottom_right, _) = bbox
                top_left = tuple(map(int, top_left))
                bottom_right = tuple(map(int, bottom_right))

                cv2.rectangle(annotated_image, top_left, bottom_right, (0, 255, 0), 2)  # Зеленый прямоугольник
                text_position = (top_left[0], top_left[1] - 10)
                cv2.putText(
                    annotated_image, f"{word} ({round(confidence, 2)})",
                    text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2  # Красный текст
                )

        # Если у нас OCRResultSelectorAlgotitm — добавляем выделение выбора алгоритма (синий)
        if text_algoritm:
            text_position_algo = (10, annotated_image.shape[0] - 20)  # Внизу слева
            cv2.putText(
                annotated_image, f"Algorithm choice: {text_algoritm}",
                text_position_algo, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 100, 0), 2  # Синий текст
            )

        axs[i].imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
        axs[i].axis("off")

    # Определяем путь для сохранения каждого изображения
    files = os.listdir(save_folder)
    output_path = os.path.join(save_folder, f"recognition_text_image_{len(files) + 1}.png")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)  # Закрываем фигуру, чтобы избежать утечек памяти
