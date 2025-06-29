import os
import cv2
import easyocr
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Tuple, Union
from matplotlib import font_manager
from matplotlib.patches import Rectangle
from ..schemas import OCRResult, OCRResultSelectorAlgotitm
from .KernDetection import YOLODetection
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import textdistance
from collections import Counter


class EasyOCRTextRecognition:
    def __init__(self, model_langs: List[str], allowlist: str):
        self.reader = easyocr.Reader(model_langs, gpu=False)
        self.allowlist = allowlist

    def recognize_text(self, image: Image.Image, text_detection: YOLODetection=None, output_folder: str=r"D:\Diplom\KernAI\KernAI-back-fastapi-celery\temp\test_user\party_test_party") -> Tuple[OCRResult, OCRResult]:
        """Распознает текст и возвращает два возможных результата."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        bboxes = []
        words = []
        confidences_words = []
        confidence_texts = []

        bboxes_180 = []
        words_180 = []
        confidences_words_180 = []
        confidence_texts_180 = []

        # save crop text
        step6_folder_crop_image = os.path.join(str(output_folder), 'crop_text_image')
        os.makedirs(step6_folder_crop_image, exist_ok=True)

        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        image_cv_180 = cv2.rotate(image_cv, cv2.ROTATE_180)
        # text_detection.visualize_predictions_obb(image, conf_threshold=50)
        crop_images, crop_bboxes = text_detection.crop_object_with_obb_predictions(image, conf_threshold=50, save_folder_path=step6_folder_crop_image)
        # print(*crop_images, sep="\n", end="\n\n\n")
        # print(*crop_bboxes, sep="\n", end="\n\n\n")

        sorted_images_with_bboxes = text_detection.sort_bboxes_top_to_bottom(crop_images, crop_bboxes)
        sort_crop_images = [[item[0] for item in crop_image_with_bbox] for crop_image_with_bbox in sorted_images_with_bboxes] 
        sort_crop_bboxes = [[item[1] for item in crop_image_with_bbox] for crop_image_with_bbox in sorted_images_with_bboxes] 
        # print(sort_crop_images, sep="\n", end="\n\n\n")
        # print(*sort_crop_bboxes, sep="\n", end="\n\n\n")

        for crop_image, bbox in zip(sort_crop_images, sort_crop_bboxes):
            # Первый проход OCR
            crop_image_cv = cv2.cvtColor(np.array(crop_image[0]), cv2.COLOR_RGB2BGR)
            ocr_result = self.reader.readtext(crop_image_cv, allowlist=self.allowlist)

            bbox = tuple(tuple([float((round(coord[0], 2))), float(round(coord[1], 2))]) for coord in bbox[0])
            bboxes.append(bbox)
            words.extend([result[1] for result in ocr_result])
            confidences_words.extend([float(result[2]) for result in ocr_result])
            confidence_texts.append((sum(confidences_words) / len(confidences_words)) if confidences_words else 0.0)

            # Второй проход (развернутое изображение)
            crop_rotated_image_180_cv = cv2.rotate(crop_image_cv, cv2.ROTATE_180)
            ocr_result_180 = self.reader.readtext(crop_rotated_image_180_cv, allowlist=self.allowlist)

            bbox_rotated_180 = text_detection.flip_bbox_180(bbox, image.width, image.height)
            bboxes_180.append(bbox_rotated_180)
            words_180.extend([result[1] for result in ocr_result_180])
            confidences_words_180.extend([float(result[2]) for result in ocr_result_180])
            confidence_texts_180.append((sum(confidences_words_180) / len(confidences_words_180)) if confidences_words_180 else 0.0)

        # Переворачиваем результаты для повернутого изображения
        words_180 = words_180[::-1]
        bboxes_180 = bboxes_180[::-1]
        confidences_words_180 = confidences_words_180[::-1]

        return (
            OCRResult(
                image=image_cv,
                bbox_ocr=bboxes,
                text_ocr=" ".join(words),
                confidence_text_ocr=-1 if len(confidence_texts) <= 0 else (sum(confidence_texts) / len(confidence_texts)),
                words_ocr=words,
                confidence_words_ocr=confidences_words
            ),
            OCRResult(
                image=image_cv_180,
                bbox_ocr=bboxes_180,
                text_ocr=" ".join(words_180),
                confidence_text_ocr=-1 if len(confidence_texts_180) <= 0 else (sum(confidence_texts_180) / len(confidence_texts_180)),
                words_ocr=words_180,
                confidence_words_ocr=confidences_words_180
            )
        )


class OCRResultSelectorOld:
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


class OCRResultSelector:
    def __init__(self, reference_data: List[str]):
        self.reference_data = reference_data
        self.algorithm_name = 'ensemble'

    def select_best_text_ensemble(self, result1: OCRResult, result2: OCRResult) -> OCRResultSelectorAlgotitm:

        if not self.reference_data:
            # Если reference_data пуст, выбираем по уверенности
            best_result = result1 if result1.confidence_text_ocr >= result2.confidence_text_ocr else result2
            return OCRResultSelectorAlgotitm(ocr_result=best_result, text_algoritm=None)

        if result1.confidence_text_ocr == -1 and result2.confidence_text_ocr == -1:
            return OCRResultSelectorAlgotitm(ocr_result=result1, text_algoritm=None)

        methods = [
            (self.find_best_match_levenshtein, False),
            (self.find_best_match_seq_match, True),
            # (self.find_best_match_jaccard, True),
            # (self.find_best_match_ngram, True),
            # (self.find_best_match_cosine, True),
            (self.find_best_match_jaro, True),
            (self.find_best_match_needleman, True),
        ]

        votes = []

        for method, maximize in methods: # TODO: add colldection statics
            match1, score1 = method(result1.text_ocr)
            match2, score2 = method(result2.text_ocr)
            # print("-"*5, method.__name__, "-"*5)
            # print("True rotation", match1, score1, f"text_ocr={result1.text_ocr}")
            # print("False rotation", match2, score2, f"text_ocr={result2.text_ocr}")
            # print("\n\n\n")

            if maximize:
                if score1 > score2:
                    votes.append((result1, match1))
                elif score2 > score1:
                    votes.append((result2, match2))
                else:
                    best_text_ocr = result1.text_ocr if result1.confidence_text_ocr >= result2.confidence_text_ocr else result2.text_ocr
                    best_result = result1 if result1.confidence_text_ocr >= result2.confidence_text_ocr else result2
                    best_match = match1 if best_text_ocr == result1.text_ocr else match2
                    votes.append((best_result, best_match))
            else:
                if score1 < score2:
                    votes.append((result1, match1))
                elif score2 < score1:
                    votes.append((result2, match2))
                else:
                    best_text_ocr = result1.text_ocr if result1.confidence_text_ocr >= result2.confidence_text_ocr else result2.text_ocr
                    best_result = result1 if result1.confidence_text_ocr >= result2.confidence_text_ocr else result2
                    best_match = match1 if best_text_ocr == result1.text_ocr else match2
                    votes.append((best_result, best_match))

        # Группировка по match (эталонному тексту)
        match_counts = Counter([vote[1] for vote in votes])
        best_match_text = match_counts.most_common(1)[0][0]

        # Среди тех, кто выбрал этот match, считаем, кто из result'ов выигрывает по количеству голосов
        candidate_results = [vote[0].text_ocr for vote in votes if vote[1] == best_match_text]
        best_result_text_ocr = Counter(candidate_results).most_common(1)[0][0]
        best_result = result1 if result1.text_ocr == best_result_text_ocr else result2

        return OCRResultSelectorAlgotitm(ocr_result=best_result, text_algoritm=best_match_text)

    def find_best_match_levenshtein(self, text: str) -> Tuple[str, float]:
        if not text:
            return "", float('inf')
        return min(
            ((ref, self.levenshtein_distance(text, ref)) for ref in self.reference_data),
            key=lambda x: x[1]
        )

    def levenshtein_distance(self, s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            s1, s2 = s2, s1
        previous_row = np.arange(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = previous_row + 1
            current_row[1:] = np.minimum(
                current_row[1:], np.add(previous_row[:-1], [c1 != c2 for c2 in s2])
            )
            current_row[1:] = np.minimum(current_row[1:], current_row[:-1] + 1)
            previous_row = current_row
        return int(previous_row[-1])

    def find_best_match_seq_match(self, text: str) -> Tuple[str, float]:
        if not text:
            return "", 0.0
        return max(
            ((ref, SequenceMatcher(None, text, ref).ratio()) for ref in self.reference_data),
            key=lambda x: x[1]
        )

    def find_best_match_jaccard(self, text: str) -> Tuple[str, float]:
        def jaccard(a, b):
            set_a, set_b = set(a), set(b)
            return len(set_a & set_b) / len(set_a | set_b) if set_a | set_b else 0

        return max(
            ((ref, jaccard(text, ref)) for ref in self.reference_data),
            key=lambda x: x[1]
        )

    def find_best_match_ngram(self, text: str, n=3) -> Tuple[str, float]:
        def ngram_similarity(a, b, n):
            ngrams_a = set([a[i:i + n] for i in range(len(a) - n + 1)])
            ngrams_b = set([b[i:i + n] for i in range(len(b) - n + 1)])
            return len(ngrams_a & ngrams_b) / len(ngrams_a | ngrams_b) if ngrams_a | ngrams_b else 0

        return max(
            ((ref, ngram_similarity(text, ref, n)) for ref in self.reference_data),
            key=lambda x: x[1]
        )

    def find_best_match_cosine(self, text: str) -> Tuple[str, float]:
        if not text or not self.reference_data:
            return "", 0.0
        texts = [text] + self.reference_data
        vectorizer = TfidfVectorizer().fit(texts)
        tfidf_matrix = vectorizer.transform(texts)
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        best_index = int(np.argmax(similarities))
        return self.reference_data[best_index], similarities[best_index]

    def find_best_match_jaro(self, text: str) -> Tuple[str, float]:
        return max(
            ((ref, textdistance.jaro.normalized_similarity(text, ref)) for ref in self.reference_data),
            key=lambda x: x[1]
        )

    def find_best_match_needleman(self, text: str) -> Tuple[str, float]:
        return max(
            ((ref, textdistance.needleman_wunsch.normalized_similarity(text, ref)) for ref in self.reference_data),
            key=lambda x: x[1]
        )

def draw_predictions(
    ocr_results: Union[OCRResultSelectorAlgotitm, Tuple[OCRResult, OCRResult]],
    save_folder_path: str
) -> Tuple[np.ndarray, str]:
    """
    Рисует предсказания на изображении и сохраняет результат.

    Аргументы:
        ocr_results (OCRResult | (OCRResult, OCRResult)): Один или два результата OCR.
        save_folder_path (str): Путь к папке для сохранения.

    Возвращает:
        Tuple[np.ndarray, str]: Изображение с разметкой и путь к сохраненному файлу.
    """

    if isinstance(ocr_results, tuple):
        result_list = list(ocr_results)
        images = [ocr_results[0].image, ocr_results[1].image]
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        sub_folder = 'two_images'
    else:
        result_list = [ocr_results.ocr_result]
        images = [ocr_results.ocr_result.image]
        fig, axs = plt.subplots(1, 1, figsize=(6, 6))
        axs = [axs]
        sub_folder = 'one_image'

    save_folder = os.path.join(save_folder_path, sub_folder)
    os.makedirs(save_folder, exist_ok=True)

    font_prop = font_manager.FontProperties(family='DejaVu Sans', size=20)

    for i, (ocr_result, img) in enumerate(zip(result_list, images)):
        rotated_image = img.copy()
        annotated = img.copy()

        axs[i].imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        axs[i].axis("off")

        for word, bbox, confidence in zip(ocr_result.words_ocr, ocr_result.bbox_ocr, ocr_result.confidence_words_ocr):
            if not bbox:
                continue

            (tl, _, br, _) = bbox
            x1, y1 = map(int, tl)
            x2, y2 = map(int, br)

            # Рисуем bbox
            rect_w, rect_h = x2 - x1, y2 - y1
            axs[i].add_patch(Rectangle((x1, y1), rect_w, rect_h, linewidth=2, edgecolor='lime', facecolor='none'))

            # Подготовка текста
            text = f"{word} ({round(confidence, 2)})"
            text_x = x1
            text_y = y1 - 5

            # Если текст выходит за верх — опускаем вниз
            if text_y < 10:
                text_y = y1 + 15

            axs[i].text(
                text_x, text_y, text,
                fontproperties=font_prop,
                color='red',
                verticalalignment='top',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1)
            )

    # Определяем путь для сохранения каждого изображения
    files = os.listdir(save_folder)
    recognition_output_path = os.path.join(save_folder, f"recognition_text_image_{len(files) // 2 + 1}.png")
    rotated_output_path = os.path.join(save_folder, f"rotated_text_image_{len(files) // 2 + 1}.png")

    plt.tight_layout()
    if not isinstance(ocr_results, tuple):
        cv2.imwrite(rotated_output_path, rotated_image)
    else:
        recognition_output_path = os.path.join(save_folder, f"recognition_text_image_{len(files) + 1}.png")
    plt.savefig(recognition_output_path)
    plt.close(fig)

    return annotated, recognition_output_path

if __name__ == "__main__":
    image_path = r"C:\Users\magmy\Downloads\Telegram Desktop\party_6ecc441e-dd69-4139-87bf-63528d4589fb\party_6ecc441e-dd69-4139-87bf-63528d4589fb\step5_rotate_image\process_image_18.png"
    img = Image.open(image_path)

    model_path = r"D:\я у мамы программист\Diplom\KernAI-backend-fastapi\models\YOLO_detect_text_v.4.pt"
    kern_text_detection = YOLODetection(model_path)

    ocr_selector = OCRResultSelector(None)
    step6_folder = r"D:\я у мамы программист\Diplom\KernAI-backend-fastapi\temp\user1\step6_recognize_text"
    model_langs = ['ru']
    allowlist = '0123456789феспФЕСПeECc-*_.,'
    text_recognition = EasyOCRTextRecognition(model_langs, allowlist)
    # Получаем два варианта результата OCR
    ocr_result_1, ocr_result_2 = text_recognition.recognize_text(img, text_detection=kern_text_detection, output_folder=step6_folder)

    draw_predictions((ocr_result_1, ocr_result_2), step6_folder)

    # Выбираем наилучший вариант
    best_result = ocr_selector.select_best_text_ensemble(ocr_result_1, ocr_result_2)

    draw_predictions(best_result, step6_folder)