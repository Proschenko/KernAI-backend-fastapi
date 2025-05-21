import os
import cv2
import easyocr
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import List, Tuple, Union
from ..schemas import OCRResult, OCRResultSelectorAlgotitm
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import textdistance
from collections import Counter

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
        methods = [
            (self.find_best_match_levenshtein, False),
            (self.find_best_match_seq_match, True),
            (self.find_best_match_jaccard, True),
            (self.find_best_match_ngram, True),
            (self.find_best_match_cosine, True),
            (self.find_best_match_jaro, True),
            (self.find_best_match_needleman, True),
        ]

        votes = []

        for method, maximize in methods:
            match1, score1 = method(result1.text_ocr)
            match2, score2 = method(result2.text_ocr)
            print("-"*5, method.__name__, "-"*5)
            print("True rotation", match1, score1, f"text_ocr={result1.text_ocr}")
            print("False rotation", match2, score2, f"text_ocr={result2.text_ocr}")
            print("\n\n\n")

            if maximize:
                if score1 > score2:
                    votes.append((result1, match1))
                elif score2 > score1:
                    votes.append((result2, match2))
                else:
                    best = result1 if result1.confidence_text_ocr >= result2.confidence_text_ocr else result2
                    best_match = match1 if best == result1 else match2
                    votes.append((best, best_match))
            else:
                if score1 < score2:
                    votes.append((result1, match1))
                elif score2 < score1:
                    votes.append((result2, match2))
                else:
                    best = result1 if result1.confidence_text_ocr >= result2.confidence_text_ocr else result2
                    best_match = match1 if best == result1 else match2
                    votes.append((best, best_match))

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
            text_position_algo = (10, annotated_image.shape[0] - 30)  # Внизу слева
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

if __name__ == "__main__":
    model_langs = ['ru']
    allowlist = '0123456789феспФЕСПeECc-*_.,'
    text_recognition = EasyOCRTextRecognition(model_langs, allowlist)

    codes = ["21353-21",
             "27353-27",
             "27353-21",
             "2135-27",
             "273-21"]

    ocr_selector = OCRResultSelector(codes)

    # Получаем два варианта результата OCR
    img_path = r"D:\я у мамы программист\Diplom\KernAI-backend-fastapi\temp\user1\party_fd0c5323-3525-4fb8-8d0c-2090ad0c444c\step5_rotate_image\process_image_3.png"
    img = Image.open(img_path)
    ocr_result_1, ocr_result_2 = text_recognition.recognize_text(img)

    # Выбираем наилучший вариант
    best_result = ocr_selector.select_best_text_ensemble(ocr_result_1, ocr_result_2)
    print(best_result)