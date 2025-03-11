from abc import ABC, abstractmethod
from PIL import Image
import cv2
import numpy as np
import easyocr
from typing import List, Tuple
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


class EasyOCRTextRecognition(TextRecognitionBase):
    def __init__(self, model_langs: List[str], allowlist: str):
        self.reader = easyocr.Reader(model_langs, gpu=False)
        self.allowlist = allowlist

    def recognize_text(self, image: Image.Image) -> Tuple[OCRResult, OCRResult]:
        """Распознает текст и возвращает два возможных результата."""
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Первый проход OCR
        ocr_result = self.reader.readtext(image_cv, allowlist=self.allowlist)
        words = [result[1] for result in ocr_result]
        confidence = sum(float(result[2]) for result in ocr_result) if ocr_result else 0.0  # Приводим к float

        # Второй проход (развернутое изображение)
        rotated_image_180_cv = cv2.rotate(image_cv, cv2.ROTATE_180)
        ocr_result_180 = self.reader.readtext(rotated_image_180_cv, allowlist=self.allowlist)
        words_180 = [result[1] for result in ocr_result_180]
        confidence_180 = sum(float(result[2]) for result in ocr_result_180) if ocr_result_180 else 0.0

        # Создаем корректные объекты Pydantic
        return (
            OCRResult(text_ocr=" ".join(words), words_ocr=words, confidence_ocr=confidence),
            OCRResult(text_ocr=" ".join(words_180), words_ocr=words_180, confidence_ocr=confidence_180)
        )


class OCRResultSelector:
    def __init__(self, reference_data: List[str]):
        self.reference_data = reference_data
        self.algorithm_name = 'levenshtein_distance'

    def select_best_text(self, result1: OCRResult, result2: OCRResult) -> OCRResultSelectorAlgotitm:
        """Выбирает лучший текст на основе сравнения с эталоном или уверенности модели."""  

        if not self.reference_data:
            # Если reference_data пуст, выбираем по уверенности
            best_result = result1 if result1.confidence_ocr >= result2.confidence_ocr else result2
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
            best_result = result1 if result1.confidence_ocr >= result2.confidence_ocr else result2
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



# class TextRecognition1:
#     def __init__(self, model_langs, allowlist):
#         """
#         Инициализация класса TextRecognition.

#         Аргументы:
#             model_langs: Список языков для модели OCR.
#             allowlist: Список разрешенных символов для распознавания.
#         """
#         self.reader = easyocr.Reader(model_langs, gpu=False)
#         self.allowlist = allowlist

#     def recognize_text(self, image, save_folder_path=None, return_both_results=False):
#         """
#         Распознает текст на изображении и возвращает результаты.

#         Аргументы:
#             image: Изображение для распознавания текста.
#             save_folder_path: Путь к папке для сохранения результатов.
#             return_both_results: Флаг для возврата результатов для обоих направлений поворота.

#         Возвращает:
#             dict или tuple: Результаты распознавания текста.
#         """
#         # Преобразование изображения PIL в формат OpenCV
#         image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

#         # Выполнение OCR на оригинальном изображении
#         ocr_result = self.reader.readtext(image_cv, allowlist=self.allowlist)
#         ocr_confidence = sum([result[2] for result in ocr_result])

#         # Поворот изображения на 180 градусов
#         rotated_image_180 = cv2.rotate(image_cv, cv2.ROTATE_180)

#         # Выполнение OCR на повернутом изображении
#         ocr_result_180 = self.reader.readtext(rotated_image_180, allowlist=self.allowlist)
#         ocr_confidence_180 = sum([result[2] for result in ocr_result_180])

#         if return_both_results:
#             if save_folder_path:
#                 _, output_path = self.draw_predictions([(image_cv, ocr_result), (rotated_image_180, ocr_result_180)],
#                                                        save_folder_path,
#                                                        return_both_results)
#             return {"image_cv": image_cv,
#                     "ocr_result": ' '.join([result[1] for result in ocr_result]),
#                     "ocr_confidence": ocr_confidence,
#                     "rotated_image_180": rotated_image_180,
#                     "ocr_result_180": ' '.join([result[1] for result in ocr_result_180]),
#                     "ocr_confidence_180": ocr_confidence_180,
#                     "path": output_path}
#         else:
#             # Сравнение результатов OCR и выбор лучшего
#             if ocr_confidence > ocr_confidence_180:
#                 best_image = image_cv
#                 best_text = ' '.join([result[1] for result in ocr_result])
#                 best_predictions = ocr_result
#             else:
#                 best_image = rotated_image_180
#                 best_text = ' '.join([result[1] for result in ocr_result_180])
#                 best_predictions = ocr_result_180
#             if save_folder_path:
#                 self.draw_predictions([best_image, best_predictions], save_folder_path, return_both_results)
#             return best_image, best_text

#     @staticmethod
#     def draw_predictions(data_prediction, save_folder_path, return_both_results=False):
#         """
#         Рисует предсказания на изображении и сохраняет результат.

#         Аргументы:
#             data_prediction: Данные предсказаний.
#             save_folder_path: Путь к папке для сохранения результатов.
#             return_both_results: Флаг для возврата результатов для обоих направлений поворота.

#         Возвращает:
#             tuple: Изображение с нарисованными предсказаниями и путь к сохраненному файлу.
#         """
#         if return_both_results:
#             fig, axs = plt.subplots(1, 2, figsize=(12, 6))
#             for i, (image, predictions) in enumerate(data_prediction):
#                 for (bbox, text, prob) in predictions:
#                     (top_left, top_right, bottom_right, bottom_left) = bbox
#                     top_left = tuple(map(int, top_left))
#                     bottom_right = tuple(map(int, bottom_right))
#                     cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
#                     cv2.putText(image, text + " " + str(round(prob, 3)), (top_left[0], top_left[1] - 10),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#                 axs[i].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#                 axs[i].axis('off')

#             # Убедиться, что папка вывода существует
#             if not os.path.exists(save_folder_path):
#                 os.makedirs(save_folder_path)
#             files = os.listdir(save_folder_path)
#             # Сохранить изображение в папку вывода
#             output_path = os.path.join(save_folder_path, f'recognition_text_image_{len(files) + 1}.png')
#             plt.savefig(output_path)
#             plt.close(fig)
#         else:
#             for (bbox, text, prob) in data_prediction:
#                 (top_left, top_right, bottom_right, bottom_left) = bbox
#                 top_left = tuple(map(int, top_left))
#                 bottom_right = tuple(map(int, bottom_right))
#                 cv2.rectangle(data_prediction, top_left, bottom_right, (0, 255, 0), 2)
#                 cv2.putText(data_prediction, text + " " + str(round(prob, 3)), (top_left[0], top_left[1] - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5,
#                             (0, 255, 0), 2)

#             # Убедиться, что папка вывода существует
#             if not os.path.exists(save_folder_path):
#                 os.makedirs(save_folder_path)
#             files = os.listdir(save_folder_path)
#             # Сохранить изображение в папку вывода
#             output_path = os.path.join(save_folder_path, f'recognition_text_image_{len(files) + 1}.png')
#             cv2.imwrite(output_path, data_prediction)

#         return data_prediction, output_path

#     def levenshtein_distance(self, s1, s2):
#         """
#         Вычисляет расстояние Левенштейна между двумя строками.

#         Аргументы:
#             s1: Первая строка.
#             s2: Вторая строка.

#         Возвращает:
#             int: Расстояние Левенштейна.
#         """
#         if len(s1) < len(s2):
#             return self.levenshtein_distance(s2, s1)

#         if len(s2) == 0:
#             return len(s1)

#         previous_row = np.arange(len(s2) + 1)
#         for i, c1 in enumerate(s1):
#             current_row = previous_row + 1
#             current_row[1:] = np.minimum(
#                 current_row[1:],
#                 np.add(previous_row[:-1], [c1 != c2 for c2 in s2])
#             )
#             current_row[1:] = np.minimum(
#                 current_row[1:],
#                 current_row[0:-1] + 1
#             )
#             previous_row = current_row

#         return previous_row[-1]

#     def __find_n_best_matches(self, ocr_result, excel_data, n=3):
#         """
#         Находит n лучших совпадений для результата OCR в данных Excel.

#         Аргументы:
#             ocr_result: Результат OCR.
#             excel_data: Данные из Excel.
#             n: Количество лучших совпадений для возврата.

#         Возвращает:
#             list: Список лучших совпадений.
#         """
#         matches = []
#         for data in excel_data:
#             distance = self.levenshtein_distance(ocr_result, data[0])
#             matches.append((data[0], distance))

#         matches.sort(key=lambda x: x[1])  # Сортировка по расстоянию
#         return matches[:n]

#     @staticmethod
#     def format_best_matches(matches):
#         """
#         Форматирует лучшие совпадения в строку.

#         Аргументы:
#             matches: Список лучших совпадений.

#         Возвращает:
#             str: Отформатированная строка лучших совпадений.
#         """
#         formatted_matches = []
#         for string, distance in matches:
#             formatted_matches.append(f"{string} {int(distance)},")
#         return " ".join(formatted_matches)

#     def process_ocr_results(self, json_data, excel_data, n=3):
#         """
#         Обрабатывает результаты OCR и находит лучшие совпадения в данных Excel.

#         Аргументы:
#             json_data: Данные JSON с результатами OCR.
#             excel_data: Данные из Excel.
#             n: Количество лучших совпадений для возврата.

#         Возвращает:
#             dict: Лучшие совпадения для результатов OCR.
#         """
#         ocr_result = json_data["ocr_reuslt"]
#         ocr_result_180 = json_data["ocr_result_180"]

#         excel_data = excel_data.replace()

#         best_matches = {
#             "best_matches_ocr_reuslt": self.format_best_matches(self.__find_n_best_matches(ocr_result, excel_data, n)),
#             "best_matches_ocr_result_180": self.format_best_matches(
#                 self.__find_n_best_matches(ocr_result_180, excel_data, n))
#         }

#         return best_matches


# def __test_image_recognition():
#     """
#     Тестовая функция для распознавания текста на изображении.
#     """
#     # Пример использования
#     model_langs = ['ru']
#     allowlist = '0123456789фесп-*_'
#     text_recognition = EasyOCRTextRecognition(model_langs, allowlist)

#     # Загрузка изображения
#     image = Image.open(r'D:\Diplom\datasets\3_cropped_images\resized_360x360\0001_kern_6.png')

#     # Распознавание текста
#     best_image, best_text = text_recognition.recognize_text(image)

#     # Показать лучшее изображение
#     best_image_pil = Image.fromarray(cv2.cvtColor(best_image, cv2.COLOR_BGR2RGB))
#     best_image_pil.show()

#     # Вывести распознанный текст
#     print("Распознанный текст:", best_text)


if __name__ == "__main__":
    model_langs = ['ru']
    allowlist = '0123456789феспФЕСПeECc-*_.,'
    text_recognizer = EasyOCRTextRecognition(model_langs=model_langs, allowlist=allowlist)

    path = "D:\\я у мамы программист\\Diplom\\datasets\\1 source images\\0007.jpg"
    image = Image.open(path)
    # Распознаем текст
    ocr_result_1, ocr_result_2 = text_recognizer.recognize_text(image)

    # Загружаем эталонные данные (например, из Excel)
    reference_data = ["SAMPLE123", "CORE456", "TEST789"]  # Пример эталонов

    # Выбираем лучший результат
    selector = OCRResultSelector(reference_data)
    best_result = selector.select_best_text(ocr_result_1, ocr_result_2)

    print(f"Лучший результат: {best_result.text}")
