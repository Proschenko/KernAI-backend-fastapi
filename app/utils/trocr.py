import time

from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch


def recognize_russian_text(image_path: str) -> str:
    """
    Распознает русский текст на изображении с помощью модели TrOCR.

    Args:
        image_path (str): Путь к изображению для распознавания

    Returns:
        str: Распознанный текст
    """
    # Загрузка изображения
    image = Image.open(image_path).convert("RGB")

    # Выберите нужную модель, раскомментировав одну из строк:
    # ------------------------------------------------------
    # 1. Официальная модель Microsoft (печатный текст, не рекомендуется для рукописного)
    # model_name = "microsoft/trocr-base-printed"

    # 2. Официальная большая модель Microsoft (печатный текст, не рекомендуется для рукописного)
    # model_name = "microsoft/trocr-large-printed"

    # 3. Модель для русского языка от сообщества (печатный текст, но может работать с рукописным)
    # model_name = "IlyaGusev/trocr-ru-printed"
    model_name = "kazars24/trocr-base-handwritten-ru"

    # 4. Маленькая версия для русского языка (печатный текст, но может работать с рукописным)
    # model_name = "IlyaGusev/trocr-ru-printed-simple"

    # 5. Модель для рукописного текста (неофициальная, может быть менее точной)
    # model_name = "IlyaGusev/trocr-ru-handwritten"

    # Загрузка процессора и модели
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForVision2Seq.from_pretrained(model_name)

    # Предобработка изображения
    inputs = processor(images=image, return_tensors="pt").pixel_values

    # Генерация текста с параметрами
    generated_ids = model.generate(
        inputs,
        max_length=64,  # Максимальная длина текста
        num_beams=8  # Параметр поиска для лучшего качества
    )

    # Декодирование результата
    generated_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    return generated_text.strip()


if __name__ == "__main__":
    image_path = r"D:\я у мамы программист\Diplom\KernAI-backend\tmp_data\tmp\step5_rotate_image\process_image_17.png"
    time_start = time.time()
    text = recognize_russian_text(image_path)
    time_end = time.time()
    print(text)
    print(time_end - time_start)
