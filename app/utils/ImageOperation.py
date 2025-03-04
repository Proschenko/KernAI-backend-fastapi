import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


class ImageOperation:
    def __init__(self):
        pass

    @staticmethod
    def process_image_white_corners(image_inner, save_folder_path=None):
        """
        Обрабатывает изображение, добавляя белые углы.

        Аргументы:
            image_inner: Исходное изображение.
            save_folder_path: Путь к папке для сохранения обработанного изображения (необязательно).

        Возвращает:
            Image: Обработанное изображение.
        """
        # Получаем размеры изображения
        width, height = image_inner.size

        # Вычисляем центр изображения
        center_x = width // 2
        center_y = height // 2

        # Вычисляем радиус изображения (половина короткой стороны)
        radius = max(width, height) // 2

        # Создаем новое изображение с белым фоном
        processed_image = Image.new(image_inner.mode, image_inner.size, (255, 255, 255))

        # Создаем маску для круга
        mask = Image.new('L', image_inner.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse([(center_x - radius, center_y - radius), (center_x + radius, center_y + radius)], fill=255)

        # Вставляем оригинальное изображение на обработанное изображение с использованием маски
        processed_image.paste(image_inner, mask=mask)

        # Сохраняем обработанное изображение, если указан путь к папке
        if save_folder_path:
            if not os.path.exists(save_folder_path):
                os.makedirs(save_folder_path)
            file_name = "processed_image"  # Вы можете настроить это базовое имя по мере необходимости
            files = os.listdir(save_folder_path)
            processed_image_path = os.path.join(save_folder_path, f"{file_name}_{len(files) + 1}.png")
            processed_image.save(processed_image_path)

        return processed_image

    @staticmethod
    def clahe_processing(image_inner, save_folder_path=None):
        """
        Применяет CLAHE к изображению для улучшения контраста.

        Аргументы:
            image_inner: Исходное изображение.
            save_folder_path: Путь к папке для сохранения обработанного изображения.

        Возвращает:
            Image: Обработанное изображение.
        """
        # Преобразование изображения в массив numpy
        image_np = np.array(image_inner)

        # Преобразование в градации серого
        if len(image_np.shape) == 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

        # Применение CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_image = clahe.apply(image_np)

        # Преобразование обратно в изображение PIL
        clahe_image_pil = Image.fromarray(clahe_image)

        if save_folder_path:
            if not os.path.exists(save_folder_path):
                os.makedirs(save_folder_path)
            file_name = "clahe_image"
            files = os.listdir(save_folder_path)
            clahe_image_path = os.path.join(save_folder_path, f"{file_name}_{len(files) + 1}.png")
            clahe_image_pil.save(clahe_image_path)

        return clahe_image_pil

    @staticmethod
    def plot_brightness_histogram(image_inner, show=False):
        """
        Строит гистограмму яркости изображения.

        Аргументы:
            image_inner: Исходное изображение.
            show: Флаг для отображения гистограммы (необязательно).

        Возвращает:
            int: Пик яркости.
        """
        # Преобразование в градации серого
        image_inner = image_inner.convert('L')

        # Преобразование изображения в массив numpy
        image_array = np.array(image_inner)

        # Вычисление гистограммы
        histogram, bin_edges = np.histogram(image_array, bins=256, range=(0, 256))

        # Нахождение пика гистограммы
        peak_brightness_inner = np.argmax(histogram[:250])

        # Отображение гистограммы
        if show:
            plt.figure(figsize=(10, 6))
            plt.plot(bin_edges[0:-1], histogram, color='black')
            plt.title('Гистограмма яркости')
            plt.xlabel('Яркость')
            plt.ylabel('Частота')
            plt.xlim([0, 256])
            plt.grid(True)
            plt.show()

        return peak_brightness_inner

    @staticmethod
    def process_image_white_by_brightness_peak(image_inner, peak_brightness_inner, tolerance, output_folder=None):
        """
        Обрабатывает изображение, заменяя пиксели в диапазоне яркости на белые.

        Аргументы:
            image_inner: Исходное изображение.
            peak_brightness_inner: Пик яркости.
            tolerance: Допуск для диапазона яркости.
            output_folder: Путь к папке для сохранения обработанного изображения (необязательно).

        Возвращает:
            Image: Обработанное изображение.
        """
        # Преобразование в градации серого
        image_inner = image_inner.convert('L')

        # Преобразование изображения в массив numpy
        image_array = np.array(image_inner)

        # Определение диапазона значений яркости для замены на белый
        lower_bound = peak_brightness_inner - tolerance
        upper_bound = peak_brightness_inner + tolerance

        # Создание маски для пикселей, которые будут заменены на белый
        mask = (image_array >= lower_bound) & (image_array <= upper_bound)

        # Замена пикселей в маске на белый (255)
        image_array[mask] = 255

        # Преобразование массива numpy обратно в изображение
        processed_image = Image.fromarray(image_array)

        # Сохранение обработанного изображения, если указан путь к папке
        if output_folder:
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            file_name = "processed_image"  # Вы можете настроить это базовое имя по мере необходимости
            processed_image_path = os.path.join(output_folder, f"{file_name}.png")
            processed_image.save(processed_image_path)

        return processed_image

    @staticmethod
    def process_cluster_image(image_inner, n_clusters=5, save_folder_path=None):
        """
        Применяет кластеризацию k-means к изображению.

        Аргументы:
            image_inner: Исходное изображение.
            n_clusters: Количество кластеров.
            save_folder_path: Путь к папке для сохранения обработанного изображения (необязательно).

        Возвращает:
            Image: Обработанное изображение.
        """
        # Преобразование изображения в массив пикселей
        image_np = np.array(image_inner)

        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            pixels = image_np.reshape(-1, 3)
        elif len(image_np.shape) == 2 or (len(image_np.shape) == 3 and image_np.shape[2] == 1):
            pixels = image_np.reshape(-1, 1)
        else:
            raise ValueError("Неподдерживаемая форма изображения")

        # Применение k-means для кластеризации
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(pixels)

        # Замена каждого пикселя на центр его кластера
        clustered_pixels = kmeans.cluster_centers_[kmeans.labels_]

        # Преобразование обратно в изображение
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            clustered_image = clustered_pixels.reshape(image_inner.size[1], image_inner.size[0], 3).astype(np.uint8)
        else:
            clustered_image = clustered_pixels.reshape(image_inner.size[1], image_inner.size[0]).astype(np.uint8)

        # Преобразование массива numpy обратно в изображение
        clustered_image = Image.fromarray(clustered_image)

        # Сохранение кластеризованного изображения, если указан путь к папке
        if save_folder_path:
            if not os.path.exists(save_folder_path):
                os.makedirs(save_folder_path)
            file_name = "clustered_image"  # Вы можете настроить это базовое имя по мере необходимости
            files = os.listdir(save_folder_path)
            clustered_image_path = os.path.join(save_folder_path, f"{file_name}_{len(files) + 1}.png")
            clustered_image.save(clustered_image_path)

        return clustered_image


if __name__ == "__main__":
    # Пример использования
    image_operation = ImageOperation()
    image = Image.open('path_to_your_image.jpg')

    # Пример использования метода process_image_white_corners
    processed_image_1 = image_operation.process_image_white_corners(image,
                                                                    save_folder_path='path_to_save_processed_image')
    processed_image_1.show()  # Показать обработанное изображение

    # Пример использования метода plot_brightness_histogram
    peak_brightness = image_operation.plot_brightness_histogram(image, show=True)

    # Пример использования метода process_image_white_by_brightness_peak
    processed_image_2 = image_operation.process_image_white_by_brightness_peak(image, peak_brightness, tolerance=30,
                                                                               output_folder='path_to_save_processed_image')
    processed_image_2.show()  # Показать обработанное изображение
