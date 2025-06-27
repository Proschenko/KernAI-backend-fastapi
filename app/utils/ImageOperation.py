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

    # @staticmethod
    # def process_cluster_image(image_inner, n_clusters=5, save_folder_path=None):
    #     """
    #     Применяет кластеризацию k-means к изображению.

    #     Аргументы:
    #         image_inner: Исходное изображение.
    #         n_clusters: Количество кластеров.
    #         save_folder_path: Путь к папке для сохранения обработанного изображения (необязательно).

    #     Возвращает:
    #         Image: Обработанное изображение.
    #     """
    #     # Преобразование изображения в массив пикселей
    #     image_np = np.array(image_inner)

    #     if len(image_np.shape) == 3 and image_np.shape[2] == 3:
    #         pixels = image_np.reshape(-1, 3)
    #     elif len(image_np.shape) == 2 or (len(image_np.shape) == 3 and image_np.shape[2] == 1):
    #         pixels = image_np.reshape(-1, 1)
    #     else:
    #         raise ValueError("Неподдерживаемая форма изображения")

    #     # Применение k-means для кластеризации
    #     kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(pixels)

    #     # Замена каждого пикселя на центр его кластера
    #     clustered_pixels = kmeans.cluster_centers_[kmeans.labels_]

    #     # Преобразование обратно в изображение
    #     if len(image_np.shape) == 3 and image_np.shape[2] == 3:
    #         clustered_image = clustered_pixels.reshape(image_inner.size[1], image_inner.size[0], 3).astype(np.uint8)
    #     else:
    #         clustered_image = clustered_pixels.reshape(image_inner.size[1], image_inner.size[0]).astype(np.uint8)

    #     # Преобразование массива numpy обратно в изображение
    #     clustered_image = Image.fromarray(clustered_image)

    #     # Сохранение кластеризованного изображения, если указан путь к папке
    #     if save_folder_path:
    #         if not os.path.exists(save_folder_path):
    #             os.makedirs(save_folder_path)
    #         file_name = "clustered_image"  # Вы можете настроить это базовое имя по мере необходимости
    #         files = os.listdir(save_folder_path)
    #         clustered_image_path = os.path.join(save_folder_path, f"{file_name}_{len(files) + 1}.png")
    #         clustered_image.save(clustered_image_path)

    #     return clustered_image


    @staticmethod
    def process_cluster_image(image_inner, n_clusters=5, save_folder_path=None, cluster_description=None):
        """
        Применяет кластеризацию k-means к изображению и при необходимости сохраняет результат.

        Аргументы:
            image_inner: Исходное изображение (PIL.Image).
            n_clusters: Количество кластеров.
            save_folder_path: Путь к папке для сохранения обработанного изображения (по умолчанию None).
            cluster_description: Имя подпапки для сохранения отдельных кластеров (по умолчанию None).

        Возвращает:
            Image: Кластеризованное изображение.
        """
        image_np = np.array(image_inner)

        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            pixels = image_np.reshape(-1, 3)
        elif len(image_np.shape) == 2 or (len(image_np.shape) == 3 and image_np.shape[2] == 1):
            pixels = image_np.reshape(-1, 1)
        else:
            raise ValueError("Неподдерживаемая форма изображения")

        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(pixels)
        clustered_pixels = kmeans.cluster_centers_[kmeans.labels_]

        if len(image_np.shape) == 3 and image_np.shape[2] == 3:
            clustered_image_np = clustered_pixels.reshape(image_inner.size[1], image_inner.size[0], 3).astype(np.uint8)
        else:
            clustered_image_np = clustered_pixels.reshape(image_inner.size[1], image_inner.size[0]).astype(np.uint8)

        clustered_image = Image.fromarray(clustered_image_np)

        if save_folder_path:
            os.makedirs(save_folder_path, exist_ok=True)
            image_index = len([f for f in os.listdir(save_folder_path) if f.endswith('.png')]) + 1
            clustered_image_path = os.path.join(save_folder_path, f"clustered_image_{image_index}.png")
            clustered_image.save(clustered_image_path)

            if cluster_description:
                cluster_folder = os.path.join(save_folder_path, cluster_description)
                os.makedirs(cluster_folder, exist_ok=True)

                labels = kmeans.labels_.reshape(image_inner.size[1], image_inner.size[0])

                for cluster_id in range(n_clusters):
                    mask = (labels == cluster_id)

                    if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                        single_cluster_img = np.full_like(clustered_image_np, 255, dtype=np.uint8)
                        single_cluster_img[mask] = clustered_image_np[mask]
                    else:
                        single_cluster_img = np.full_like(clustered_image_np, 255, dtype=np.uint8)
                        single_cluster_img[mask] = clustered_image_np[mask]

                    single_image = Image.fromarray(single_cluster_img)
                    filename = f"clustered_image_{image_index}_cluster_{cluster_id}.png"
                    single_image.save(os.path.join(cluster_folder, filename))

                    # Построение и сохранение гистограммы с сортировкой по убыванию
                    cluster_colors = kmeans.cluster_centers_.astype(int)
                    counts = np.bincount(kmeans.labels_, minlength=n_clusters)

                    # Создание структуры с сортировкой
                    cluster_data = [
                        (i, count, cluster_colors[i]) for i, count in enumerate(counts)
                    ]
                    cluster_data.sort(key=lambda x: x[1], reverse=True)

                    sorted_ids = [item[0] for item in cluster_data]
                    sorted_counts = [item[1] for item in cluster_data]
                    sorted_colors = [tuple(color / 255) if len(color) == 3 else (color[0] / 255,) * 3 for _, _, color in cluster_data]

                    plt.figure(figsize=(8, 6))
                    plt.bar(range(n_clusters), sorted_counts, color=sorted_colors)
                    plt.xlabel("Номер кластера")
                    plt.ylabel("Количество пикселей")
                    plt.title(f"Распределение пикселей по кластерам (изображение {image_index})")
                    plt.xticks(range(n_clusters), [f"{cluster_id}" for cluster_id in sorted_ids])

                    # Подписи над столбцами
                    for i, count in enumerate(sorted_counts):
                        plt.text(i, count + max(sorted_counts) * 0.01, str(count), ha='center', va='bottom', fontsize=9)

                    hist_path = os.path.join(cluster_folder, f"clustered_image_{image_index}_histogram.png")
                    plt.tight_layout()
                    plt.savefig(hist_path)
                    plt.close()

        return clustered_image


if __name__ == "__main__":
    from PIL import Image

    image_path = r"D:\я у мамы программист\Diplom\KernAI-backend-fastapi\temp\user1\party_fd0c5323-3525-4fb8-8d0c-2090ad0c444c\step3_clahe\clahe_image_7.png"
    image = Image.open(image_path).convert("RGB")

    cluster_description = "cluster_description"
    save_folder_path = r"D:\я у мамы программист\Diplom\KernAI-backend-fastapi\temp\user1\step4_cluster_image"
    ImageOperation().process_cluster_image(image_inner=image, save_folder_path=save_folder_path, cluster_description=cluster_description)
    