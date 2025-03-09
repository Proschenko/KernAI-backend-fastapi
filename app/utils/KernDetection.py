import os
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2


class KernDetection:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    @staticmethod
    def __extract_predictions_obb(results):
        """
        Извлекает ориентированные ограничивающие рамки, уверенности, классы и имена классов из результатов предсказания.

        Аргументы:
            results: Результаты предсказания модели YOLO.

        Возвращает:
            tuple: Кортеж, содержащий списки ориентированных ограничивающих рамок, уверенностей, классов и имен классов.
        """
        # Извлечение ориентированных ограничивающих рамок
        bboxes_ = results[0].obb.xyxyxyxy.tolist()
        bboxes = list(map(lambda x: list(map(lambda y: list(map(lambda z: float(z), y)), x)), bboxes_))

        # Извлечение уверенностей
        confs_ = results[0].obb.conf.tolist()
        confs = list(map(lambda x: int(x * 100), confs_))

        # Извлечение классов
        classes_ = results[0].obb.cls.tolist()
        classes = list(map(lambda x: int(x), classes_))

        # Извлечение имен классов
        cls_dict = results[0].names
        class_names = list(map(lambda x: cls_dict[x], classes))

        return bboxes, confs, classes, class_names

    @staticmethod
    def __get_bounding_box(bbox, scale=0.05):
        x_coords = [bbox[0][0], bbox[1][0], bbox[2][0], bbox[3][0]]
        y_coords = [bbox[0][1], bbox[1][1], bbox[2][1], bbox[3][1]]

        x_min = min(x_coords)
        y_min = min(y_coords)
        x_max = max(x_coords)
        y_max = max(y_coords)

        # Вычисление ширины и высоты ограничивающей рамки
        width = x_max - x_min
        height = y_max - y_min

        # Вычисление коэффициента масштабирования
        scale_width = width * scale
        scale_height = height * scale

        # Расширение ограничивающей рамки
        x_min = int(x_min - scale_width)
        y_min = int(y_min - scale_height)
        x_max = int(x_max + scale_width)
        y_max = int(y_max + scale_height)

        return x_min, y_min, x_max, y_max

    @staticmethod
    def __calculate_rotation_angle(bbox):
        """
        Вычисляет угол поворота изображения так, чтобы длинная сторона ограничивающей рамки была параллельна горизонту.

        Аргументы:
            bbox: Список из 4 точек (каждая точка - список из 2 координат).

        Возвращает:
            float: Угол поворота изображения.
        """
        # Вычисление векторов сторон ограничивающей рамки
        vec1 = np.array(bbox[1]) - np.array(bbox[0])
        vec2 = np.array(bbox[2]) - np.array(bbox[1])

        # Вычисление длин векторов
        len1 = np.linalg.norm(vec1)
        len2 = np.linalg.norm(vec2)

        # Определение длинной стороны
        if len1 > len2:
            longest_vec = vec1
        else:
            longest_vec = vec2

        # Вычисление угла длинной стороны с горизонтальной осью
        angle = np.arctan2(longest_vec[1], longest_vec[0]) * 180 / np.pi

        return angle

    def visualize_predictions_obb(self, image, conf_threshold=50):
        """
        Визуализирует предсказания модели с ориентированными ограничивающими рамками.

        Аргументы:
            image: Изображение для визуализации.
            conf_threshold: Порог уверенности для отображения предсказаний.
        """
        # Выполнение инференса
        results = self.model(image)

        # Извлечение предсказаний
        bboxes, confs, classes, class_names = self.__extract_predictions_obb(results)

        # Создание фигуры и оси
        fig, ax = plt.subplots(1)

        # Отображение изображения
        ax.imshow(image)

        # Рисование ориентированных ограничивающих рамок и меток
        for bbox, conf, cls_name in zip(bboxes, confs, class_names):
            if conf >= conf_threshold:
                label = f"{cls_name} {conf}%"

                # Создание многоугольника из углов
                polygon = patches.Polygon(bbox, closed=True, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(polygon)
                ax.text(bbox[0][0], bbox[0][1], label, color='b', fontsize=10, ha='center', va='center')

        # Показ графика
        plt.axis('off')
        plt.show()

    def crop_kern_with_obb_predictions(self, image, save_folder_path=None, conf_threshold=90):
        """
        Обрезает объекты на изображении на основе предсказаний модели с ориентированными ограничивающими рамками.

        Аргументы:
            image: Изображение для обрезки.
            save_folder_path: Путь к папке для сохранения обрезанных изображений.
            conf_threshold: Порог уверенности для обрезки объектов.

        Возвращает:
            list: Список обрезанных изображений.
        """
        # Проверка ориентации изображения
        if image.width < image.height:
            # Поворот изображения на 90 градусов влево
            image = image.rotate(90, expand=True)

        # Выполнение инференса
        results = self.model(image, verbose=False)

        # Извлечение предсказаний
        bboxes, confs, classes, class_names = self.__extract_predictions_obb(results)

        cropped_images = []

        # Обработка каждой ограничивающей рамки
        for i, (bbox, conf, cls_name) in enumerate(zip(bboxes, confs, class_names)):
            if conf >= conf_threshold:
                # Извлечение координат ограничивающей рамки
                x_min, y_min, x_max, y_max = self.__get_bounding_box(bbox=bbox)

                # Обрезка объекта из изображения
                cropped_image = image.crop((x_min, y_min, x_max, y_max))
                cropped_images.append(cropped_image)

                # Сохранение обрезанного изображения, если указан путь к папке
                if save_folder_path:
                    if not os.path.exists(save_folder_path):
                        os.makedirs(save_folder_path)
                    cropped_image_path = os.path.join(save_folder_path, f"{cls_name}_{i + 1}.png")
                    cropped_image.save(cropped_image_path)

        return cropped_images

    def image_rotated_with_obb_predictions(self, image, save_folder_path=None, size_inner=360):
        """
        Поворачивает изображение на основе предсказаний модели с ориентированными ограничивающими рамками.

        Аргументы:
            image: Изображение для поворота.
            save_folder_path: Путь к папке для сохранения повернутого изображения.
            size_inner: Внутренний размер изображения.

        Возвращает:
            numpy.ndarray: Повернутое изображение.
        """
        image = image.resize((size_inner, size_inner))
        # Преобразование изображения PIL в формат OpenCV
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Выполнение инференса
        results = self.model(image, verbose=False)

        # Извлечение предсказаний
        bboxes, confs, classes, class_names = self.__extract_predictions_obb(results)

        if not bboxes:
            print("Предсказания не найдены")
            return image_cv

        # Нахождение предсказания с наибольшей уверенностью
        max_conf_index = np.argmax(confs)
        best_bbox = bboxes[max_conf_index]

        # Вычисление угла поворота изображения
        angle = self.__calculate_rotation_angle(best_bbox)

        # Создание большого белого изображения
        larger_image = np.ones((500, 500, 3), dtype=np.uint8) * 255
        h, w = image_cv.shape[:2]
        x_offset = (500 - w) // 2
        y_offset = (500 - h) // 2
        larger_image[y_offset:y_offset + h, x_offset:x_offset + w] = image_cv

        # Поворот большого изображения
        M = cv2.getRotationMatrix2D((250, 250), angle, 1.0)
        rotated_image = cv2.warpAffine(larger_image, M, (500, 500))

        # Обрезка повернутого изображения обратно до 360x360
        cropped_image = rotated_image[70:430, 70:430]

        if save_folder_path:
            if not os.path.exists(save_folder_path):
                os.makedirs(save_folder_path)

            # Получение списка файлов в папке
            files = os.listdir(save_folder_path)

            # Фильтрация файлов, соответствующих базовому имени и расширению
            base_name = "process_image"  # Вы можете настроить это базовое имя по мере необходимости
            ext = ".png"  # Вы можете настроить расширение по мере необходимости

            # Создание уникального имени файла
            output_path = os.path.join(save_folder_path, f"{base_name}_{len(files) + 1}{ext}")
            cv2.imwrite(output_path, cropped_image)

        return cropped_image


if __name__ == "__main__":
    pass
