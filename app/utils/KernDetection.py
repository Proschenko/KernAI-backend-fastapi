import os
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from tqdm import tqdm
import cv2


class YOLODetection:
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
    def __get_bounding_box(bbox, class_name, scale=0.05):
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
        if class_name == "Text":
            scale_width = width * scale * 4 # Текст иногда обрезается, поэтому scale увеличиваем
        else:
            scale_width = width * scale
        scale_height = height * scale

        # Расширение ограничивающей рамки
        x_min = int(x_min - scale_width)
        y_min = int(y_min - scale_height)
        x_max = int(x_max + scale_width)
        y_max = int(y_max + scale_height)

        return x_min, y_min, x_max, y_max

    def filter_overlapping_bboxes(self, image, conf_threshold=50, iou_threshold=0.5, visualize=False):
        results = self.model(image)
        bboxes, confs, classes, class_names = self.__extract_predictions_obb(results)

        keep = [True] * len(bboxes)
        overlapped_indices = set()

        def compute_iou(poly1, poly2):
            """Вычисление IOU двух ориентированных bbox"""
            poly1_np = np.array(poly1, dtype=np.int32)
            poly2_np = np.array(poly2, dtype=np.int32)

            img = np.zeros((1000, 1000), dtype=np.uint8)
            cv2.fillPoly(img, [poly1_np], 1)
            mask1 = img.copy()
            img.fill(0)
            cv2.fillPoly(img, [poly2_np], 1)
            mask2 = img.copy()

            intersection = np.logical_and(mask1, mask2).sum()
            union = np.logical_or(mask1, mask2).sum()

            return intersection / union if union != 0 else 0

        for i in range(len(bboxes)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(bboxes)):
                if not keep[j]:
                    continue
                iou = compute_iou(bboxes[i], bboxes[j])
                if iou > iou_threshold:
                    if confs[i] >= confs[j]:
                        keep[j] = False
                        overlapped_indices.add(j)
                    else:
                        keep[i] = False
                        overlapped_indices.add(i)

        # Отфильтрованные и перекрытые
        filtered_bboxes = [bboxes[i] for i in range(len(bboxes)) if keep[i] and confs[i] >= conf_threshold]
        filtered_confs = [confs[i] for i in range(len(bboxes)) if keep[i] and confs[i] >= conf_threshold]
        filtered_classes = [classes[i] for i in range(len(bboxes)) if keep[i] and confs[i] >= conf_threshold]
        filtered_class_names = [class_names[i] for i in range(len(bboxes)) if keep[i] and confs[i] >= conf_threshold]

        overlapped_bboxes = [bboxes[i] for i in overlapped_indices]

        if visualize:
            fig, ax = plt.subplots(1)
            ax.imshow(image)
            for bbox in filtered_bboxes:
                polygon = patches.Polygon(bbox, closed=True, linewidth=2, edgecolor='g', facecolor='none')
                ax.add_patch(polygon)
            for bbox in overlapped_bboxes:
                polygon = patches.Polygon(bbox, closed=True, linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(polygon)
            plt.axis('off')
            plt.title("Green: Kept | Red: Overlapped Removed")
            plt.show()

        return filtered_bboxes, filtered_confs, filtered_classes, filtered_class_names, overlapped_bboxes

    @staticmethod
    def flip_bbox_180(bbox, image_width, image_height):
        # x_coords = [bbox[0][0], bbox[1][0], bbox[2][0], bbox[3][0]]
        # y_coords = [bbox[0][1], bbox[1][1], bbox[2][1], bbox[3][1]]

        # x_min = min(x_coords)
        # y_min = min(y_coords)
        # x_max = max(x_coords)
        # y_max = max(y_coords)

        # new_x_min = image_width - x_max
        # new_y_min = image_height - y_max
        # new_x_max = image_width - x_min
        # new_y_max = image_height - y_min

        flipped_bbox = []
        for x, y in bbox:
            new_x = image_width - x
            new_y = image_height - y
            flipped_bbox.append([new_x, new_y])

        return flipped_bbox

    @staticmethod
    def sort_bboxes_top_to_bottom(crop_images, bboxes, y_threshold=20, image = None):
        """
            Сортировка найденного текста на изображении керна по центрам bbox
        """
        def get_y_center(bbox):
            y_coords = [point[1] for point in bbox]
            return sum(y_coords) / len(y_coords)
        
        def get_x_min(bbox):
            return min(point[0] for point in bbox)
        
        images_with_bbox = [(crop_image, bbox) for crop_image, bbox in zip(crop_images, bboxes)]
        sorted_by_y = sorted(images_with_bbox, key=lambda img_bb: get_y_center(img_bb[1]))
        rows = []
        current_row = []

        for crop_images, bbox in sorted_by_y:
            if not current_row:
                current_row.append((crop_images, bbox))
            else:
                avg_y_prev = get_y_center(current_row[-1][1])
                avg_y_current = get_y_center(bbox)
                if abs(avg_y_current - avg_y_prev) <= y_threshold:
                    current_row.append((crop_images, bbox))
                else:
                    rows.append(current_row)
                    current_row = [(crop_images, bbox)]
        if current_row:
            rows.append(current_row)

        sorted_images_with_bboxes = []
        for row in rows:
            sorted_row = sorted(row, key=lambda img_bb: get_x_min(img_bb[1]))
            sorted_images_with_bboxes.append(sorted_row)
        
        if image:
            # Создание фигуры и оси
            fig, ax = plt.subplots(1)

            # Отображение изображения
            ax.imshow(image)
            sort_crop_bboxes = [[item[1] for item in crop_image_with_bbox] for crop_image_with_bbox in sorted_images_with_bboxes]
            # print(sort_crop_bboxes) 
            # Рисование ориентированных ограничивающих рамок и меток
            for i, bbox in enumerate(sort_crop_bboxes):
                label = f"text {i}"
                bbox = bbox[0]
                # Создание многоугольника из углов
                polygon = patches.Polygon(bbox, closed=True, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(polygon)
                ax.text(bbox[0][0], bbox[0][1], label, color='b', fontsize=10, ha='center', va='center')

            # Показ графика
            plt.axis('off')
            plt.show()

        return sorted_images_with_bboxes

    @staticmethod
    def __calculate_longest_vector(bbox):
        # Вычисление векторов сторон ограничивающей рамки
        vec1 = np.array(bbox[1]) - np.array(bbox[0])
        vec2 = np.array(bbox[2]) - np.array(bbox[1])

        # Вычисление длин векторов
        len1 = np.linalg.norm(vec1)
        len2 = np.linalg.norm(vec2)

        # Определение длинной стороны
        if len1 > len2:
            longest_vec = vec1
            longest_len = len1
        else:
            longest_vec = vec2
            longest_len = len2

        return longest_vec, longest_len

    def __calculate_rotation_angle(self, bbox):
        """
        Вычисляет угол поворота изображения так, чтобы длинная сторона ограничивающей рамки была параллельна горизонту.

        Аргументы:
            bbox: Список из 4 точек (каждая точка - список из 2 координат).

        Возвращает:
            float: Угол поворота изображения.
        """
        longest_vec, _ = self.__calculate_longest_vector(bbox)

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

    def crop_object_with_obb_predictions(self, image, save_folder_path=None, conf_threshold=90, filter_overlapping_bboxes=True):
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

        if filter_overlapping_bboxes:
            bboxes, confs, classes, class_names, _ = self.filter_overlapping_bboxes(image, conf_threshold=conf_threshold, iou_threshold=0.3, visualize=False)
        else:
            # Выполнение инференса
            results = self.model(image, verbose=False)
            # Извлечение предсказаний
            bboxes, confs, classes, class_names = self.__extract_predictions_obb(results)

        cropped_images = []
        cropped_bboxes = []
        # Обработка каждой ограничивающей рамки
        for i, (bbox, conf, cls_name) in enumerate(zip(bboxes, confs, class_names)):
            if conf >= conf_threshold:
                # Извлечение координат ограничивающей рамки
                x_min, y_min, x_max, y_max = self.__get_bounding_box(bbox=bbox, class_name=cls_name)

                # Обрезка объекта из изображения
                cropped_image = image.crop((x_min, y_min, x_max, y_max))
                cropped_images.append(cropped_image)
                cropped_bboxes.append(bbox)
                # Сохранение обрезанного изображения, если указан путь к папке
                if save_folder_path:
                    if not os.path.exists(save_folder_path):
                        os.makedirs(save_folder_path)
                    # Получение списка файлов в папке
                    files = os.listdir(save_folder_path)
                    cropped_image_path = os.path.join(save_folder_path, f"{cls_name}_{len(files) + 1}.png")
                    cropped_image.save(cropped_image_path)

        return cropped_images, cropped_bboxes

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
            self.save_image_in_folder(save_folder_path, image_cv)
            return image_cv

        # Нахождение предсказания с наибольшей шириной
        max_longest_len = 0 
        for index, bbox in enumerate(bboxes):
            _, longest_len = self.__calculate_longest_vector(bbox)
            if longest_len > max_longest_len:
                max_longest_len = longest_len
                best_index = index
        best_bbox = bboxes[best_index]

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

        self.save_image_in_folder(save_folder_path, cropped_image)

        return cropped_image

    def save_image_in_folder(self, save_folder_path, image):
        if save_folder_path:
                if not os.path.exists(save_folder_path):
                    os.makedirs(save_folder_path)

                # Получение списка файлов в папке
                files = os.listdir(save_folder_path)

                # Фильтрация файлов, соответствующих базовому имени и расширению
                base_name = "process_image" 
                ext = ".png" 

                # Создание уникального имени файла
                output_path = os.path.join(save_folder_path, f"{base_name}_{len(files) + 1}{ext}")
                cv2.imwrite(output_path, image)

if __name__ == "__main__":
    from PIL import Image
    model_path = r"D:\я у мамы программист\Diplom\KernAI-backend-fastapi\models\YOLO_detect_kern.pt"
    # model_path = r"D:\я у мамы программист\Diplom\KernAI-backend-fastapi\models\YOLO_detect_text_v.4.pt"
    yolo_det = YOLODetection(model_path)

    image_path = r"C:\Users\magmy\Downloads\Telegram Desktop\0,66 8e67bc26-fb72-482d-bbfc-a253e2e9e1cd.jpg"
    # image_path = r"C:\Users\magmy\Downloads\Telegram Desktop\party_6ecc441e-dd69-4139-87bf-63528d4589fb\party_6ecc441e-dd69-4139-87bf-63528d4589fb\step4_cluster_image\clustered_image_1.png"
    image = Image.open(image_path).convert("RGB")
    yolo_det.visualize_predictions_obb(image, conf_threshold=95)