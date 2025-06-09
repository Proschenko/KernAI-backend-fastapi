import os
import shutil
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm


class FolderOperation:
    def __init__(self):
        pass

    @staticmethod
    def rename_files_in_directory(directory, extension=".jpg", extension2=".JPG"):
        """
        Переименовывает файлы в директории с указанными расширениями.

        Аргументы:
            directory: Путь к директории с файлами.
            extension: Основное расширение файлов.
            extension2: Дополнительное расширение файлов.
        """
        # Получаем список всех файлов в директории
        files = [f for f in os.listdir(directory) if f.endswith(extension) or f.endswith(extension2)]

        # Сортируем файлы, если нужно (например, по имени)
        files.sort()

        # Переименовываем файлы
        for index, filename in enumerate(files, start=1):
            # Формируем новое имя файла
            new_filename = f"{index:04d}{extension}"

            # Полный путь к старому и новому файлу
            old_file = os.path.join(directory, filename)
            new_file = os.path.join(directory, new_filename)

            # Переименовываем файл
            os.rename(old_file, new_file)
            print(f"Переименован: {old_file} -> {new_file}")

    @staticmethod
    def delete_files_in_folder(folder_path):
        """
        Удаляет все файлы из папки.

        Аргументы:
            folder_path: Путь к папке.
        """
        # Создание папки для сохранения кадров, если она не существует
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        for root, dirs, files in os.walk(folder_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                try:
                    os.unlink(file_path)
                except Exception as e:
                    print(f"Ошибка удаления файла: {file_path} -- {e}")

    @staticmethod
    def get_image_paths(folder_path):
        """
        Получает пути к изображениям в указанной папке.

        Аргументы:
            folder_path: Путь к папке с изображениями.

        Возвращает:
            list: Список путей к изображениям.
        """
        image_extensions = ['.jpg', '.jpeg', '.png']
        image_paths = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_paths.append(os.path.join(root, file))
        return image_paths

    def resize_source_image(self, input_folder_inner, output_folder_inner, size_inner=(960, 540)):
        """
        Изменяет размер изображений в папке и сохраняет их в новой папке.

        Аргументы:
            input_folder_inner: Путь к папке с исходными изображениями.
            output_folder_inner: Путь к папке для сохранения измененных изображений.
            size_inner: Новый размер изображений.
        """
        self.delete_files_in_folder(output_folder_inner)

        # Получаем список файлов в папке с входными изображениями
        files = os.listdir(input_folder_inner)

        for file in tqdm(files):
            # Открываем изображение
            img = Image.open(os.path.join(input_folder_inner, file))

            # Проверяем ориентацию изображения
            if img.width < img.height:
                # Поворачиваем изображение на 90 градусов влево
                img = img.rotate(90, expand=True)

            # Конвертируем изображение в заданный размер
            img = img.resize(size_inner)

            # Создаем новое квадратное изображение с пустыми областями
            new_img = Image.new("RGB", (size_inner[0], size_inner[0]), (0, 0, 0))

            # Вычисляем положение для вставки исходного изображения
            paste_x = 0
            paste_y = (size_inner[0] - img.height) // 2

            # Вставляем исходное изображение в центр нового изображения
            new_img.paste(img, (paste_x, paste_y))

            # Сохраняем изображение в выходную папку
            new_img.save(os.path.join(output_folder_inner, file))

    @staticmethod
    def __resize_image(image_path, output_path, size_inner=360):
        """
        Изменяет размер изображения.

        Аргументы:
            image_path: Путь к исходному изображению.
            output_path: Путь для сохранения измененного изображения.
            size_inner: Новый размер изображения.
        """
        with Image.open(image_path) as img:
            img = img.resize((size_inner, size_inner))
            img.save(output_path)

    def resize_kern_images_in_folder(self, folder_path_inner, size_inner):
        """
        Изменяет размер всех изображений в папке.

        Аргументы:
            folder_path_inner: Путь к папке с изображениями.
            size_inner: Новый размер изображений.
        """
        image_paths = self.get_image_paths(folder_path_inner)
        if not image_paths:
            print("Изображения не найдены в папке.")
            return

        output_folder_inner = os.path.join(folder_path_inner, f'resized_{size_inner}x{size_inner}')
        os.makedirs(output_folder_inner, exist_ok=True)

        for image_path in image_paths:
            output_path = os.path.join(output_folder_inner, os.path.basename(image_path))
            self.__resize_image(image_path, output_path, size_inner)
            print(f"Изменен размер {image_path} до {size_inner}x{size_inner} и сохранено в {output_path}")

    @staticmethod
    def prepare_dataset_with_hierarchy(dataset_dir, dataset_with_annotations_dir, yaml_file):
        """
        Подготавливает датасет с иерархической структурой.

        Аргументы:
            dataset_dir: Путь к папке для сохранения датасета.
            dataset_with_annotations_dir: Путь к папке с аннотациями.
            yaml_file: Путь к YAML файлу.
        """
        images_dir = os.path.join(dataset_with_annotations_dir, 'images\\Train')
        labels_dir = os.path.join(dataset_with_annotations_dir, 'labels\\Train')

        # Удалите содержимое папки dataset_dir, если она существует
        if os.path.exists(dataset_dir):
            shutil.rmtree(dataset_dir)

        # Создайте структуру директорий для датасета
        os.makedirs(dataset_dir, exist_ok=True)
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(dataset_dir, split, 'images'), exist_ok=True)
            os.makedirs(os.path.join(dataset_dir, split, 'labels'), exist_ok=True)

        # Скопируйте yaml файл в директорию датасета
        shutil.copy(yaml_file, dataset_dir)

        # Получите список файлов аннотаций и файлов изображений
        annotation_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
        image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg') or f.endswith('.png')]

        # Извлеките базовые имена без расширений
        annotation_names = [os.path.splitext(f)[0] for f in annotation_files]
        image_names = [os.path.splitext(f)[0] for f in image_files]

        # Найдите изображения без соответствующих аннотаций
        images_to_delete = set(image_names) - set(annotation_names)

        # Удалите изображения без соответствующих аннотаций
        for image in tqdm(images_to_delete, desc="Удаление файлов"):
            for ext in ['.jpg', '.png']:
                image_path = os.path.join(images_dir, image + ext)
                if os.path.exists(image_path):
                    os.remove(image_path)
                    # print(f"Удалено {image_path}")

        # Разделите данные на обучающую, валидационную и тестовую выборки
        train_names, test_names = train_test_split(annotation_names, test_size=0.3, random_state=42)
        val_names, test_names = train_test_split(test_names, test_size=0.5, random_state=42)
        print(
            f"Размер обучающей выборки {len(train_names)}, размер валидационной выборки {len(val_names)}, размер тестовой выборки {len(test_names)}")

        # Скопируйте изображения и аннотации в соответствующие директории
        for split, names in tqdm(zip(['train', 'val', 'test'], [train_names, val_names, test_names]),
                                 desc="Копирование файлов"):
            for name in names:
                for ext in ['.jpg', '.png']:
                    image_src = os.path.join(images_dir, name + ext)
                    image_dst = os.path.join(dataset_dir, split, 'images', name + ext)
                    if os.path.exists(image_src):
                        shutil.copy(image_src, image_dst)
                        # print(f"Копирование файла из {image_src} в {image_dst}")
                annotation_src = os.path.join(labels_dir, name + '.txt')
                annotation_dst = os.path.join(dataset_dir, split, 'labels', name + '.txt')
                if os.path.exists(annotation_src):
                    shutil.copy(annotation_src, annotation_dst)
                    # print(f"Копирование файла из {annotation_src} в {annotation_dst}")
                else:
                    print(f"Путь {annotation_src} не существует")
