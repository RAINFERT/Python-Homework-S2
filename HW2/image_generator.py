import os
import cv2
import random
import numpy as np
from dataset_loader import DatasetLoader


class ImageGenerator:
    """
    Класс для генерации синтетических изображений с наложением случайных клеток на фон.
    Поддерживает зашумление изображений с рандомной интенсивностью.
    """

    def __init__(self, dataset_loader, img_size=(480, 640), num_imgs=5,
             min_cells=5, max_cells=25, seed=None):
        """
        :param min_cells: минимальное количество клеток на изображении
        :param max_cells: максимальное количество клеток на изображении
        """
        self.dataset_loader = dataset_loader
        self.img_size = img_size
        self.num_imgs = num_imgs
        self.min_cells = min_cells
        self.max_cells = max_cells  # ← новое поле

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def generate_and_save(self, save_dir):
        """
        Генерирует указанное количество изображений и сохраняет их в заданную директорию.

        :param save_dir: путь к директории для сохранения сгенерированных изображений
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx in range(self.num_imgs):
            img = self._generate_image()
            cv2.imwrite(os.path.join(save_dir, f"result_{idx}.png"), img)

    def _generate_image(self):
        """
        Генерирует одно синтетическое изображение с количеством клеток от min_cells до max_cells.
        """
        num_cells_this_img = random.randint(self.min_cells, self.max_cells)

        coords = self._generate_cell_coords(num_cells_this_img)

        background = self._load_image(self.dataset_loader.get_random_background(), resize=True)

        for i in range(num_cells_this_img):
            cell = self._load_image(self.dataset_loader.get_random_cell(), resize=True, with_alpha=True)
            transparency = random.uniform(0.6, 1.0)
            background = self._overlay(background, cell, coords[i], transparency)

        self._apply_random_noise(background)

        return background

    def _apply_random_noise(self, image):
        """
        Применяет случайное зашумление к изображению с процентом от 1% до 5%.

        :param image: изображение (NumPy массив) для зашумления
        """
        noise_percent = random.uniform(1, 50)  # от 1% до 5%
        self.noise_image(image, noise_percent)

    def noise_image(self, image, percent):
        """
        Добавляет случайный RGB-шум на изображение.

        :param image: изображение (NumPy массив)
        :param percent: процент пикселей, подверженных шуму
        """
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if random.random() < percent / 100:
                    image[i, j] = np.random.randint(0, 256, size=3)

    def _generate_cell_coords(self, count):
        """
        Генерирует координаты для заданного количества клеток.
        """
        return [{"h": random.randint(0, self.img_size[0] - int(self.img_size[0] * 0.01)),
                "w": random.randint(0, self.img_size[1] - int(self.img_size[1] * 0.01))}
                for _ in range(count)]

    def _load_image(self, image_path, resize=False, with_alpha=False):
        """
        Загружает изображение по указанному пути.

        :param image_path: путь к файлу изображения
        :param resize: если True — изменить размер под целевой размер изображения
        :param with_alpha: если True — загрузить изображение с альфа-каналом
        :return: загруженное изображение (NumPy массив)
        """
        flags = cv2.IMREAD_UNCHANGED if with_alpha else cv2.IMREAD_COLOR
        image = cv2.imread(image_path, flags)

        if image is None:
            raise ValueError(f"Error: Failed to load image {image_path}")

        if resize:
            image = cv2.resize(image, (self.img_size[1], self.img_size[0]))

        return image

    def _overlay(self, background, cell, coord, transparency=1.0):
        """
        Накладывает изображение клетки на фоновое изображение с заданной прозрачностью.

        :param background: фоновое изображение
        :param cell: изображение клетки
        :param coord: координаты размещения (словарь с 'h' и 'w')
        :param transparency: прозрачность наложения (от 0 до 1)
        :return: обновлённое фоновое изображение
        """
        scale = round(random.uniform(0.05, 0.20), 2)
        cell_size = int(min(self.img_size) * scale)
        cell = cv2.resize(cell, (cell_size, cell_size))

        cell, cell_h, cell_w = self._rotate_cell(cell, cell_size, cell_size)

        if coord["h"] + cell_h > background.shape[0]:
            cell_h = background.shape[0] - coord["h"]
        if coord["w"] + cell_w > background.shape[1]:
            cell_w = background.shape[1] - coord["w"]

        cell = cell[:cell_h, :cell_w]

        if cell.shape[-1] == 4:
            alpha = (cell[:, :, 3] / 255.0) * transparency
            for c in range(3):
                background[coord["h"]:coord["h"] + cell_h, coord["w"]:coord["w"] + cell_w, c] = (
                    (1 - alpha) * background[coord["h"]:coord["h"] + cell_h, coord["w"]:coord["w"] + cell_w, c] +
                    alpha * cell[:, :, c]
                )

        return background

    def _rotate_cell(self, cell, cell_h, cell_w):
        """
        Поворачивает изображение клетки на случайный угол и корректирует размер.

        :param cell: изображение клетки
        :param cell_h: исходная высота клетки
        :param cell_w: исходная ширина клетки
        :return: повернутое изображение, новая высота, новая ширина
        """
        center = (cell_w // 2, cell_h // 2)
        angle = random.randint(0, 360)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        new_w = int((cell_h * abs(matrix[0, 1])) + (cell_w * abs(matrix[0, 0])))
        new_h = int((cell_h * abs(matrix[0, 0])) + (cell_w * abs(matrix[0, 1])))

        matrix[0, 2] += (new_w / 2) - center[0]
        matrix[1, 2] += (new_h / 2) - center[1]

        rotated = cv2.warpAffine(cell, matrix, (new_w, new_h))

        return rotated, new_h, new_w