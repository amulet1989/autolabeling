import os
from PIL import Image


def remove_empty_labels(image_dir, label_dir, remove_multiple=False):
    for label_file in os.listdir(label_dir):
        label_path = os.path.join(label_dir, label_file)
        lab_filename = label_file.replace(".txt", ".jpg")

        # Check if the label file is empty or if remove_multiple is True and it contains multiple BBoxes
        if os.stat(label_path).st_size == 0 or (
            remove_multiple and contains_multiple_bboxes(label_path)
        ):
            os.remove(label_path)
            os.remove(os.path.join(image_dir, lab_filename))


def contains_multiple_bboxes(label_path):
    with open(label_path, "r") as f:
        lines = f.readlines()
        num_bboxes = len(lines)
        return num_bboxes > 1


# Corregir coordenadas con valores negativos o superiores a 1
def correct_bad_coords(label_dir):
    for label_file in os.listdir(label_dir):
        label_path = os.path.join(label_dir, label_file)
        with open(label_path, "r") as f:
            annotations = f.readlines()
        updated_annotations = []
        for annotation in annotations:
            class_id, x, y, w, h = map(float, annotation.strip().split())
            if x < 0:
                x = 0
            if y < 0:
                y = 0
            if x > 1:
                x = 1
            if y > 1:
                y = 1
            updated_annotations.append(f"{int(class_id)} {x} {y} {w} {h}\n")
        with open(label_path, "w") as f:
            f.writelines(updated_annotations)


def remove_large_bboxes(label_dir, max_size):
    for label_file in os.listdir(label_dir):
        label_path = os.path.join(label_dir, label_file)
        with open(label_path, "r") as f:
            annotations = f.readlines()
        updated_annotations = []
        for annotation in annotations:
            class_id, x, y, w, h = map(float, annotation.strip().split())
            if w < max_size and h < max_size:
                updated_annotations.append(annotation)
        with open(label_path, "w") as f:
            f.writelines(updated_annotations)


def remove_small_bboxes(label_dir, min_size):
    for label_file in os.listdir(label_dir):
        label_path = os.path.join(label_dir, label_file)
        with open(label_path, "r") as f:
            annotations = f.readlines()
        updated_annotations = []
        for annotation in annotations:
            class_id, x, y, w, h = map(float, annotation.strip().split())
            if w > min_size and h > min_size:
                updated_annotations.append(annotation)
        with open(label_path, "w") as f:
            f.writelines(updated_annotations)


def remove_overlapping_bboxes(label_dir, iou_threshold=0.2):
    for label_file in os.listdir(label_dir):
        label_path = os.path.join(label_dir, label_file)
        with open(label_path, "r") as f:
            annotations = f.readlines()
        updated_annotations = []
        for i, annotation in enumerate(annotations):
            bbox1 = list(map(float, annotation.strip().split()[1:]))
            skip = False
            for j in range(i + 1, len(annotations)):
                bbox2 = list(map(float, annotations[j].strip().split()[1:]))
                if iou(bbox1, bbox2) > iou_threshold:
                    skip = True
                    break
            if not skip:
                updated_annotations.append(annotation)
        with open(label_path, "w") as f:
            f.writelines(updated_annotations)


def iou(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    intersection = x_overlap * y_overlap
    union = w1 * h1 + w2 * h2 - intersection
    return intersection / union


def run_processing_dataset(
    image_dir: str,
    label_dir: str,
    max_size: float = 0.5,
    min_size: float = 0.05,
    iou_threshold: float = 0.4,
    remove_empty: bool = True,
    remove_large: bool = True,
    remove_overlapping: bool = True,
    remove_multiple: bool = True,
    remove_small: bool = True,
) -> None:
    """
    Process a dataset of images and labels.

    Parameters
    ----------
    image_dir : str
        Path to the directory containing the images.
    label_dir : str
        Path to the directory containing the labels.
    max_size : float
        Maximum size of a bounding box.
    iou_threshold : float
        Minimum IoU between two bounding boxes.
    remove_empty : bool
        If True, remove images with no labels.
    remove_large : bool
        If True, remove images with bounding boxes larger than max_size.
    remove_overlapping : bool
        If True, remove bounding boxes that overlap with each other.
    remove_multiple : bool
        If True, remove images with multiple bounding boxes.

    Returns
    -------
    None.

    Notes
    -----
    This function is used to process the Merged_Dataset/train and Merged_Dataset/valid
    datasets. It removes empty labels, images with bounding boxes larger than
    max_size, images with bounding boxes that overlap with each other, and
    images with no labels.

    """

    if remove_large and max_size is not None:
        remove_large_bboxes(label_dir, max_size)
        print("removed large")
    if remove_small and min_size is not None:
        remove_small_bboxes(label_dir, min_size)
        print("removed small")
    if remove_overlapping:
        remove_overlapping_bboxes(label_dir, iou_threshold)
        print("removed overlaping")
    if remove_empty:
        remove_empty_labels(image_dir, label_dir, remove_multiple=remove_multiple)
        print("removed empty")
    # correct_bad_coords(label_dir)
    # print("corrected negative")


def filter_images(
    input_folder,
    output_folder="/filtered_images",
    original_width=1280,
    original_height=720,
    aspect_ratio_threshold=0.9,
    min_area_ratio=0.005,
    overwrite=False,
):
    """
    Filtra y elimina imágenes de una carpeta según proporción de ancho/alto y área mínima en relación al frame original.

    Args:
        input_folder (str): Carpeta de entrada con imágenes a filtrar.
        output_folder (str): Carpeta de salida para las imágenes que pasan el filtro.
        original_width (int): Ancho del frame original.
        original_height (int): Altura del frame original.
        aspect_ratio_threshold (float): Umbral máximo de proporción ancho/alto (default: 0.9).
        min_area_ratio (float): Área mínima en relación al área del frame original (default: 0.01).
        overwrite (bool): Si es True, elimina las imágenes que no pasan el filtro (default: False).
    """
    if not os.path.exists(output_folder):
        if overwrite == False:
            os.makedirs(output_folder)

    # Área mínima permitida
    original_area = original_width * original_height
    min_area = original_area * min_area_ratio

    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg"):
            img_path = os.path.join(input_folder, filename)
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
                    aspect_ratio = width / height
                    area = width * height

                    # Filtrar por proporción y área mínima
                    if aspect_ratio > aspect_ratio_threshold or area < min_area:
                        print(
                            f"Eliminando: {filename} (aspect_ratio={aspect_ratio:.2f}, area={area})"
                        )
                        if overwrite:
                            img.close()  # Cierra explícitamente la imagen
                            os.remove(img_path)
                    else:
                        # Copiar imagen al directorio de salida
                        if overwrite == False:
                            img.save(os.path.join(output_folder, filename))
            except Exception as e:
                print(f"Error al procesar {filename}: {e}")
