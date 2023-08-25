import os
import supervision as sv

# Definir el tamaño mínimo y máximo de las BBox a eliminar
min_size = 10
max_size = 100

# Definir el umbral de solapamiento (IoU)
iou_threshold = 0.5


# Crear una función para determinar si una BBox está dentro del rango de tamaño
def is_bbox_in_size_range(bbox):
    width = bbox.xmax - bbox.xmin
    height = bbox.ymax - bbox.ymin
    return min_size <= width <= max_size and min_size <= height <= max_size


# Crear una función para calcular el índice de solapamiento (IoU) entre dos BBox
def bbox_iou(bbox1, bbox2):
    x1 = max(bbox1.xmin, bbox2.xmin)
    y1 = max(bbox1.ymin, bbox2.ymin)
    x2 = min(bbox1.xmax, bbox2.xmax)
    y2 = min(bbox1.ymax, bbox2.ymax)
    inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    bbox1_area = (bbox1.xmax - bbox1.xmin + 1) * (bbox1.ymax - bbox1.ymin + 1)
    bbox2_area = (bbox2.xmax - bbox2.xmin + 1) * (bbox2.ymax - bbox2.ymin + 1)
    union_area = bbox1_area + bbox2_area - inter_area
    return inter_area / union_area


# Eliminar imágenes sin anotaciones
def remove_nolabeled_data(dataset):
    images_without_annotations = [image for image in dataset if not image.annotations]
    for image in images_without_annotations:
        dataset.remove(image)


# Eliminar BBox de un tamaño determinado
def remove_big_bbox(dataset):
    for image in dataset:
        image.annotations = [
            ann for ann in image.annotations if not is_bbox_in_size_range(ann.bbox)
        ]


# Eliminar BBox superpuestos
def remove_overlapped_bbox(dataset):
    for image in dataset:
        # Ordenar las anotaciones según su puntuación de confianza, de mayor a menor
        annotations = sorted(
            image.annotations, key=lambda ann: ann.confidence, reverse=True
        )
        final_annotations = []
        for ann in annotations:
            # Comprobar si la BBox actual se solapa con alguna de las BBox finales
            if not any(
                bbox_iou(ann.bbox, final_ann.bbox) > iou_threshold
                for final_ann in final_annotations
            ):
                final_annotations.append(ann)
        image.annotations = final_annotations


def process_dataset(dataset):
    """
    Procesa un conjunto de datos.

    Args:
        dataset (sv.DetectionDataset): Conjunto de datos a procesar.
    """
    remove_nolabeled_data(dataset)
    remove_big_bbox(dataset)
    remove_overlapped_bbox(dataset)
