import os
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path


def load_models():
    """Carga los modelos YOLOv8 para personas y productos"""
    person_model = YOLO(
        "trained_models/yolov8_oracle_person_all_camHD_v04.pt"
    )  # Ajusta la ruta si usas un modelo personalizado
    product_model = YOLO(
        "trained_models/yolov8_product_hand_no_cell_HD_v01.pt"
    )  # Ajusta la ruta si usas un modelo personalizado
    return person_model, product_model


def calculate_iou(box1, box2):
    """Calcula el Intersection over Union entre dos bounding boxes"""
    # box format: [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection

    return intersection / union if union > 0 else 0.0


def process_images(input_dir, output_dir):
    """Procesa todas las imágenes en el directorio de entrada"""
    # Crear directorio de salida si no existe
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Cargar modelos
    person_model, product_model = load_models()

    # Colores para cada persona (ciclo de colores)
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
    ]

    # Procesar cada imagen
    for img_file in os.listdir(input_dir):
        if img_file.endswith(".jpg"):
            img_path = os.path.join(input_dir, img_file)
            img = cv2.imread(img_path)
            if img is None:
                continue

            # Detectar personas
            person_results = person_model(img, imgsz=640, conf=0.4)
            person_boxes = []
            for r in person_results:
                boxes = r.boxes
                for box in boxes:
                    if int(box.cls) == 0:  # Clase 0 es persona en YOLO estándar
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        person_boxes.append([x1, y1, x2, y2])

            # Detectar productos
            product_results = product_model(img, imgsz=640, conf=0.15)
            product_boxes = []
            for r in product_results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    product_boxes.append([x1, y1, x2, y2])

            # Asociar productos a personas
            associations = []
            for prod_box in product_boxes:
                max_iou = 0
                associated_person = -1
                for i, pers_box in enumerate(person_boxes):
                    iou = calculate_iou(prod_box, pers_box)
                    if iou > max_iou:  # Umbral mínimo de IoU
                        max_iou = iou
                        associated_person = i
                associations.append(associated_person)

            # Dibujar boxes en la imagen
            for i, pers_box in enumerate(person_boxes):
                color = colors[i % len(colors)]
                cv2.rectangle(
                    img,
                    (pers_box[0], pers_box[1]),
                    (pers_box[2], pers_box[3]),
                    color,
                    2,
                )
                cv2.putText(
                    img,
                    f"Person {i}",
                    (pers_box[0], pers_box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    color,
                    2,
                )

            for j, (prod_box, person_idx) in enumerate(
                zip(product_boxes, associations)
            ):
                if person_idx != -1:
                    color = colors[person_idx % len(colors)]
                    cv2.rectangle(
                        img,
                        (prod_box[0], prod_box[1]),
                        (prod_box[2], prod_box[3]),
                        color,
                        2,
                    )
                    cv2.putText(
                        img,
                        f"Product {j}",
                        (prod_box[0], prod_box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        color,
                        2,
                    )

            # Guardar imagen procesada
            output_path = os.path.join(output_dir, f"processed_{img_file}")
            cv2.imwrite(output_path, img)
            print(f"Procesada: {img_file}")


def main():
    input_directory = "images/20250224_153015_g2_e1_s1"  # Cambia esta ruta
    output_directory = "images/20250224_153015_g2_e1_s1/output"  # Cambia esta ruta
    process_images(input_directory, output_directory)


if __name__ == "__main__":
    main()
