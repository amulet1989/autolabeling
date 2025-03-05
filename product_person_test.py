
import os
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import torch

# Verificar disponibilidad de GPU
device = "0" if torch.cuda.is_available() else "cpu"  # "0" para GPU, "cpu" para CPU
print(f"Usando dispositivo: {device}")

def load_models():
    """Carga los modelos YOLOv8 para personas y productos"""
    person_model = YOLO(
        "trained_models/yolov8_oracle_person_all_camHD_v04.pt"
    )  # Ajusta la ruta si usas un modelo personalizado
    product_model = YOLO(
        "trained_models/yolov8_product_hand_no_cell_HD_v01.pt"
    )  # Ajusta la ruta si usas un modelo personalizado
    return person_model, product_model

def point_in_polygon(point, polygon):
    """Determina si un punto está dentro de un polígono usando ray-casting"""
    x, y = point
    inside = False
    n = len(polygon)
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def crop_image(image, box):
    """Corta una región de la imagen según un bounding box"""
    x1, y1, x2, y2 = box
    return image[y1:y2, x1:x2]

def calculate_iou(box1, box2):
    """Calcula el Intersection over Union entre dos bounding boxes"""
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

def process_images(input_dir, output_dir, roi_polygon, batch_size=64):
    """Procesa imágenes y retorna crops de productos y personas asociadas"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    person_model, product_model = load_models()

    # Obtener lista de imágenes
    img_files = [f for f in os.listdir(input_dir) if f.endswith('.jpg')]
    img_paths = [os.path.join(input_dir, f) for f in img_files]
    
    # Leer todas las imágenes
    images = [cv2.imread(path) for path in img_paths]
    images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images if img is not None]
    
    if not images:
        print("No se encontraron imágenes válidas.")
        return tuple(), tuple()  # Retornar tuplas vacías

    # Colores para las detecciones
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), 
              (255, 255, 0), (255, 0, 255), (0, 255, 255)]

    # Listas para almacenar todos los crops de todas las imágenes
    all_product_crops = []
    all_person_crops = []

    for batch_start in range(0, len(images), batch_size):
        batch_end = min(batch_start + batch_size, len(images))
        batch_images = images[batch_start:batch_end]
        batch_filenames = img_files[batch_start:batch_end]

        # Inferencia en batch con Ultralytics
        person_results = person_model(batch_images, imgsz=1280, conf=0.4, device=device)
        product_results = product_model(batch_images, imgsz=1280, conf=0.25, device=device)

        # Procesar cada imagen del batch
        for i, (person_result, product_result, original_img, filename) in enumerate(
            zip(person_results, product_results, batch_images, batch_filenames)):

            # Procesar detecciones de personas
            person_boxes = []
            for box in person_result.boxes:
                if int(box.cls) == 0:  # Clase 0 para personas
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    person_boxes.append([x1, y1, x2, y2])

            # Procesar detecciones de productos y filtrar por ROI
            product_boxes = []
            for box in product_result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                if point_in_polygon((center_x, center_y), roi_polygon):
                    product_boxes.append([x1, y1, x2, y2])

            # Asociar productos a personas y filtrar personas sin productos
            associations = []
            valid_product_boxes = []
            person_has_product = set()  # Conjunto para rastrear personas con productos

            for j, prod_box in enumerate(product_boxes):
                max_iou = 0
                associated_person = -1
                for k, pers_box in enumerate(person_boxes):
                    iou = calculate_iou(prod_box, pers_box)
                    if iou > max_iou:
                        max_iou = iou
                        associated_person = k
                if associated_person != -1:
                    associations.append(associated_person)
                    valid_product_boxes.append(prod_box)
                    person_has_product.add(associated_person)
                else:
                    associations.append(-1)

            # Obtener crops de productos válidos y personas asociadas
            for prod_box, person_idx in zip(valid_product_boxes, associations):
                if person_idx != -1:
                    product_crop = crop_image(original_img, prod_box)
                    person_crop = crop_image(original_img, person_boxes[person_idx])
                    all_product_crops.append(product_crop)
                    all_person_crops.append(person_crop)

            # # Dibujar solo personas con productos asociados y sus productos
            # for j in person_has_product:
            #     pers_box = person_boxes[j]
            #     color = colors[j % len(colors)]
            #     cv2.rectangle(original_img, (pers_box[0], pers_box[1]), 
            #                 (pers_box[2], pers_box[3]), color, 2)
            #     cv2.putText(original_img, f'Person {j}', (pers_box[0], pers_box[1]-10),
            #                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # for k, (prod_box, person_idx) in enumerate(zip(valid_product_boxes, associations)):
            #     if person_idx != -1:
            #         color = colors[person_idx % len(colors)]
            #         cv2.rectangle(original_img, (prod_box[0], prod_box[1]), 
            #                     (prod_box[2], prod_box[3]), color, 2)
            #         cv2.putText(original_img, f'Product {k}', (prod_box[0], prod_box[1]-10),
            #                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # # Dibujar ROI
            # cv2.polylines(original_img, [np.array(roi_polygon, np.int32)], True, (255, 255, 255), 2)

            # Guardar imagen procesada
            output_path = os.path.join(output_dir, f'processed_{filename}')
            # cv2.imwrite(output_path, cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR))
            print(f"Procesada: {filename}")

    # Retornar como tuplas inmutables
    return tuple(all_product_crops), tuple(all_person_crops)

def main():
    input_directory = "images/product_person/20250225_151825_g2_e1_s1"  # Cambia esta ruta
    output_directory = "images/product_person/20250225_151825_g2_e1_s1/output"  # Cambia esta ruta
    roi_polygon =  [[29, 524], [509, 276], [876, 277], [1172, 499]] # ROI especificada
    #G02-[[29, 524], [509, 276], [876, 277], [1172, 499]] / G01-[[4, 445], [303, 225], [956, 184], [1258, 376]]  
    product_crops, person_crops = process_images(input_directory, output_directory, roi_polygon, batch_size=64)

    # Ejemplo de uso de los crops retornados
    print(f"Total de productos válidos detectados: {len(product_crops)}")
    print(f"Total de personas asociadas: {len(person_crops)}")
    
    # Opcional: guardar los crops como imágenes
    for i, (prod_crop, pers_crop) in enumerate(zip(product_crops, person_crops)):
        cv2.imwrite(os.path.join(output_directory, f'product_crop_{i}.jpg'), cv2.cvtColor(prod_crop, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_directory, f'person_crop_{i}.jpg'), cv2.cvtColor(pers_crop, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    main()
