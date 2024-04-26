from autodistill.detection import CaptionOntology
from autodistill_grounding_dino import GroundingDINO
from autodistill.helpers import split_data

# from autodistill_yolov8 import YOLOv8
from ultralytics import YOLO

from src import config

import cv2
import supervision as sv
from tqdm import tqdm
import glob
import os
import gc
import shutil
import random
import yaml

import numpy as np


def label_multiple(
    self,
    input_folder: str,
    extension: str = ".jpg",
    output_folder: str = None,
    num_datasets: int = 4,
    val: bool = False,
    val_ratio: float = 0.2,
) -> None:
    if output_folder is None:
        output_folder = input_folder + "_labeled"

    os.makedirs(output_folder, exist_ok=True)

    directory_name = os.path.basename(os.path.normpath(input_folder))

    files = glob.glob(input_folder + "/*" + extension)
    file_chunks = np.array_split(files, num_datasets)

    for i, chunk in enumerate(file_chunks):
        images_map = {}
        detections_map = {}

        progress_bar = tqdm(chunk, desc=f"Labeling images for dataset {i+1}")
        # iterate through images in input_folder
        for f_path in progress_bar:
            progress_bar.set_description(
                desc=f"Labeling {f_path} for dataset {i+1}", refresh=True
            )
            image = cv2.imread(f_path)

            f_path_short = os.path.basename(f_path)
            images_map[f_path_short] = image.copy()
            detections = self.predict(f_path)
            detections_map[f_path_short] = detections

        dataset = sv.DetectionDataset(
            self.ontology.classes(), images_map, detections_map
        )

        dataset.as_yolo(
            os.path.join(output_folder, f"{directory_name}_{i+1}", "images"),
            os.path.join(output_folder, f"{directory_name}_{i+1}", "annotations"),
            min_image_area_percentage=0.01,
            data_yaml_path=os.path.join(
                output_folder, f"{directory_name}_{i+1}", "data.yaml"
            ),
        )

        if val:
            split_data(
                output_folder + f"/{directory_name}_{i+1}", split_ratio=val_ratio
            )
        else:
            # copiar la carpeta images a train/images
            shutil.copytree(
                output_folder + f"/{directory_name}_{i+1}/images",
                output_folder + f"/{directory_name}_{i+1}/train/images",
            )
            # Eliminar el directorio de origen y su contenido
            shutil.rmtree(output_folder + f"/{directory_name}_{i+1}/images")
            # copiar la carpeta annotations a train/labels
            shutil.copytree(
                output_folder + f"/{directory_name}_{i+1}/annotations",
                output_folder + f"/{directory_name}_{i+1}/train/labels",
            )
            shutil.rmtree(output_folder + f"/{directory_name}_{i+1}/annotations")

        # Liberar memoria
        images_map.clear()
        detections_map.clear()
        del dataset
        gc.collect()

        print(f"Labeled dataset {i+1} created - ready for distillation.")


# Modificar el método label de la clase
GroundingDINO.label = label_multiple
base_model = GroundingDINO(ontology=CaptionOntology({"product held by": "product"}))


# Funcion principal
def autolabel_images(
    input_folder=config.IMAGE_DIR_PATH,
    ontology={"hand holding": "hand", "product held by": "product"},
    box_threshold=0.35,
    text_threshold=0.25,
    output_folder=config.DATASET_DIR_PATH,
    extension=".jpg",
    num_datasets=4,
    val=False,
    val_ratio=0.2,
):
    """
    Autolabel images in a folder.

    Args:
        input_folder (str, optional): Path to the folder with the images. Defaults to config.IMAGE_DIR_PATH.
        ontology (Dict[str, str], optional): Ontology of the captions. Defaults to {"hand holding": "hand", "product held by": "product"}.
        box_threshold (float, optional): Box threshold. Defaults to 0.35.
        text_threshold (float, optional): Text threshold. Defaults to 0.25.
        output_folder (str, optional): Path to the folder to save the labeled images. Defaults to config.DATASET_DIR_PATH.
        extension (str, optional): Extension of the images. Defaults to ".jpg".
        num_datasets (int, optional): Number of datasets to split the images. Defaults to 4.
    """

    # create the ontology
    ontology = CaptionOntology(ontology)

    # base_model = GroundingDINO(
    #    ontology=ontology, box_threshold=box_threshold, text_threshold=text_threshold
    # )
    base_model.ontology = ontology
    base_model.box_threshold = box_threshold
    base_model.text_threshold = text_threshold

    # label all images in a folder called `context_images`
    base_model.label(
        input_folder=input_folder,
        extension=extension,
        output_folder=output_folder,
        num_datasets=num_datasets,
        val=val,
        val_ratio=val_ratio,
    )


def predict_and_visualice(
    image_path,
    ontology,
    box_threshold=0.35,
    text_threshold=0.25,
):
    """
    Predict and visualize the image.

    Args:
        image_path (str): Path to the image.
        ontology (Dict[str, str]): Ontology of the captions.
        box_threshold (float, optional): Box threshold. Defaults to 0.35.
        text_threshold (float, optional): Text threshold. Defaults to 0.25.

    Returns:
        result (sv.DetectionResult): Detection result.

    """
    image = cv2.imread(image_path)
    ontology = CaptionOntology(ontology)
    base_model.ontology = ontology
    base_model.box_threshold = box_threshold
    base_model.text_threshold = text_threshold

    result = base_model.predict(image_path)

    box_annotator = sv.BoxAnnotator()

    labels = [
        f"{base_model.ontology.classes()[class_id]} {confidence:.2f}"
        for class_id, confidence in zip(result.class_id, result.confidence)
    ]
    annotated_image = box_annotator.annotate(
        image.copy(), detections=result, labels=labels
    )

    sv.plot_image(image=annotated_image, size=(16, 10))


# funcion para crear una carpeta dentro de un directorio si no existe
def crear_carpeta(directorio, nombre=None):
    if nombre != None:
        directorio = os.path.join(directorio, nombre)
    if not os.path.exists(directorio):
        os.mkdir(directorio)
    return directorio


def inferir_y_guardar(
    image_paths, model, imag_folder, label_folder, confidence=0.7, iou=0.7, imgsz=1280
):
    # Para visualizar
    win_name = "Camera Preview"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    # Inferencia
    for image_path in image_paths:
        results = model(source=image_path, conf=confidence, iou=iou, imgsz=imgsz)
        # print(len(results))
        # salvar imagen en train_images_folder
        image_name = os.path.basename(image_path)
        shutil.copy(image_path, imag_folder)

        # crear un txt para almacenar las labels de la imagen
        txt_name = image_name.replace(".jpg", ".txt")
        txt_path = os.path.join(label_folder, txt_name)
        with open(txt_path, "a") as f:
            label = results[0].boxes.cls
            coordenadas = results[0].boxes.xywhn
            for lab, (x, y, w, h) in zip(label, coordenadas):
                f.write("{} {} {} {} {}\n".format(int(lab), x, y, w, h))

        # Visualiza
        for r in results:
            boxes = r.boxes  # Boxes object for bbox outputs

            image = r.orig_img.copy()
            if boxes.cls.numel() > 0:
                classe = boxes.cls.tolist()
                label = r.names
                scores = boxes.conf.tolist()  # Confidence scores

                # Draw BBoxes on the image
                # for box, label, score in zip(boxes, labels, scores):
                for i, box in enumerate(boxes.xyxy):
                    x1, y1, x2, y2 = map(int, box)  # box
                    color = (0, 255, 0)  # Green color
                    thickness = 2

                    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

                    text = f"{label[int(classe[i])]} ({scores[i]:.2f})"
                    # print(text)

                    cv2.putText(
                        image,
                        text,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        color,
                        thickness,
                    )

            cv2.imshow(win_name, image)

            # Wait for a key press and check the pressed key
            key = cv2.waitKey(1)  # & 0xFF
            if key == ord("q"):  # Press 'q' to exit
                break
            elif key == ord("n"):  # Press 'n' to show the next image
                continue

    # Release VideoCapture and destroy windows
    # cap.release()
    cv2.destroyAllWindows()

    return results[0].names


def label_multiple_yolov8(
    model_path,
    input_folder: str,
    output_folder: str = None,
    confidence=0.8,
    iou=0.7,
    imgsz=1280,
    tracking=False,
    val_ratio=0.2,
):
    # hacer una lista de todos los path de imagenes en el directorio input_folder
    glob_path = os.path.join(input_folder, "*.jpg")
    image_paths = glob.glob(glob_path)
    modelyolo = YOLO(model_path)  # YOLOv8

    # hacer una particion aleatoria 80 - 20 de la lista de imagenes
    if tracking:
        train_images = random.sample(image_paths, int(1.0 * len(image_paths)))
        valid_images = list(set(image_paths) - set(train_images))
    else:
        train_images = random.sample(
            image_paths, int((1 - val_ratio) * len(image_paths))
        )
        valid_images = list(set(image_paths) - set(train_images))

    # obtener en nombre de carpeta de input_folder
    directory_name = os.path.basename(os.path.normpath(input_folder))
    output_folder = crear_carpeta(output_folder, directory_name)

    train = crear_carpeta(output_folder, "train")
    train_images_folder = crear_carpeta(train, "images")
    train_labels_folder = crear_carpeta(train, "labels")

    if tracking == False:
        valid = crear_carpeta(output_folder, "valid")
        valid_images_folder = crear_carpeta(valid, "images")
        valid_labels_folder = crear_carpeta(valid, "labels")

    train_names = inferir_y_guardar(
        train_images,
        modelyolo,
        train_images_folder,
        train_labels_folder,
        confidence,
        iou=iou,
        imgsz=imgsz,
    )
    if tracking == False:
        inferir_y_guardar(
            valid_images,
            modelyolo,
            valid_images_folder,
            valid_labels_folder,
            confidence,
            iou=iou,
            imgsz=imgsz,
        )

    # print(train_names)
    # Definir otras variables
    nc = len(train_names)
    if tracking == False:
        data = {
            "names": list(train_names.values()),
            "nc": nc,
            "train": train,
            "val": valid,
        }
    else:
        data = {"names": list(train_names.values()), "nc": nc, "train": train}

    # Escribir el diccionario en un archivo YAML
    yaml_path = os.path.join(output_folder, "data.yaml")
    with open(yaml_path, "w") as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False)
