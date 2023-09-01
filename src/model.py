from autodistill.detection import CaptionOntology
from autodistill_grounding_dino import GroundingDINO
from src import config

import cv2
import supervision as sv
from tqdm import tqdm
import glob
import os
import gc

from autodistill.helpers import split_data


import numpy as np


def label_multiple(
    self,
    input_folder: str,
    extension: str = ".jpg",
    output_folder: str = None,
    num_datasets: int = 4,
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

        split_data(output_folder + f"/{directory_name}_{i+1}")

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
    )
