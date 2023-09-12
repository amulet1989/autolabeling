import supervision as sv
from src import config
import os
from typing import List
import shutil
import yaml
import cv2

import tkinter as tk
from tkinter import filedialog

from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)


def video2images(
    video_path=config.VIDEO_DIR_PATH,
    image_dir_path=config.IMAGE_DIR_PATH,
    frame_rate=1,
    height=None,
    width=None,
) -> None:
    """
    Convert a video to images.

    Args:
        video_path (str): Path to the video.
        image_dir_path (str): Path to the directory to save the images.
        frame_rate (int, optional): Frame rate of the video. Defaults to 1.
        heigth (int, optional): Higth to resize input frames. Defaults to None.
        width (int, optional): Width to resize input frames. Defaults to None.

    """
    # Get all videos paths in the directory

    video_paths = sv.list_files_with_extensions(
        directory=video_path, extensions=["mp4"]
    )

    logging.info(f"Converting {len(video_paths)} videos to images...")

    # Convert each video to images
    cont = 0
    for out_video_path in tqdm(video_paths):
        video_name = out_video_path.stem
        image_name_pattern = video_name + "-{:05d}.jpg"
        with sv.ImageSink(
            target_dir_path=image_dir_path, image_name_pattern=image_name_pattern
        ) as sink:
            for image in sv.get_video_frames_generator(
                source_path=str(out_video_path), stride=frame_rate
            ):
                if height is not None and width is not None:
                    image = cv2.resize(image, (width, height))

                sink.save_image(image=image)
                cont += 1
    logging.info(f"Obtained  {cont} images.")


def get_video_folder_paths(directory: str) -> List:
    """
    Get all video folder paths in a directory.

    Args:
        directory (str): Path to the directory.

    Returns:
        list: List of folder paths.

    """
    list_of_dir = os.listdir(directory)
    folder_paths = []

    if len(list_of_dir) == 1:
        return [os.path.join(directory, list_of_dir[0])]
    else:
        for item in list_of_dir:
            item_path = os.path.join(directory, item)
            if os.path.isdir(item_path):
                folder_paths.append(item_path)

        return folder_paths


import os
import shutil
import logging
from typing import List
from tqdm import tqdm
import yaml


def merge_datasets(dataset_paths: List, output_path: str):
    """
    Merge datasets.

    Args:
        dataset_paths (list): List of paths to the datasets.
        output_path (str): Path to the output directory.
    """
    logging.info("Merging datasets...")
    merged_data = {
        "names": [],
        "nc": 0,
        "train": os.path.join(output_path, "train", "images"),
        "val": os.path.join(output_path, "valid", "images"),
    }

    os.makedirs(merged_data["train"], exist_ok=True)
    os.makedirs(merged_data["val"], exist_ok=True)

    # Inicializar el mapeo entre class_id y los valores de data["names"]
    class_id_mapping = {}

    for dataset_path in tqdm(dataset_paths):
        with open(os.path.join(dataset_path, "data.yaml"), "r") as yaml_file:
            data = yaml.safe_load(yaml_file)
            merged_data["names"].extend(data["names"])

            for class_name in data["names"]:
                if class_name not in class_id_mapping:
                    # Asigna un nuevo class_id basado en el mapeo actual
                    class_id = len(class_id_mapping)
                    class_id_mapping[class_name] = class_id

        for class_name in data["names"]:
            for split in ["train", "valid"]:
                src_images_dir = os.path.join(dataset_path, split, "images")
                src_labels_dir = os.path.join(dataset_path, split, "labels")

                dest_images_dir = os.path.join(output_path, split, "images")
                os.makedirs(dest_images_dir, exist_ok=True)

                dest_labels_dir = os.path.join(output_path, split, "labels")
                os.makedirs(dest_labels_dir, exist_ok=True)

                for image_file in os.listdir(src_images_dir):
                    shutil.copy(
                        os.path.join(src_images_dir, image_file),
                        os.path.join(dest_images_dir, image_file),
                    )
                    image_name = os.path.splitext(image_file)[0]

                    src_labels_file = os.path.join(src_labels_dir, f"{image_name}.txt")
                    dest_labels_file = os.path.join(
                        dest_labels_dir, f"{image_name}.txt"
                    )
                    if os.path.exists(src_labels_file):
                        with open(src_labels_file, "r") as src_labels:
                            with open(dest_labels_file, "a") as dest_labels:
                                for line in src_labels:
                                    parts = line.strip().split()
                                    if len(parts) >= 5:
                                        class_id = class_id_mapping[class_name]
                                        parts[0] = str(class_id)
                                        dest_labels.write(" ".join(parts) + "\n")

    # Actualizar la información del conjunto de datos fusionado
    print(merged_data['names'])
    #merged_data["names"] = list(set(merged_data["names"]))
    
    seen = set()
    unique_names = []

    for name in merged_data["names"]:
        if name not in seen:
            unique_names.append(name)
            seen.add(name)

    merged_data["names"] = unique_names

    merged_data["nc"] = len(merged_data["names"])
    print(merged_data['names'])

    # Guardar metadatos en un archivo data.yaml
    with open(os.path.join(output_path, "data.yaml"), "w") as output_yaml:
        yaml.dump(merged_data, output_yaml, default_flow_style=False)

    logging.info(f"Merged datasets created at {output_path}")


def seleccionar_imagen():
    root = tk.Tk()
    root.withdraw()  # Oculta la ventana principal de tkinter

    file_path = filedialog.askopenfilename(
        title="Seleccionar una imagen",
        filetypes=[
            (
                "Archivos de imagen",
                "*.jpg *.jpeg *.png *.gif *.bmp *.ppm *.pgm *.tif *.tiff",
            )
        ],
    )

    if file_path:
        return file_path
    else:
        return None
