import supervision as sv
from src import config
import os
from typing import List
import shutil
import yaml
import cv2

# from typing import Dict
# import time
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)


def video2images(
    video_path=config.VIDEO_DIR_PATH, image_dir_path=config.IMAGE_DIR_PATH, frame_rate=1
) -> None:
    """
    Convert a video to images.

    Args:
        video_path (str): Path to the video.
        image_dir_path (str): Path to the directory to save the images.
        frame_rate (int, optional): Frame rate of the video. Defaults to 1.
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
                resized_image = cv2.resize(image, (640, 480))  # Resize image
                sink.save_image(image=resized_image)
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


def merge_datasets(dataset_paths: List, output_path: str):
    """
    Merge datasets.

    Args:
        dataset_paths (list): List of paths to the datasets.
        output_path (str): Path to the output directory.
    """
    merged_data = {
        "names": [],
        "nc": 0,
        "train": os.path.join(output_path, "train", "images"),
        "val": os.path.join(output_path, "valid", "images"),
    }

    os.makedirs(merged_data["train"], exist_ok=True)
    os.makedirs(merged_data["val"], exist_ok=True)

    label_counter = -1
    existing_classes = []
    for dataset_path in dataset_paths:
        with open(os.path.join(dataset_path, "data.yaml"), "r") as yaml_file:
            data = yaml.safe_load(yaml_file)
            merged_data["names"].extend(data["names"])

            for class_name in data["names"]:
                if class_name not in existing_classes:
                    existing_classes.append(class_name)
                    label_counter += 1

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
                                        class_id = str(label_counter)
                                        parts[0] = class_id
                                        dest_labels.write(" ".join(parts) + "\n")

        # label_counter += 1

    merged_data["names"] = list(set(merged_data["names"]))
    merged_data["nc"] = len(merged_data["names"])

    with open(os.path.join(output_path, "data.yaml"), "w") as output_yaml:
        yaml.dump(merged_data, output_yaml, default_flow_style=False)
