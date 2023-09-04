import os
import pathlib

HOME = str(pathlib.Path(__file__).parent.parent)
VIDEO_DIR_PATH = os.path.join(HOME, "videos")
IMAGE_DIR_PATH = os.path.join(HOME, "images")
DATASET_DIR_PATH = os.path.join(HOME, "dataset")
DATA_YAML_PATH = os.path.join(DATASET_DIR_PATH, "Augmented_Dataset", "data.yaml")


IMAGE_TEST = os.path.join(HOME, "test", "image_test")
DATA_TEST = os.path.join(HOME, "test", "data_test")
VIDEO_TEST = os.path.join(HOME, "test", "video_test")
