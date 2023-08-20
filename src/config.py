import os
import pathlib

HOME = str(pathlib.Path(__file__).parent.parent)
VIDEO_DIR_PATH = os.path.join(HOME, "videos")
IMAGE_DIR_PATH = os.path.join(HOME, "images")
DATASET_DIR_PATH = os.path.join(HOME, "dataset")

if not os.path.exists(VIDEO_DIR_PATH):
    os.mkdir(VIDEO_DIR_PATH)

if not os.path.exists(IMAGE_DIR_PATH):
    os.mkdir(IMAGE_DIR_PATH)

if not os.path.exists(DATASET_DIR_PATH):
    os.mkdir(DATASET_DIR_PATH)
