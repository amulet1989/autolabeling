import supervision as sv
from src import config
import os
from typing import List

# from typing import Dict
# import time
from tqdm import tqdm
import logging


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
        directory=video_path, extensions=["mov", "mp4"]
    )

    logging.info(f"Converting {len(video_paths)} videos to images...")

    # Convert each video to images
    cont = 0
    for video_path in tqdm(video_paths):
        video_name = video_path.stem
        image_name_pattern = video_name + "-{:05d}.jpg"
        with sv.ImageSink(
            target_dir_path=image_dir_path, image_name_pattern=image_name_pattern
        ) as sink:
            for image in sv.get_video_frames_generator(
                source_path=str(video_path), stride=frame_rate
            ):
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
    folder_paths = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            folder_paths.append(item_path)

    return folder_paths
