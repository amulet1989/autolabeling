import argparse
from src import config
import json
from src.util import video2images, get_video_folder_paths, merge_datasets
from src.model import autolabel_images
from tqdm import tqdm
import os
from typing import Dict


def mypipeline(
    video_path: str,
    image_dir_path: str,
    frame_rate: isinstance,
    ontology: Dict,
    output_images: str,
    extension: str,
    box_threshold: float,
    text_threshold: float,
) -> None:
    """
    Pipeline para procesar videos y detección de objetos

    Args:
        video_path (str): Ruta al archivo de video
        image_dir_path (str): Ruta al directorio de imágenes
        frame_rate (int): Tasa de cuadros por segundo
        ontology (Dict): {caption: class}
        output_images (str): Carpeta de salida para imágenes procesadas
        extension (str): Extension de los archivos de imágenes
        box_threshold (float): Box threshold
        text_threshold (float): Text threshold
    """
    # create image folder for each video folder
    image_dir_path = os.path.join(image_dir_path, os.path.basename(video_path))
    if not os.path.exists(image_dir_path):
        os.mkdir(image_dir_path)

    # create dataset folder for each video folder
    dataset_dir_path = os.path.join(output_images, os.path.basename(video_path))
    if not os.path.exists(dataset_dir_path):
        os.mkdir(dataset_dir_path)

    # convert video to images
    video2images(video_path, image_dir_path, frame_rate)

    # 2- Run autolabel for each image in image_dir_path
    autolabel_images(
        input_folder=image_dir_path,
        ontology=ontology,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        output_folder=dataset_dir_path,
        extension=extension,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline para procesar videos y detección de objetos"
    )
    parser.add_argument(
        "--video",
        default=config.VIDEO_DIR_PATH,
        type=str,
        help="Ruta al archivo de video",
    )
    parser.add_argument(
        "--image_dir",
        default=config.IMAGE_DIR_PATH,
        type=str,
        help="Ruta al directorio de imágenes",
    )
    parser.add_argument(
        "--frame_rate", default=10, type=int, help="Tasa de cuadros por segundo"
    )
    parser.add_argument(
        "--ontology",
        default="ontology.json",
        type=str,
        help="Archivo JSON de ontología",
    )

    parser.add_argument(
        "--output_images",
        default=config.DATASET_DIR_PATH,
        type=str,
        help="Carpeta de salida para imágenes procesadas",
    )
    parser.add_argument(
        "--extension", default=".jpg", type=str, help="Extensión de archivo"
    )
    parser.add_argument(
        "--box_threshold", default=0.35, type=float, help="Umbral de caja"
    )
    parser.add_argument(
        "--text_threshold", default=0.25, type=float, help="Umbral de texto"
    )
    args = parser.parse_args()

    # Run pipeline
    # 1- Convert videos to images
    # 2- Run autolabel for each image
    # 3- Union de los Datasets

    # get all folders into video_dir
    video_paths = get_video_folder_paths(args.video)
    # run autolabels
    if len(video_paths) == 0:
        print("No se encontraron videos")
        return
    elif len(video_paths) == 1:
        mypipeline(
            video_paths[0],
            args.image_dir,
            args.frame_rate,
            json.load(open(os.path.join(video_paths[0], "ontology.json"))),
            args.output_images,
            args.extension,
            args.box_threshold,
            args.text_threshold,
        )
    else:
        for video_path in tqdm(video_paths):
            mypipeline(
                video_path,
                args.image_dir,
                args.frame_rate,
                json.load(open(os.path.join(video_path, "ontology.json"))),
                args.output_images,
                args.extension,
                args.box_threshold,
                args.text_threshold,
            )

    # Unir datasets
    # Nombtre d ecarpeta de cada datatset individual
    folders = os.listdir(args.output_images)
    # Lista de paths de cada dataset individual
    dataset_paths = [os.path.join(args.output_images, folder) for folder in folders]
    # Union de los datasets
    output_path = os.path.join(config.DATASET_DIR_PATH, "Merged_Dataset")
    merge_datasets(dataset_paths, output_path)
    print("Proceso finalizado")


if __name__ == "__main__":
    main()
