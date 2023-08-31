import argparse
from src import config
import json
from src.util import video2images, get_video_folder_paths, merge_datasets
from src.model import autolabel_images
from src.dataset_processing import run_processing_dataset
from src.data_augmentation import augment_dataset
from tqdm import tqdm
import os
from typing import Dict
import shutil


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
    if os.path.exists(image_dir_path):
        shutil.rmtree(image_dir_path)  # Si ya existe Borra el directorio y su contenido
    os.mkdir(image_dir_path)  # y lo Crea el directorio nuevamente

    # create dataset folder for each video folder
    dataset_dir_path = output_images
    # dataset_dir_path = os.path.join(output_images, os.path.basename(video_path))
    # if os.path.exists(dataset_dir_path):
    #    shutil.rmtree(
    #        dataset_dir_path
    #    )  # Si ya existe borra el directorio y su contenido
    # os.mkdir(dataset_dir_path)  # y lo crea el directorio nuevamente

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
        "--frame_rate", default=1, type=int, help="Tasa de cuadros por segundo"
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
    parser.add_argument(
        "--max_size", default=0.5, type=float, help="Tamaño maximo de BBox"
    )
    parser.add_argument(
        "--iou_threshold",
        default=0.1,
        type=float,
        help="IoU maxima permitaida de dos BBox",
    )
    parser.add_argument(
        "--remove_empty",
        default=True,
        type=bool,
        help="Eliminar imagenes sin BBox detectados",
    )
    parser.add_argument(
        "--remove_large",
        default=True,
        type=bool,
        help="Eliminar BBox demasiado grandes",
    )
    parser.add_argument(
        "--remove_overlapping",
        default=True,
        type=bool,
        help="Si hay BBox que se superpongan dejar solo uno",
    )
    parser.add_argument(
        "--remove_multiple",
        default=True,
        type=bool,
        help="Si queda mas de un objeto detectado eliminar imagen",
    )
    parser.add_argument(
        "--augmented_for",
        default=4,
        type=int,
        help="Proporción en la que se aumentará el dataset",
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
    # Si existe la carpeta Merged_Dataset Borra el directorio y su contenido
    output_path = os.path.join(config.DATASET_DIR_PATH, "Merged_Dataset")
    if os.path.exists(output_path):
        shutil.rmtree(output_path)  # Si ya existe Borra el directorio y su contenido
    # Nombre de carpeta de cada datatset individual
    folders = os.listdir(args.output_images)
    # Lista de paths de cada dataset individual
    dataset_paths = [os.path.join(args.output_images, folder) for folder in folders]
    # Hacer el merge
    merge_datasets(dataset_paths, output_path)

    # Procesar Merged_Dataset/train para eliminar errores de anotación
    run_processing_dataset(
        os.path.join(output_path, "train", "images"),
        os.path.join(output_path, "train", "labels"),
        max_size=args.max_size,
        iou_threshold=args.iou_threshold,
        remove_empty=args.remove_empty,
        remove_large=args.remove_large,
        remove_overlapping=args.remove_overlapping,
        remove_multiple=args.remove_multiple,
    )
    # Procesar Merged_Dataset/valid para eliminar errores de anotación
    run_processing_dataset(
        os.path.join(output_path, "valid", "images"),
        os.path.join(output_path, "valid", "labels"),
        max_size=args.max_size,
        iou_threshold=args.iou_threshold,
        remove_empty=args.remove_empty,
        remove_large=args.remove_large,
        remove_overlapping=args.remove_overlapping,
        remove_multiple=args.remove_multiple,
    )

    # Generar la aumentación de datos
    dataset_path = os.path.join(config.DATASET_DIR_PATH, "Merged_Dataset")
    augmented_dataset_path = os.path.join(config.DATASET_DIR_PATH, "Augmented_Dataset")
    # creating aumented datased directory
    if not os.path.exists(augmented_dataset_path):
        os.makedirs(augmented_dataset_path)

    augment_dataset(
        dataset_path, augmented_dataset_path, augmented_for=args.augmented_for
    )

    print("Proceso finalizado")


if __name__ == "__main__":
    main()
