import argparse
from src import config
import json
from src.util import video2images, get_video_folder_paths, merge_datasets
from src.model import autolabel_images, label_multiple_yolov8
from src.dataset_processing import run_processing_dataset
from src.data_augmentation import augment_dataset
from tqdm import tqdm
import os
from typing import Dict
import shutil


def mypipeline_track(
    video_path: str,
    image_dir_path: str,
    frame_rate: isinstance,
    output_images: str,
    extension: str,
    # box_threshold: float,
    # text_threshold: float,
    # num_datasets: int,
    height: int = None,
    width: int = None,
    # use_yolo: bool = False,
    model_path: str = "trained_models/yolov8m_cf_4cam_verano_pies_v3.pt",
    confidence=0.4,
    iou=0.7,
    imgsz=1280,
    # val=True,
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

    # convert video to images
    video2images(video_path, image_dir_path, frame_rate, height, width)

    # Run autolabel con YOLO
    label_multiple_yolov8(
        model_path=model_path,
        input_folder=image_dir_path,
        output_folder=dataset_dir_path,
        confidence=confidence,
        iou=iou,
        imgsz=imgsz,
        tracking=True,
        # val_ratio=0.8,
    )

    print("pipeline ended")


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
        "--frame_rate", default=20, type=int, help="Tasa de cuadros por segundo"
    )
    parser.add_argument(
        "--output_dataset",
        default=config.DATASET_DIR_PATH,
        type=str,
        help="Carpeta de salida para el dataset de imagenes anotadas",
    )
    parser.add_argument(
        "--extension", default=".jpg", type=str, help="Extensión de archivo"
    )
    parser.add_argument(
        "--max_size", default=0.5, type=float, help="Tamaño maximo de BBox"
    )
    parser.add_argument(
        "--min_size", default=0.05, type=float, help="Tamaño minimo de BBox"
    )
    parser.add_argument(
        "--iou_threshold",
        default=0.4,
        type=float,
        help="IoU maxima permitaida de dos BBox",
    )
    parser.add_argument(
        "--not_remove_large",
        default=True,
        action="store_false",
        help="Colocar si no deseamos eliminar BBox demasiado grandes",
    )
    parser.add_argument(
        "--not_remove_small",
        default=True,
        action="store_false",
        help="Colocar si no deseamos eliminar BBox demasiado pequeños",
    )
    parser.add_argument(
        "--not_remove_overlapping",
        default=True,
        action="store_false",
        help="Colocar si no queremos eliminar BBox que se superpongan",
    )
    parser.add_argument(
        "--not_remove_empty",
        default=True,
        action="store_false",
        help="colocar si no queremos eliminar imágenes sin BBox detectados",
    )
    parser.add_argument(
        "--not_remove_multiple",
        default=True,
        action="store_false",
        help="Colocar si no queremos eliminar imágenes con más de un objeto detectado",
    )
    parser.add_argument(
        "--not_augment",
        default=True,
        action="store_false",
        help="Si se desea no aumentar el dataset",
    )
    parser.add_argument(
        "--augmented_for",
        default=4,
        type=int,
        help="Proporción en la que se aumentará el dataset",
    )
    parser.add_argument(
        "--height", default=None, type=int, help="Altura para resize de las imagenes"
    )
    parser.add_argument(
        "--width", default=None, type=int, help="Ancho para resize de las imagenes"
    )

    args = parser.parse_args()

    # Run pipeline
    # 1- Convertir videos a images
    # 2- Run autolabel para cada imagen (se genera un dataset por producto)
    # 3- Union de los Datasets de todos los productos (merged dataset)
    # 4- Procesar Merged_Dataset/train para eliminar errores de anotación
    # 5- Procesar Merged_Dataset/valid para eliminar errores de anotación
    # 6- Generar la aumentación de datos

    # Obtener las rutas de las carpetas de videos
    video_paths = get_video_folder_paths(args.video)

    # Crear directorio de salida para imágenes
    if not os.path.exists(args.image_dir):
        os.makedirs(args.image_dir)

    # Crear directorio de salida para dataset
    if not os.path.exists(args.output_dataset):
        os.makedirs(args.output_dataset)

    # run autolabels
    if len(video_paths) == 0:
        print("No se encontraron videos")
        return
    elif len(video_paths) == 1:
        mypipeline_track(
            video_paths[0],
            args.image_dir,
            args.frame_rate,
            # json.load(open(os.path.join(video_paths[0], "ontology.json"))),
            args.output_dataset,
            args.extension,
            # args.box_threshold,
            # args.text_threshold,
            # args.num_datasets,
            args.height,
            args.width,
            # args.use_yolo,
        )
    else:
        for video_path in tqdm(video_paths):
            mypipeline_track(
                video_path,
                args.image_dir,
                args.frame_rate,
                # json.load(open(os.path.join(video_path, "ontology.json"))),
                args.output_dataset,
                args.extension,
                # args.box_threshold,
                # args.text_threshold,
                # args.num_datasets,
                args.height,
                args.width,
                # args.use_yolo,
            )
    # Procesar Merged_Dataset/train para eliminar errores de anotación
    contenido = os.listdir(args.output_dataset)
    print("Contenido \n", contenido)

    for data_p in contenido:
        output_path = os.path.join(args.output_dataset, data_p)
        run_processing_dataset(
            os.path.join(output_path, "train", "images"),
            os.path.join(output_path, "train", "labels"),
            max_size=args.max_size,
            min_size=args.min_size,
            iou_threshold=args.iou_threshold,
            remove_empty=args.not_remove_empty,
            remove_large=args.not_remove_large,
            remove_small=args.not_remove_small,
            remove_overlapping=args.not_remove_overlapping,
            remove_multiple=args.not_remove_multiple,
        )


if __name__ == "__main__":
    main()
