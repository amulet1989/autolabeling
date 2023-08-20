import argparse
from src import config
import json
from src.util import video2images, get_video_folder_paths
from src.model import autolabel_images
from tqdm import tqdm
import os


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
    # get all folders into video_dir

    video_paths = get_video_folder_paths(args.video)

    for video_path in tqdm(video_paths):
        # create image folder for each video
        image_dir_path = os.path.join(args.image_dir, os.path.basename(video_path))
        if not os.path.exists(image_dir_path):
            os.mkdir(image_dir_path)

        # convert video to images
        video2images(video_path, image_dir_path, args.frame_rate)

    # 2- Run autolabel model
    autolabel_images(
        input_folder=args.image_dir,
        ontology=json.load(open(args.ontology)),
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        output_folder=args.output_images,
        extension=args.extension,
    )


if __name__ == "__main__":
    main()
