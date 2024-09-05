import supervision as sv
from src import config
import os
from typing import List
import shutil
import yaml
import cv2
import numpy as np
import matplotlib.pyplot as plt

# from scipy import stats

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
        # print(out_video_path)
        with sv.ImageSink(
            target_dir_path=image_dir_path,
            image_name_pattern=image_name_pattern,
        ) as sink:
            for image in sv.get_video_frames_generator(
                source_path=str(out_video_path),
                stride=frame_rate,
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


def merge_datasets(dataset_paths: List, output_path: str, val=True):
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
            if val:
                my_splits = ["train", "valid"]
            else:
                my_splits = ["train"]

            for split in my_splits:
                src_images_dir = os.path.join(dataset_path, split, "images")
                src_labels_dir = os.path.join(dataset_path, split, "labels")

                dest_images_dir = os.path.join(output_path, split, "images")
                os.makedirs(dest_images_dir, exist_ok=True)

                dest_labels_dir = os.path.join(output_path, split, "labels")
                os.makedirs(dest_labels_dir, exist_ok=True)

                # Crear directorios para las etiquetas de validación si no existen
                if split == "train":
                    os.makedirs(
                        os.path.join(output_path, "valid", "labels"), exist_ok=True
                    )

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
    print(merged_data["names"])
    # merged_data["names"] = list(set(merged_data["names"]))

    seen = set()
    unique_names = []

    for name in merged_data["names"]:
        if name not in seen:
            unique_names.append(name)
            seen.add(name)

    merged_data["names"] = unique_names

    merged_data["nc"] = len(merged_data["names"])
    print(merged_data["names"])

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


def seleccionar_video():
    root = tk.Tk()
    root.withdraw()  # Oculta la ventana principal de tkinter

    file_path = filedialog.askopenfilename(
        title="Seleccionar una imagen",
        filetypes=[
            (
                "Archivos de video",
                "*.mp4 *.avi *.mkv *.mov",
            )
        ],
    )

    if file_path:
        return file_path
    else:
        return None


# Función para calcular estadísticas de imágenes
# Función para calcular estadísticas de imágenes
def calcular_estadisticas_imagenes(directorio):
    # Listar archivos en el directorio
    archivos = os.listdir(directorio)

    # Inicializar listas para almacenar todos los píxeles de cada canal
    todos_canales_r = []
    todos_canales_g = []
    todos_canales_b = []
    anchos = []
    alturas = []

    # Iterar sobre cada imagen en el directorio
    for archivo in archivos:
        ruta_imagen = os.path.join(directorio, archivo)

        # Leer la imagen usando OpenCV
        imagen = cv2.imread(ruta_imagen)

        # Obtener dimensiones de la imagen
        alto, ancho, _ = imagen.shape

        # Almacenar anchos y largos
        anchos.append(ancho)
        alturas.append(alto)

        # Separar los canales de color
        canal_r = imagen[:, :, 0].flatten()
        canal_g = imagen[:, :, 1].flatten()
        canal_b = imagen[:, :, 2].flatten()

        # Agregar los canales a las listas de todos los canales
        todos_canales_r.extend(canal_r)
        todos_canales_g.extend(canal_g)
        todos_canales_b.extend(canal_b)

    # Convertir listas a matrices NumPy
    todos_canales_r = np.array(todos_canales_r)
    todos_canales_g = np.array(todos_canales_g)
    todos_canales_b = np.array(todos_canales_b)

    # Calcular estadísticas finales
    # color
    media_r = np.mean(todos_canales_r)
    min_r = np.min(todos_canales_r)
    max_r = np.max(todos_canales_r)
    std_r = np.std(todos_canales_r)
    media_g = np.mean(todos_canales_g)
    std_g = np.std(todos_canales_g)
    media_b = np.mean(todos_canales_b)
    std_b = np.std(todos_canales_b)
    # Tamaño
    # Anchos
    min_ancho = np.min(anchos)
    max_ancho = np.max(anchos)
    media_ancho = np.mean(anchos)
    median_ancho = np.median(anchos)
    std_ancho = np.std(anchos)
    # moda_ancho = stats.mode(anchos)
    # Alto
    min_alto = np.min(alturas)
    max_alto = np.max(alturas)
    media_alto = np.mean(alturas)
    median_alto = np.median(alturas)
    std_alto = np.std(alturas)
    # moda_alto = stats.mode(alturas)

    # Histogramas
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    # plt.hist(anchos, bins=10, color="blue", alpha=0.7)
    frecuencia, bins, _ = plt.hist(
        anchos,
        bins=30,
        color="blue",
    )  # alpha=0.7
    # Encontrar el índice del bin con la frecuencia más alta
    indice_max_frecuencia = np.argmax(frecuencia)
    # Determinar el valor del bin correspondiente al índice máximo de frecuencia
    bin_mas_repetido_ancho = bins[indice_max_frecuencia]
    plt.title("Histograma de anchos")
    plt.xlabel("Ancho")
    plt.ylabel("Frecuencia")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    # plt.hist(alturas, bins=10, color="green", alpha=0.7)
    frecuencia, bins, _ = plt.hist(
        alturas,
        bins=30,
        color="green",
    )  # alpha=0.7
    # Encontrar el índice del bin con la frecuencia más alta
    indice_max_frecuencia = np.argmax(frecuencia)
    # Determinar el valor del bin correspondiente al índice máximo de frecuencia
    bin_mas_repetido_alto = bins[indice_max_frecuencia]
    plt.title("Histograma de alturas")
    plt.xlabel("Altura")
    plt.ylabel("Frecuencia")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Imprimir resultados
    # Color
    print("Estadísticas de los canales RGB:")
    print("Canal Rojo - Min:", min_r, "Max:", max_r)
    print("Canal Rojo - Media:", media_r / 255, "Desviación estándar:", std_r / 255)
    print("Canal Verde - Media:", media_g / 255, "Desviación estándar:", std_g / 255)
    print("Canal Azul - Media:", media_b / 255, "Desviación estándar:", std_b / 255)
    print()
    # Tamaños
    # Anchos
    print("Valor mínimo del ancho:", min_ancho)
    print("Valor máximo del ancho:", max_ancho)
    print("Valor medio del ancho:", media_ancho)
    print("Valor mediana del ancho:", median_ancho)
    print("Valor std del ancho:", std_ancho)
    print("Bin mas repetido ancho:", bin_mas_repetido_ancho)
    # Largos
    print("Valor mínimo del alto:", min_alto)
    print("Valor máximo del alto:", max_alto)
    print("Valor medio del alto:", media_alto)
    print("Valor mediana del alto:", median_alto)
    print("Valor std del alto:", std_alto)
    print("Bin mas repetido alto:", bin_mas_repetido_alto)

    return {
        "color_r": {"media": media_r, "std": std_r},
        "color_g": {
            "media": media_g,
            "std": std_g,
        },
        "color_b": {
            "media": media_b,
            "std": std_b,
        },
        "ancho": {
            "min": min_ancho,
            "max": max_ancho,
            "media": media_ancho,
            "median": median_ancho,
            "std": std_ancho,
        },
        "alto": {
            "min": min_alto,
            "max": max_alto,
            "media": media_alto,
            "median": median_alto,
            "std": std_alto,
        },
    }


## Muestrear directorio de imágenes ##
def eliminar_cada_m(directorio, valor_muestreo):
    """
    Elimina las imágenes que no cumplen con el valor de muestreo.

    Args:
        directorio (str): Ruta al directorio que contiene las imágenes.
        valor_muestreo (int): Valor para muestrear las imágenes.
    """
    # Obtener una lista de todos los archivos en el directorio
    archivos = sorted(os.listdir(directorio))

    # Filtrar solo las imágenes (puedes ajustar las extensiones según tus necesidades)
    extensiones_validas = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    imagenes = [
        archivo for archivo in archivos if archivo.lower().endswith(extensiones_validas)
    ]

    # Iterar sobre las imágenes y eliminar las que no cumplen con el valor de muestreo
    for i, imagen in enumerate(imagenes):
        # Mantener solo la imagen cada n-ésimo valor de muestreo
        if (i + 1) % valor_muestreo != 0:
            os.remove(os.path.join(directorio, imagen))
            print(f"Imagen eliminada: {imagen}")
