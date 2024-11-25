import os
import shutil
import yaml
import cv2
import albumentations as A
from tqdm import tqdm
import random
import math
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)


#### Calse custom para generar patch aleatorios en escala de grises #########
class LocalGrayscalePatchReplacement(A.ImageOnlyTransform):
    def __init__(
        self, probability=0.2, sl=0.02, sh=0.4, r1=0.3, always_apply=False, p=1.0
    ):
        super().__init__(always_apply, p)
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def apply(self, img, **params):
        if random.uniform(0, 1) >= self.probability:
            return img

        height, width = img.shape[0], img.shape[1]

        for attempt in range(200):
            area = height * width
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < width and h < height:
                x1 = random.randint(0, width - w)
                y1 = random.randint(0, height - h)

                # Convertir la región seleccionada a escala de grises
                patch_gray = cv2.cvtColor(
                    img[y1 : y1 + h, x1 : x1 + w], cv2.COLOR_RGB2GRAY
                )
                # Convertir la región de escala de grises a imagen de 3 canales
                patch_gray_rgb = cv2.cvtColor(patch_gray, cv2.COLOR_GRAY2RGB)

                # Reemplazar el parche en la imagen original
                img[y1 : y1 + h, x1 : x1 + w] = patch_gray_rgb

                return img

        return img

    def get_transform_init_args_names(self):
        return ("probability", "sl", "sh", "r1")


def single_obj_bb_yolo_conversion(transformed_bboxes, class_names):
    if len(transformed_bboxes):
        class_num = class_names.index(transformed_bboxes[-1])
        bboxes = list(transformed_bboxes)[:-1]  # .insert(0, '0')
        bboxes.insert(0, class_num)
    else:
        bboxes = []
    return bboxes


def multi_obj_bb_yolo_conversion(aug_labs, class_names):
    yolo_labels = []
    for aug_lab in aug_labs:
        bbox = single_obj_bb_yolo_conversion(aug_lab, class_names)
        yolo_labels.append(bbox)
    return yolo_labels


def save_aug_lab(transformed_bboxes, lab_pth, lab_name):
    lab_out_pth = os.path.join(lab_pth, lab_name)
    with open(lab_out_pth, "w") as output:
        for bbox in transformed_bboxes:
            # updated_bbox = str(bbox).replace(",", "").replace("[", "").replace("]", "")
            updated_bbox = " ".join([str(int(bbox[0]))] + [str(float(coord)) for coord in bbox[1:]])
            output.write(updated_bbox + "\n")


def save_aug_image(transformed_image, out_img_pth, img_name):
    out_img_path = os.path.join(out_img_pth, img_name)
    cv2.imwrite(out_img_path, transformed_image)


def get_album_bb_list(yolo_bbox, class_names):
    album_bb = []
    str_bbox_list = yolo_bbox.split(" ")
    for index, value in enumerate(str_bbox_list):
        if index == 0:  # class number is placed at index 0
            class_name = class_names[int(value)]
        else:
            album_bb.append(float(value))
    album_bb.append(class_name)  # [x_center, y_center, width, height, class_name]
    return album_bb


def get_album_bb_lists(yolo_str_labels, classes):
    album_bb_lists = []
    yolo_list_labels = yolo_str_labels.split("\n")
    for yolo_str_label in yolo_list_labels:
        if len(yolo_str_label) > 0:
            album_bb_list = get_album_bb_list(yolo_str_label, classes)
            album_bb_lists.append(album_bb_list)
    return album_bb_lists


def get_bboxes_list(inp_lab_pth, classes):
    yolo_str_labels = open(inp_lab_pth, "r").read()
    if yolo_str_labels:
        if "\n" in yolo_str_labels:
            # print("multi-objs")
            album_bb_lists = get_album_bb_lists(yolo_str_labels, classes)
        else:
            # print("single line ")
            album_bb_lists = get_album_bb_list(yolo_str_labels, classes)
            album_bb_lists = [
                album_bb_lists
            ]  # require 2d list in alumbentation function
    else:
        # print("No object")
        album_bb_lists = []
    return album_bb_lists


def apply_aug(
    image,
    bboxes,
    out_lab_pth,
    out_img_pth,
    transformed_file_name,
    classes,
    val=False,
    height=480,  # 576,
    width=640,  # 704,
):
    if val:
        transform = A.Compose(
            [
                A.Resize(
                    always_apply=True,
                    p=1.0,
                    height=height,  # 480,
                    width=width,  # 640,
                ),
            ],
            bbox_params=A.BboxParams(format="yolo"),
        )
    else:
        transform = A.Compose(
            [
                A.Resize(
                    always_apply=True,
                    p=1.0,
                    height=height,
                    width=width,
                ),
                A.HorizontalFlip(always_apply=False, p=0.5),
                # A.VerticalFlip(p=0.2),
                # A.RandomBrightnessContrast(always_apply=False, p=0.3),
                A.RandomBrightnessContrast(
                    always_apply=False, brightness_limit=0.2, contrast_limit=0, p=0.3
                ),
                # A.CLAHE(
                #     always_apply=False, clip_limit=(0, 1), tile_grid_size=(8, 8), p=0.3
                # ),
                # A.ShiftScaleRotate(
                #     always_apply=False,
                #     p=0.2,
                #     shift_limit_x=(-0.02, 0.02),
                #     shift_limit_y=(-0.02, 0.02),
                #     scale_limit=(-0.09999999999999998, 0.10000000000000009),
                #     rotate_limit=(-5, 5),
                #     interpolation=1,
                #     border_mode=2,
                #     value=(0, 0, 0),
                #     mask_value=None,
                #     rotate_method="largest_box",
                # ),
                A.RandomToneCurve(always_apply=False, p=0.3, scale=0.1),
                # A.ChannelShuffle(always_apply=False, p=0.3),
                # A.Blur(always_apply=False, p=0.5, blur_limit=(1, 3)),
                A.MotionBlur(
                    always_apply=False, p=0.2, blur_limit=(3, 7), allow_shifted=True
                ),
                A.AdvancedBlur(
                    always_apply=False,
                    p=0.1,
                    blur_limit=(3, 7),
                    sigmaX_limit=(0.2, 1.0),
                    sigmaY_limit=(0.2, 1.0),
                    rotate_limit=(-90, 90),
                    beta_limit=(0.5, 8.0),
                    noise_limit=(0.9, 1.1),
                ),
                # A.Downscale(always_apply=False, p=0.3, scale_min=0.5, scale_max=0.99),
                A.ToGray(always_apply=False, p=0.1),
                LocalGrayscalePatchReplacement(probability=1.0, p=0.4),
            ],
            bbox_params=A.BboxParams(format="yolo"),
        )
    transformed = transform(image=image, bboxes=bboxes)
    transformed_image = transformed["image"]
    transformed_bboxes = transformed["bboxes"]
    tot_objs = len(bboxes)
    if tot_objs != 0:
        transformed_bboxes = multi_obj_bb_yolo_conversion(transformed_bboxes, classes)
        save_aug_lab(transformed_bboxes, out_lab_pth, transformed_file_name + ".txt")
        save_aug_image(transformed_image, out_img_pth, transformed_file_name + ".jpg")
        # draw_yolo(transformed_image, transformed_bboxes)
    else:
        save_aug_image(transformed_image, out_img_pth, transformed_file_name + ".jpg")
        save_aug_lab(transformed_bboxes, out_lab_pth, transformed_file_name + ".txt")
        print("label file is empty")


def augment_dataset(
    input_path: str,
    output_path: str,
    just_rezize: bool = False,
    augmented_for: int = 10,
    height: int = 480,  # 576, height, width
    width: int = 640,  # 704
    val: bool = True,
) -> None:
    """
    Aplica transformaciones a los datos de entrada.
    Args:
        input_path (str): Path de entrada.
        output_path (str): Path de salida.
        augmented_for (int): Cantidad de veces que se aplicaran las transformaciones.
    Returns:
        None.
    """

    inp_img_pth = os.path.join(input_path, "train", "images")
    inp_lab_pth = os.path.join(input_path, "train", "labels")
    inp_img_pth_valid = os.path.join(input_path, "valid", "images")
    inp_lab_pth_valid = os.path.join(input_path, "valid", "labels")

    out_img_pth_valid = os.path.join(output_path, "valid", "images")
    out_lab_pth_valid = os.path.join(output_path, "valid", "labels")
    out_img_pth = os.path.join(output_path, "train", "images")
    out_lab_pth = os.path.join(output_path, "train", "labels")

    # create output folders if not exis
    if not os.path.exists(out_img_pth):
        os.makedirs(out_img_pth)
    if not os.path.exists(out_lab_pth):
        os.makedirs(out_lab_pth)

    if val:
        if not os.path.exists(out_img_pth_valid):
            os.makedirs(out_img_pth_valid)
        if not os.path.exists(out_lab_pth_valid):
            os.makedirs(out_lab_pth_valid)

    # Copy valid dataset to output folder
    # logging.info("Coping valid dataset ...")
    # for file in os.listdir(inp_img_pth_valid):
    #    shutil.copy(
    #        os.path.join(inp_img_pth_valid, file),
    #        os.path.join(out_img_pth_valid, file),
    #    )
    # for file in os.listdir(inp_lab_pth_valid):
    #    shutil.copy(
    #        os.path.join(inp_lab_pth_valid, file),
    #        os.path.join(out_lab_pth_valid, file),
    #    )

    logging.info("Transforming data.yaml ...")
    # Abrir el archivo YAML
    with open(os.path.join(input_path, "data.yaml"), "r") as file:
        # Cargar el contenido del archivo en una variable
        data = yaml.load(file, Loader=yaml.FullLoader)

    # Actualizar los valores de las claves "train" y "val"
    data["train"] = os.path.join(output_path, "train", "images")

    if val:
        data["val"] = os.path.join(output_path, "valid", "images")

    # Guardar el archivo YAML actualizado
    with open(os.path.join(output_path, "data.yaml"), "w") as file:
        yaml.dump(data, file)

    transformed_file_name = "aug"

    with open(os.path.join(input_path, "data.yaml"), "r") as stream:
        data = yaml.safe_load(stream)

    CLASSES = data["names"]

    # Aumentando datos de train
    imgs = os.listdir(inp_img_pth)
    logging.info("Generating train augmented images ...")
    for img_file in tqdm(imgs):
        # file_name = img_file.split(".")[0]
        file_name = os.path.splitext(img_file)[0]
        image = cv2.imread(os.path.join(inp_img_pth, img_file))
        lab_pth = os.path.join(inp_lab_pth, file_name + ".txt")
        # print(lab_pth)
        album_bboxes = get_bboxes_list(lab_pth, CLASSES)

        for i in range(augmented_for):
            aug_file_name = f"{file_name}_{transformed_file_name}_{i}"
            apply_aug(
                image,
                album_bboxes,
                out_lab_pth,
                out_img_pth,
                aug_file_name,
                CLASSES,
                val=just_rezize,
                height=height,
                width=width,
            )
    logging.info("Data train augmentation ended ...")

    # Transformando datos de validación (no aumenta solo rescala)
    if val:
        imgs = os.listdir(inp_img_pth_valid)
        logging.info("Addjusting validation data  ...")
        for img_file in tqdm(imgs):
            # file_name = img_file.split(".")[0]
            file_name = os.path.splitext(img_file)[0]
            image = cv2.imread(os.path.join(inp_img_pth_valid, img_file))
            lab_pth = os.path.join(inp_lab_pth_valid, file_name + ".txt")
            album_bboxes = get_bboxes_list(lab_pth, CLASSES)

            aug_file_name = f"{file_name}_{transformed_file_name}"
            apply_aug(
                image,
                album_bboxes,
                out_lab_pth_valid,
                out_img_pth_valid,
                aug_file_name,
                CLASSES,
                val=True,
                height=height,
                width=width,
            )
    logging.info("Data augmentation ended ...")


############## Aumentar dataset ReID ####################
def augment_images_reid(input_dir, output_dir, num_augmentations=3):
    # Lista de transformaciones de aumentación que deseas aplicar
    transform = A.Compose(
        [
            # A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=1.0),
            # A.Rotate(limit=30, p=0.5),
            # A.Blur(always_apply=False, p=0.5, blur_limit=(1, 3)),
            # A.AdvancedBlur(
            #     always_apply=False,
            #     p=1.0,
            #     blur_limit=(3, 7),
            #     sigmaX_limit=(0.2, 1.0),
            #     sigmaY_limit=(0.2, 1.0),
            #     rotate_limit=(-90, 90),
            #     beta_limit=(0.5, 8.0),
            #     noise_limit=(0.9, 1.1),
            # ),
            # A.Flip(always_apply=False, p=0.5),
            A.HorizontalFlip(always_apply=False, p=0.5),
            A.MotionBlur(
                always_apply=False, p=0.2, blur_limit=(3, 7), allow_shifted=True
            ),
            A.RandomScale(
                always_apply=False,
                p=0.5,
                interpolation=0,
                scale_limit=(-0.09999999999999998, 0.10000000000000009),
            ),
            A.Perspective(
                always_apply=False,
                p=0.5,
                scale=(0.05, 0.1),
                keep_size=0,
                pad_mode=0,
                pad_val=(0, 0, 0),
                mask_pad_val=0,
                fit_output=0,
                interpolation=0,
            ),
            A.ToGray(always_apply=False, p=0.1),
            LocalGrayscalePatchReplacement(probability=1.0, always_apply=False, p=0.4),
        ]
    )

    # Crear el directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)

    # Iterar sobre todas las imágenes en el directorio de entrada
    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg"):
            parts = filename.split("_")
            img_id = parts[0]
            camara = parts[1]
            secuencia = parts[2][:-4]
            # Leer la imagen
            image = cv2.imread(os.path.join(input_dir, filename))

            # Copiar la imagen original al directorio de salida
            shutil.copyfile(
                os.path.join(input_dir, filename), os.path.join(output_dir, filename)
            )

            # Aplicar transformaciones de aumentación varias veces
            for i in range(num_augmentations):
                augmented = transform(image=image)
                augmented_image = augmented["image"]
                # Generar nuevo nombre de archivo para la imagen aumentada
                new_filename = f"{img_id}_{camara}_{secuencia}_{i}.jpg"
                # Guardar la imagen aumentada en el directorio de salida
                cv2.imwrite(os.path.join(output_dir, new_filename), augmented_image)

                print(f"Imagen aumentada guardada: {new_filename}")
