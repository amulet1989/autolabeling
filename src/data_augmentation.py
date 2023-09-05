import os
import shutil
import yaml
import cv2
import albumentations as A
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)


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
            updated_bbox = str(bbox).replace(",", " ").replace("[", "").replace("]", "")
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
    image, bboxes, out_lab_pth, out_img_pth, transformed_file_name, classes, val=False
):
    if val:
        transform = A.Compose(
            [
                A.Resize(
                    always_apply=True,
                    p=1.0,
                    height=480,
                    width=640,
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
                    height=480,
                    width=640,
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0),
                A.CLAHE(clip_limit=(0, 1), tile_grid_size=(8, 8), always_apply=True),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0, rotate_limit=10, p=0.5
                ),
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
        print("label file is empty")


def augment_dataset(input_path: str, output_path: str, augmented_for: int = 10) -> None:
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
        file_name = img_file.split(".")[0]
        image = cv2.imread(os.path.join(inp_img_pth, img_file))
        lab_pth = os.path.join(inp_lab_pth, file_name + ".txt")
        album_bboxes = get_bboxes_list(lab_pth, CLASSES)

        for i in range(augmented_for):
            aug_file_name = f"{file_name}_{transformed_file_name}_{i}"
            apply_aug(
                image, album_bboxes, out_lab_pth, out_img_pth, aug_file_name, CLASSES
            )
    logging.info("Data train augmentation ended ...")

    # Transformando datos de validación (no aumenta solo rescala)
    imgs = os.listdir(inp_img_pth_valid)
    logging.info("Addjusting validation data  ...")
    for img_file in tqdm(imgs):
        file_name = img_file.split(".")[0]
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
        )
    logging.info("Data augmentation ended ...")
