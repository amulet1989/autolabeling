import os
import shutil
import yaml
import zipfile
import argparse


###################################
## Modleos datasets de segmentacion ##
#######################################
def ultralytics_to_cvat(ultralytics_dataset_path, output_path):
    # Crear las carpetas de salida
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, "images/Train"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "images/Validation"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "labels/Train"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "labels/Validation"), exist_ok=True)

    # Cargar el archivo data.yaml de Ultralytics
    with open(os.path.join(ultralytics_dataset_path, "data.yaml"), "r") as f:
        data = yaml.safe_load(f)

    class_names = data["names"]

    # Procesar Train
    train_image_dir = os.path.join(ultralytics_dataset_path, "train/images")
    train_label_dir = os.path.join(ultralytics_dataset_path, "train/labels")

    train_txt_path = os.path.join(output_path, "Train.txt")
    with open(train_txt_path, "w") as f_train:
        for image_name in os.listdir(train_image_dir):
            image_path = os.path.join("images/Train", image_name)
            f_train.write(f"{image_path}\n")
            # Copiar imágenes y etiquetas
            shutil.copy(
                os.path.join(train_image_dir, image_name),
                os.path.join(output_path, image_path),
            )
            label_name = image_name.replace(".jpg", ".txt").replace(".png", ".txt")
            shutil.copy(
                os.path.join(train_label_dir, label_name),
                os.path.join(output_path, "labels/Train", label_name),
            )

    # Procesar Validation (si existe)
    if "val" in data:
        val_image_dir = os.path.join(ultralytics_dataset_path, "valid/images")
        val_label_dir = os.path.join(ultralytics_dataset_path, "valid/labels")

        val_txt_path = os.path.join(output_path, "Validation.txt")
        with open(val_txt_path, "w") as f_val:
            for image_name in os.listdir(val_image_dir):
                image_path = os.path.join("images/Validation", image_name)
                f_val.write(f"{image_path}\n")
                # Copiar imágenes y etiquetas
                shutil.copy(
                    os.path.join(val_image_dir, image_name),
                    os.path.join(output_path, image_path),
                )
                label_name = image_name.replace(".jpg", ".txt").replace(".png", ".txt")
                shutil.copy(
                    os.path.join(val_label_dir, label_name),
                    os.path.join(output_path, "labels/Validation", label_name),
                )

    # Crear el archivo data.yaml para CVAT
    cvat_data = {
        "Train": "Train.txt",
        "Validation": "Validation.txt" if "val" in data else None,
        "names": {i: name for i, name in enumerate(class_names)},
        "path": ".",
    }

    with open(os.path.join(output_path, "data.yaml"), "w") as f:
        yaml.dump(cvat_data, f)


def cvat_to_ultralytics(cvat_dataset_path, output_path):
    # Crear las carpetas de salida
    os.makedirs(os.path.join(output_path, "train/images"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "train/labels"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "valid/images"), exist_ok=True)
    os.makedirs(os.path.join(output_path, "valid/labels"), exist_ok=True)

    # Cargar el archivo data.yaml de CVAT
    with open(os.path.join(cvat_dataset_path, "data.yaml"), "r") as f:
        data = yaml.safe_load(f)

    class_names = data["names"]
    output_rel_path = os.path.relpath(output_path)

    def replace_extension(image_name):
        # Reemplaza la extensión de la imagen con .txt
        return os.path.splitext(image_name)[0] + ".txt"

    # Procesar las imágenes y labels de Train
    with open(os.path.join(cvat_dataset_path, "Train.txt"), "r") as f:
        train_images = f.read().splitlines()

    for image_path in train_images:
        # Ajustar la ruta: eliminar "data/" del inicio de la ruta
        image_path = image_path.replace("data/", "")
        image_name = os.path.basename(image_path)
        label_name = replace_extension(image_name)

        # Nueva ruta para la imagen en el directorio de salida
        new_image_path = os.path.join(output_path, "train/images", image_name)

        # Copiar la imagen y el archivo de etiquetas
        shutil.copy(os.path.join(cvat_dataset_path, image_path), new_image_path)
        shutil.copy(
            os.path.join(cvat_dataset_path, "labels/Train", label_name),
            os.path.join(output_path, "train/labels", label_name),
        )

    # Procesar las imágenes y labels de Validation (si existen)
    validation_images_path = os.path.join(cvat_dataset_path, "Validation.txt")
    if os.path.exists(validation_images_path):
        with open(validation_images_path, "r") as f:
            validation_images = f.read().splitlines()

        for image_path in validation_images:
            # Ajustar la ruta: eliminar "data/" del inicio de la ruta
            image_path = image_path.replace("data/", "")
            image_name = os.path.basename(image_path)
            label_name = replace_extension(image_name)

            # Nueva ruta para la imagen en el directorio de salida
            new_image_path = os.path.join(output_path, "valid/images", image_name)

            # Copiar la imagen y el archivo de etiquetas
            shutil.copy(os.path.join(cvat_dataset_path, image_path), new_image_path)
            shutil.copy(
                os.path.join(cvat_dataset_path, "labels/Validation", label_name),
                os.path.join(output_path, "valid/labels", label_name),
            )

    # Crear el archivo data.yaml para Ultralytics
    ultralytics_data = {
        "names": class_names,
        "nc": len(class_names),
        "train": os.path.join(output_rel_path, "train"),
        "val": (
            os.path.join(output_rel_path, "valid")
            if os.path.exists(validation_images_path)
            else None
        ),
    }

    with open(os.path.join(output_path, "data.yaml"), "w") as f:
        yaml.dump(ultralytics_data, f)


###################################
## Modelos datasets de deteccion ##
#######################################
def convert_to_yolov1_format(dataset_path, with_val=True):
    # Crear una carpeta con el sufijo "_YOLOV1"
    yolov1_path = dataset_path + "_YOLOV1"
    os.makedirs(yolov1_path, exist_ok=True)

    # Leer las clases y el número de clases desde data.yaml
    data_yaml_path = os.path.join(dataset_path, "data.yaml")
    with open(data_yaml_path, "r") as yaml_file:
        data = yaml.safe_load(yaml_file)
    classes = data["names"]
    num_classes = data["nc"]

    # Copiar obj.names a la carpeta YOLOv1
    with open(os.path.join(yolov1_path, "obj.names"), "w") as obj_names_file:
        for class_name in classes:
            obj_names_file.write(f"{class_name}\n")

    # Crear obj.data
    if with_val:
        with open(os.path.join(yolov1_path, "obj.data"), "w") as obj_data_file:
            obj_data_file.write(f"classes = {num_classes}\n")
            obj_data_file.write(f"Validation = data/Validation.txt\n")
            obj_data_file.write(f"train = data/Train.txt\n")
            obj_data_file.write(f"names = data/obj.names\n")
            obj_data_file.write("backup = backup/\n")

        # Crear las carpetas obj_Train_data y obj_Validation_data
        obj_train_data_path = os.path.join(yolov1_path, "obj_Train_data")
        obj_validation_data_path = os.path.join(yolov1_path, "obj_Validation_data")
        os.makedirs(obj_train_data_path, exist_ok=True)
        os.makedirs(obj_validation_data_path, exist_ok=True)

        # Copiar imágenes y etiquetas de la carpeta "valid" del dataset original
        valid_images_path = os.path.join(dataset_path, "valid", "images")
        valid_labels_path = os.path.join(dataset_path, "valid", "labels")

        for image_file in os.listdir(valid_images_path):
            if image_file.endswith(".jpg"):
                image_path = os.path.join(valid_images_path, image_file)
                shutil.copy(image_path, obj_validation_data_path)

                label_file = image_file.replace(".jpg", ".txt")
                label_path = os.path.join(valid_labels_path, label_file)
                shutil.copy(label_path, obj_validation_data_path)

        with open(
            os.path.join(yolov1_path, "Validation.txt"), "w"
        ) as validation_txt_file:
            for image_file in os.listdir(obj_validation_data_path):
                if image_file.endswith(".jpg"):
                    validation_txt_file.write(
                        f"data/obj_Validation_data/{image_file}\n"
                    )

    else:
        with open(os.path.join(yolov1_path, "obj.data"), "w") as obj_data_file:
            obj_data_file.write(f"classes = {num_classes}\n")
            # obj_data_file.write(f"Validation = data/Validation.txt\n")
            obj_data_file.write(f"train = data/Train.txt\n")
            obj_data_file.write(f"names = data/obj.names\n")
            obj_data_file.write("backup = backup/\n")

        # Crear las carpetas obj_Train_data y obj_Validation_data
        obj_train_data_path = os.path.join(yolov1_path, "obj_Train_data")
        # obj_validation_data_path = os.path.join(yolov1_path, "obj_Validation_data")
        os.makedirs(obj_train_data_path, exist_ok=True)
        # os.makedirs(obj_validation_data_path, exist_ok=True)

        # Copiar imágenes y etiquetas de la carpeta "valid" del dataset original
        # valid_images_path = os.path.join(dataset_path, "valid", "images")
        # valid_labels_path = os.path.join(dataset_path, "valid", "labels")

        # for image_file in os.listdir(valid_images_path):
        #     if image_file.endswith(".jpg"):
        #         image_path = os.path.join(valid_images_path, image_file)
        #         shutil.copy(image_path, obj_validation_data_path)

        #         label_file = image_file.replace(".jpg", ".txt")
        #         label_path = os.path.join(valid_labels_path, label_file)
        #         shutil.copy(label_path, obj_validation_data_path)

    # Leer y copiar imágenes y archivos de etiquetas de la carpeta "train" del dataset original
    for image_file in os.listdir(os.path.join(dataset_path, "train", "images")):
        if image_file.endswith(".jpg"):
            image_path = os.path.join(dataset_path, "train", "images", image_file)
            shutil.copy(image_path, obj_train_data_path)

            label_file = image_file.replace(".jpg", ".txt")
            label_path = os.path.join(dataset_path, "train", "labels", label_file)
            shutil.copy(label_path, obj_train_data_path)

    # Crear Train.txt y Validation.txt
    with open(os.path.join(yolov1_path, "Train.txt"), "w") as train_txt_file:
        for image_file in os.listdir(obj_train_data_path):
            if image_file.endswith(".jpg"):
                train_txt_file.write(f"data/obj_Train_data/{image_file}\n")

    # Comprimir la carpeta "_YOLOV1"
    shutil.make_archive(yolov1_path, "zip", yolov1_path)
    shutil.rmtree(yolov1_path)


def convert_to_yolov8_format(yolov1_zip_path, output_dir, val=True):
    # Obtener el nombre del archivo ZIP sin extensión
    yolov1_dir_name = os.path.splitext(os.path.basename(yolov1_zip_path))[0]

    # Crear una carpeta con el mismo nombre más un sufijo en la carpeta de salida
    yolov8_dir = os.path.join(output_dir, yolov1_dir_name + "_YOLOV8")
    os.makedirs(yolov8_dir, exist_ok=True)

    # Extraer el archivo ZIP en la carpeta recién creada
    with zipfile.ZipFile(yolov1_zip_path, "r") as zip_ref:
        zip_ref.extractall(yolov8_dir)

    # Rutas de directorios YOLOv8
    data_dir = os.path.join(yolov8_dir, "data")
    train_images_dir = os.path.join(yolov8_dir, "train", "images")
    train_labels_dir = os.path.join(yolov8_dir, "train", "labels")
    valid_images_dir = os.path.join(yolov8_dir, "valid", "images")
    valid_labels_dir = os.path.join(yolov8_dir, "valid", "labels")

    # Crear directorios necesarios
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    if val:
        os.makedirs(valid_images_dir, exist_ok=True)
        os.makedirs(valid_labels_dir, exist_ok=True)

        # Mover imágenes y etiquetas desde obj_Train_data a train/images y train/labels
    for filename in os.listdir(os.path.join(yolov8_dir, "obj_train_data")):
        if filename.endswith(".jpg"):
            image_path = os.path.join(yolov8_dir, "obj_train_data", filename)
            label_filename = filename.replace(".jpg", ".txt")
            label_path = os.path.join(yolov8_dir, "obj_train_data", label_filename)

            # Mover imágenes
            shutil.move(image_path, os.path.join(train_images_dir, filename))

            # Mover etiquetas
            shutil.move(label_path, os.path.join(train_labels_dir, label_filename))

    # Mover imágenes y etiquetas desde obj_Validation_data a valid/images y valid/labels
    if val:
        for filename in os.listdir(os.path.join(yolov8_dir, "obj_Validation_data")):
            if filename.endswith(".jpg"):
                image_path = os.path.join(yolov8_dir, "obj_Validation_data", filename)
                label_filename = filename.replace(".jpg", ".txt")
                label_path = os.path.join(
                    yolov8_dir, "obj_Validation_data", label_filename
                )

                # Mover imágenes
                shutil.move(image_path, os.path.join(valid_images_dir, filename))

                # Mover etiquetas
                shutil.move(label_path, os.path.join(valid_labels_dir, label_filename))

    # Copiar data.names a obj.names
    data_names_path = os.path.join(yolov8_dir, "obj.names")
    obj_names_path = os.path.join(data_dir, "obj.names")
    shutil.copy(data_names_path, obj_names_path)

    # Crear data.yaml
    # Obtener nombres de clases
    obj_names_path = os.path.join(yolov8_dir, "data", "obj.names")
    with open(obj_names_path, "r") as obj_names_file:
        class_names = [line.strip() for line in obj_names_file]

    # Crear data.yaml con nc y paths actualizados
    data_yaml_path = os.path.join(yolov8_dir, "data.yaml")
    with open(data_yaml_path, "w") as data_yaml_file:
        data_yaml_file.write("names:\n")
        for class_name in class_names:
            data_yaml_file.write(f"  - {class_name}\n")
        data_yaml_file.write(f"nc: {len(class_names)}\n")
        data_yaml_file.write(f"train: {os.path.join(yolov8_dir, 'train', 'images')}\n")
        if val:
            data_yaml_file.write(
                f"val: {os.path.join(yolov8_dir, 'valid', 'images')}\n"
            )

    # Mover data.yaml a output_dir
    # data_yaml_path = os.path.join(yolov8_dir, "data", "data.yaml")
    # shutil.move(data_yaml_path, os.path.join(yolov8_dir, "data.yaml"))

    # Eliminar archivos y directorios innecesarios
    for item in os.listdir(yolov8_dir):
        item_path = os.path.join(yolov8_dir, item)
        if os.path.isfile(item_path) and item != "data.yaml":
            os.remove(item_path)
        elif os.path.isdir(item_path) and item not in ["train", "valid"]:
            shutil.rmtree(item_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert YOLOv8 dataset to YOLOv1 format"
    )
    parser.add_argument(
        "--dataset_path",
        default="dataset",
        type=str,
        help="Ruta al archivo de video",
    )

    args = parser.parse_args()

    for dataset_name in os.listdir(args.dataset_path):
        dataset_path = os.path.join(args.dataset_path, dataset_name)

        # Verificar si el elemento en DATASET_DIR_PATH es un directorio
        if os.path.isdir(dataset_path):
            print(f"Procesando el dataset: {dataset_name}")
            convert_to_yolov1_format(dataset_path)

        print("Proceso de conversión completado.")
