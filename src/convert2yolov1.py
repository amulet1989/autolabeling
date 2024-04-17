import os
import shutil
import yaml
import zipfile
import argparse


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


def convert_to_yolov8_format(yolov1_zip_path, output_dir):
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
    for filename in os.listdir(os.path.join(yolov8_dir, "obj_Validation_data")):
        if filename.endswith(".jpg"):
            image_path = os.path.join(yolov8_dir, "obj_Validation_data", filename)
            label_filename = filename.replace(".jpg", ".txt")
            label_path = os.path.join(yolov8_dir, "obj_Validation_data", label_filename)

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
        data_yaml_file.write(f"val: {os.path.join(yolov8_dir, 'valid', 'images')}\n")

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
