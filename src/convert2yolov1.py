import os
import shutil
import yaml


def convert_to_yolov1_format(dataset_path):
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

    with open(os.path.join(yolov1_path, "Validation.txt"), "w") as validation_txt_file:
        for image_file in os.listdir(obj_validation_data_path):
            if image_file.endswith(".jpg"):
                validation_txt_file.write(f"data/obj_Validation_data/{image_file}\n")


if __name__ == "__main__":
    dataset_path = (
        "RUTA_DEL_DATASET_YOLOV8"  # Reemplaza con la ruta de tu dataset YOLOv8
    )
    convert_to_yolov1_format(dataset_path)
