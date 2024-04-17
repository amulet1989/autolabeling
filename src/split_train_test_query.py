import os
import random
import shutil


def split_dataset(dataset_path, output_path):
    # Crear directorios de salida
    output_train_path = os.path.join(output_path, "bounding_box_train")
    output_test_path = os.path.join(output_path, "bounding_box_test")
    output_query_path = os.path.join(output_path, "query")

    os.makedirs(output_train_path, exist_ok=True)
    os.makedirs(output_test_path, exist_ok=True)
    os.makedirs(output_query_path, exist_ok=True)

    # Obtener la lista de identificadores de las personas
    person_ids = [
        folder
        for folder in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, folder))
    ]

    # Dividir aleatoriamente las identidades para entrenamiento y prueba
    random.seed(0)
    random.shuffle(person_ids)

    # Asignar el ID sobrante al conjunto de entrenamiento si el número total de IDs es impar
    if len(person_ids) % 2 == 1:
        train_ids = person_ids[: len(person_ids) // 2 + 1]
        test_ids = person_ids[len(person_ids) // 2 + 1 :]
    else:
        train_ids = person_ids[: len(person_ids) // 2]
        test_ids = person_ids[len(person_ids) // 2 :]

    # Mover imágenes al directorio de entrenamiento o prueba
    for person_id in person_ids:
        person_folder = os.path.join(dataset_path, person_id)
        output_folder = (
            output_train_path if person_id in train_ids else output_test_path
        )

        for file_name in os.listdir(person_folder):
            file_path = os.path.join(person_folder, file_name)
            shutil.copy(file_path, os.path.join(output_folder, file_name))

    # Construir el set de imágenes de consulta
    for test_id in test_ids:
        person_folder = os.path.join(dataset_path, test_id)
        # print("Procesando", person_folder)
        query_images = set()

        for file_name in os.listdir(person_folder):
            # print("Archivo:", file_name)
            if "_c" in file_name:
                camera_id = file_name.split("_")[1]
                query_images.add(camera_id)

        # print("Cámaras de consulta:", query_images)
        for camera_ID in query_images:
            # Tomar una imagen por cámara y moverla al directorio de consultas
            query_candidates = [
                file for file in os.listdir(person_folder) if f"_{camera_ID}_" in file
            ]

            if query_candidates:
                query_image = random.choice(query_candidates)
                print("Copiando", query_image)
                shutil.copy(
                    os.path.join(person_folder, query_image),
                    os.path.join(output_query_path, query_image),
                )
            else:
                print(
                    f"No se encontraron imágenes para la cámara {camera_ID} en la carpeta {person_folder}"
                )


if __name__ == "__main__":
    dataset_path = "D:\Alexander\Go2Future\Autolabel_roboflow\dataset\Pilar_11cam_ReID_sec01\Pilar_11cam_ReID_sec01_market_format"  # Reemplazar con la ruta de tu dataset
    output_path = "D:\Alexander\Go2Future\Autolabel_roboflow\dataset\Pilar_11cam_ReID_sec01"  # Reemplazar con la ruta deseada para el nuevo dataset

    split_dataset(dataset_path, output_path)
    print("Proceso completado.")
