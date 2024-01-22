import os
from xml.etree import ElementTree as ET
from PIL import Image
import argparse


def parse_cvat_annotations(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    annotations = []

    for image_elem in root.findall(".//image"):
        image_data = {
            "id": int(image_elem.get("id")),
            "name": image_elem.get("name"),
            # "subset": image_elem.get("subset"),
            # "task_id": int(image_elem.get("task_id")),
            "width": int(image_elem.get("width")),
            "height": int(image_elem.get("height")),
            "boxes": [],
        }

        for box_elem in image_elem.findall(".//box"):
            box_data = {
                "label": box_elem.get("label"),
                "person_id": int(box_elem.find(".//attribute[@name='person_id']").text),
                "camera_id": int(box_elem.find(".//attribute[@name='camera_id']").text),
                "secuence_id": int(
                    box_elem.find(".//attribute[@name='secuence_id']").text
                ),
                "xtl": float(box_elem.get("xtl")),
                "ytl": float(box_elem.get("ytl")),
                "xbr": float(box_elem.get("xbr")),
                "ybr": float(box_elem.get("ybr")),
            }
            image_data["boxes"].append(box_data)

        annotations.append(image_data)

    return annotations


def process_cvat_annotations(cvat_annotations, output_folder, path_to_images):
    for image_data in cvat_annotations:
        for box_data in image_data["boxes"]:
            person_id = str(box_data["person_id"]).zfill(4)
            secuence_id = f's{str(box_data["person_id"]).zfill(3)}'
            camera_id = f'c{str(box_data["camera_id"])}'
            image_id = str(image_data["id"]).zfill(6)

            person_folder = os.path.join(output_folder, person_id)
            os.makedirs(person_folder, exist_ok=True)

            image_name = f"{person_id}_{camera_id}{secuence_id}_{image_id}.jpg"
            image_path = os.path.join(person_folder, image_name)

            image = Image.open(os.path.join(path_to_images, image_data["name"]))
            bbox = (
                int(box_data["xtl"]),
                int(box_data["ytl"]),
                int(box_data["xbr"]),
                int(box_data["ybr"]),
            )
            cropped_image = image.crop(bbox)
            cropped_image.save(image_path)


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline para procesar videos y detección de objetos"
    )
    parser.add_argument(
        "--xml_dir",
        default="./dataset/cvat_dataset",
        type=str,
        help="Ruta al archivo de video",
    )
    parser.add_argument(
        "--output_dir",
        default="default",
        type=str,
        help="Ruta al directorio de imágenes",
    )
    args = parser.parse_args()

    xml_dir = (
        args.xml_dir
    )  # "D:\Alexander\Go2Future\Autolabel_roboflow\dataset\cvat_3cam_3id"
    xml_file = "annotations.xml"
    xml_path = os.path.join(xml_dir, xml_file)
    print(xml_path)

    # obtener el nombre de la carpeta de xml_dir
    output_folder = os.path.basename(xml_dir)

    if args.output_dir == "default":
        output_dir = os.path.join(xml_dir, f"{output_folder}_market_format")
    else:
        args.output_dir = os.path.join(output_dir, f"{output_folder}_market_format")

    cvat_annotations = parse_cvat_annotations(xml_path)
    process_cvat_annotations(
        cvat_annotations,
        output_dir,
        path_to_images=os.path.join(xml_dir, "images", "train"),
    )


if __name__ == "__main__":
    main()
