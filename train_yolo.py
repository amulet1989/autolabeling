# Load Yolov5 model from ultralitics
from ultralytics import YOLO
import argparse
from src import config

# data_yaml = (
#    "D:/Alexander/Go2Future/Autolabel_roboflow/dataset/Augmented_Dataset/data.yaml"
# )

# Create a YOLOv5 object
model = YOLO("yolov5su.pt")


def main():
    parser = argparse.ArgumentParser(
        description="Pipeline para procesar videos y detección de objetos"
    )
    parser.add_argument(
        "--data_yaml",
        default=config.DATA_YAML_PATH,
        type=str,
        help="Ruta al archivo de data_yaml",
    )
    parser.add_argument(
        "--epochs",
        default=50,
        type=int,
        help="Numero de epocas a entrenar",
    )
    parser.add_argument(
        "--batch",
        default=-1,
        type=int,
        help="Batch size",
    )
    parser.add_argument(
        "--device",
        default=None,
        type=str,
        help="Batch size",
    )
    args = parser.parse_args()

    # Train the model
    model.train(
        data=args.data_yaml,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        pretrained=False,
    )

    # pred = model.predict(
    #    source="D:/Alexander/Go2Future/Autolabel_roboflow/dataset/Augmented_Dataset/train/images/7790040133487_vertical_1-00000_aug_0.jpg",
    #    conf=0.25,
    # )


if __name__ == "__main__":
    main()
