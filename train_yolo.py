# Load Yolov5 model from ultralitics
from ultralytics import YOLO
import argparse
from src import config

# data_yaml = (
#    "D:/Alexander/Go2Future/Autolabel_roboflow/dataset/Augmented_Dataset/data.yaml"
# )

# Create a YOLOv5 object
# model = YOLO("yolov5su.pt")
model = YOLO("best.pt")


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
        pretrained=True,
        #optimizer="AdamW",
        #cos_lr=True,
        #resume=True,
    )


if __name__ == "__main__":
    main()
