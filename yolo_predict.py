import cv2
from ultralytics import YOLO, RTDETR
from src.util import seleccionar_video, seleccionar_imagen
import numpy as np
import os


model = YOLO(
    "trained_models/yolov8m_cf_caja_person_cart_640x480_v2.pt"
)  # yolov8m_640x480_cf_9cam_v44 / yolov8m_cf_caja_640x480_v18
# model = RTDETR("rtdetr-l.pt")  # rtdetr-l.pt


# Create VideoCapture object
# INPUT_VIDEO = seleccionar_video()
# INPUT_IMAGE = seleccionar_imagen()
# INPUT_VIDEO = "rtsp://admin:2Raffin001.@10.20.2.151"
# INPUT_VIDEO = "rtsp://admin:2Mini001.@181.164.198.186:9556/live1"
# INPUT_VIDEO = "rtsp://admin:2Mini001.@10.93.27.196/h263/ch1/sub/av_stream"


def show_inference():
    INPUT_VIDEO = seleccionar_video()
    cap = cv2.VideoCapture(INPUT_VIDEO)
    win_name = "Camera Preview"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    classes = 0
    results = model.track(
        source=INPUT_VIDEO,  # INPUT_VIDEO / INPUT_IMAGE
        stream=True,  # True
        save=False,
        conf=0.35,
        imgsz=640 ,  # 704 1280
        iou=0.7,
        # verbose=False,
        # classes=classes,
        show=True,
        tracker="bytetrack.yaml", # bytetrack.yaml, botsort.yaml
        persist=True,
        show_boxes=True,
        retina_masks=True,
    )   # generator of Results objects

    for r in results:
        boxes = r.boxes  # Boxes object for bbox outputs

        image = r.orig_img.copy()

        if boxes.cls.numel() > 0:
            classe = boxes.cls.tolist()
            label = r.names
            scores = boxes.conf.tolist()  # Confidence scores
######### hasta aca #########
#         # print("clases:", classe)
#         # print("scores:", scores)
#         # print("labels:", label)

#         # Draw BBoxes on the image
#         # for box, label, score in zip(boxes, labels, scores):
#         for i, box in enumerate(boxes.xyxy):
#             x1, y1, x2, y2 = map(int, box)  # box
#             color = (0, 255, 0)  # Green color
#             thickness = 2

#             cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

#             text = f"{label[int(classe[i])]} ({scores[i]:.2f})"
#             # print(text)

#             cv2.putText(
#                 image,
#                 text,
#                 (x1, y1 - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 1,
#                 color,
#                 thickness,
#             )

#     cv2.imshow(win_name, image)

#     # Wait for a key press and check the pressed key
#     key = cv2.waitKey(1)  # & 0xFF
#     if key == ord("q"):  # Press 'q' to exit
#         break
#     elif key == ord("n"):  # Press 'n' to show the next image
#         continue

# Release VideoCapture and destroy windows
    cap.release()
    cv2.destroyAllWindows()
######################################################################################
def discart_by_tracking():
    # Definir las regiones de interés (ROI)
    # Definir las regiones de interés (ROI) como polígonos
    roi_gondola = np.array([[173, 1], [192, 337], [1082, 370], [1111, 3]], dtype=np.int32)
    roi_persona = np.array([[190, 349], [221, 712], [1055, 713], [1081, 379]], dtype=np.int32)

    # Diccionario para almacenar el historial de objetos
    object_history = {}

    # Abrir el video
    INPUT_VIDEO = seleccionar_video()
    cap = cv2.VideoCapture(INPUT_VIDEO)
    frame_id=0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_id+=1
        # Realizar tracking con ByteTrack
        results = list (model.track(
            source=frame,  # INPUT_VIDEO / INPUT_IMAGE
            stream=True,  # True
            save=False,
            conf=0.15,
            imgsz=1280,  # 704
            iou=0.7,
            verbose=False,
            # classes=classes,
            show=False,
            tracker="bytetrack.yaml", # trained_models/bytetrack.yaml, botsort.yaml
            persist=True,
        # show_boxes=False,
        # retina_masks=True,
            ) )  # generator of Results objects

        # Extraer detecciones y IDs de ByteTrack
        if results[0].boxes.id is not None:
            tracked_objects = results[0].boxes.xyxy.cpu().numpy()  # Coordenadas (x1, y1, x2, y2)
            track_ids = results[0].boxes.id.int().cpu().numpy()  # IDs de los objetos

            for obj, obj_id in zip(tracked_objects, track_ids):
                x1, y1, x2, y2 = obj
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)  # Centro del objeto

                # # Registrar la primera aparición en la góndola usando polígonos
                # if obj_id not in object_history:
                #     if cv2.pointPolygonTest(roi_gondola, (cx, cy), False) >= 0:
                #         object_history[obj_id] = {"status": "en_gondola", "last_position": (cx, cy)}
                #         print(obj_id, {"status": "en_gondola", "last_position": (cx, cy)})

                # # Si el objeto estaba en la góndola y ahora está en la zona de la persona
                # if obj_id in object_history and object_history[obj_id]["status"] == "en_gondola":
                #     if cv2.pointPolygonTest(roi_persona, (cx, cy), False) >= 0:
                #         print(f"⚠️ Objeto {obj_id} ha sido sacado de la góndola!")
                #         object_history[obj_id]["status"] = "tomado"
                # Registrar la posición del objeto en el historial
                if obj_id not in object_history:
                    object_history[obj_id] = {"positions": [], "status": "desconocido"}

                object_history[obj_id]["positions"].append((cx, cy))

                # Verificar si el objeto apareció en la góndola
                if object_history[obj_id]["status"] == "desconocido":
                    if cv2.pointPolygonTest(roi_gondola, (cx, cy), False) >= 0:
                        object_history[obj_id]["status"] = "tomado"
                        print(f"⚠️ Objeto {obj_id} ha sido tomado (detectado en góndola).")

                # Si el objeto nunca apareció en la góndola, pero su trayectoria va en dirección al cliente
                
                if object_history[obj_id]["status"] == "desconocido" and len(object_history[obj_id]["positions"]) > 5:
                    start_x, start_y = object_history[obj_id]["positions"][0]  # Primer punto de la trayectoria
                    if start_y + 100 < cy:  # Movimiento de arriba hacia abajo (dirección de góndola a cliente)
                        if cv2.pointPolygonTest(roi_persona, (cx, cy), False) >= 0:
                            object_history[obj_id]["status"] = "tomado"
                            print(f"⚠️ Objeto {obj_id} ha sido tomado (trayectoria desde la góndola).")

                # Verificar si el objeto pasó de la góndola al cliente
                if object_history[obj_id]["status"] == "tomado":
                    if cv2.pointPolygonTest(roi_persona, (cx, cy), False) >= 0:
                        print(f"✅ Objeto {obj_id} confirmado en posesión del cliente.")
                                      
                    # Recortar el objeto de la imagen actual
                    x1c, y1c, x2c, y2c = map(int, obj[:4])
                    crop = frame[y1c:y2c, x1c:x2c]

                    # Guardar el crop con un nombre único
                    crop_filename = os.path.join("images", f"obj_{obj_id}_{frame_id}.jpg")
                    cv2.imwrite(crop_filename, crop)
        
                # Dibujar bounding box e ID del objeto
                color = (0, 255, 255)  # Amarillo
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f"ID: {obj_id}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                

        # Dibujar las ROIs en la imagen
        cv2.polylines(frame, [roi_gondola], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.polylines(frame, [roi_persona], isClosed=True, color=(255, 0, 0), thickness=2)
        
        # Mostrar el frame con las detecciones
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__=="__main__":
    show_inference()
    # discart_by_tracking()
