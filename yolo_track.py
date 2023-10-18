from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
from src.util import seleccionar_video

# Load the YOLOv8 model
<<<<<<< HEAD
model = YOLO('yolov8n.pt')
classes=0
=======
model = YOLO("yolov5su.pt")
# classes = 0
>>>>>>> c177074a5e10a2b13b72aa0d12ec22da67214a9d

# Open the video file
video_path = seleccionar_video()
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
<<<<<<< HEAD
        results = model.track(frame, persist=True, classes=classes, imgsz=640)
=======
        results = model.track(frame, persist=True, imgsz=640)
>>>>>>> c177074a5e10a2b13b72aa0d12ec22da67214a9d

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
<<<<<<< HEAD
cv2.destroyAllWindows()
=======
cv2.destroyAllWindows()
>>>>>>> c177074a5e10a2b13b72aa0d12ec22da67214a9d
