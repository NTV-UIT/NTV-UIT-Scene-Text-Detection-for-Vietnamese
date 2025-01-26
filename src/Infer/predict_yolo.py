from ultralytics import YOLO
import numpy as np
def predict_yolov8x(image):
    # Load YOLOv8x model
    model = YOLO(r'E:\Docs\CVA\App\checkpoint\yolo\best.pt')  # Đường dẫn đến tệp trọng số
    results = model(image, conf=0.5)
    
    # Chuyển đổi bounding boxes về định dạng cần thiết
    dt_boxes = []
    for result in results[0].boxes.xyxy.cpu().numpy():
        x_min, y_min, x_max, y_max = map(int, result)
        dt_boxes.append(np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]))
    
    return dt_boxes, image
