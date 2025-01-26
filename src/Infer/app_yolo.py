import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from predict_yolo import predict_yolov8x

# Hàm vẽ bounding boxes lên hình ảnh
def draw_bounding_boxes(img, dt_boxes):
    for box in dt_boxes:
        # In giá trị của box để kiểm tra
        print(f"Box: {box}")
        
        # Chuyển đổi box thành 4 tọa độ x_min, y_min, x_max, y_max
        x_min = np.min(box[:, 0])  # Lấy tọa độ x nhỏ nhất
        y_min = np.min(box[:, 1])  # Lấy tọa độ y nhỏ nhất
        x_max = np.max(box[:, 0])  # Lấy tọa độ x lớn nhất
        y_max = np.max(box[:, 1])  # Lấy tọa độ y lớn nhất

        # Vẽ hình chữ nhật (bounding box)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    
    return img

# Hàm chính cho ứng dụng Streamlit
def main():
    st.title("Scene Text Detection With Yolov8")
    st.write("Upload an image to detect text regions.")

    # Tải ảnh từ người dùng
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Gọi mô hình PaddleOCR
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        dt_boxes, img = predict_yolov8x(image)  # Khởi tạo PaddleOCR
        
        # Vẽ bounding boxes lên ảnh
        processed_img = draw_bounding_boxes(image.copy(), dt_boxes)
        
        # Hiển thị ảnh với bounding boxes
        st.subheader("Detected Text Regions:")
        st.image(processed_img, caption="Image with Bounding Boxes", use_container_width=True, width=400)

if __name__ == "__main__":
    main()
