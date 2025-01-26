import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from predict import predict

# Hàm vẽ bounding boxes lên hình ảnh
def draw_bounding_boxes(img, dt_boxes):
    for box in dt_boxes:
        points = box.astype(int)
        cv2.polylines(img, [points], isClosed=True, color=(0, 255, 0), thickness=1)
    return img

# Hàm chính cho ứng dụng Streamlit
def main():
    st.title("Scene Text Detection with DBNet")
    st.write("Upload an image to detect text regions.")

    # Tải ảnh từ người dùng
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Gọi mô hình PaddleOCR
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        dt_boxes, img = predict(image)  # Khởi tạo PaddleOCR
        
        # Vẽ bounding boxes lên ảnh
        processed_img = draw_bounding_boxes(image.copy(), dt_boxes)
        
        # Hiển thị ảnh với bounding boxes
        st.subheader("Detected Text Regions:")
        st.image(processed_img, caption="Image with Bounding Boxes", use_container_width=True, width=400)

if __name__ == "__main__":
    main()
