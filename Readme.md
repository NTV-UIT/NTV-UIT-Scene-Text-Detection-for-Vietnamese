# APP - Scene Text Detection Using YOLOv8 And DBNet

## 📋 **Giới thiệu**
Đồ án này giới thiệu về bài toán Scene Text Detection sử dụng YOLOv8 và DBNet

## Các thành viên nhóm:
1. Nguyễn Thế Vĩnh - 22521677  
2. Nguyễn Xuân Linh - 22520775

---

## 🗂 **Cấu trúc thư mục**
```plaintext
APP/
│
├── checkpoint/          # Checkpoint lưu trạng thái huấn luyện mô hình
├── data/                # Thư mục chứa dữ liệu đầu vào
├── src/                 # Thư mục chứa mã nguồn chính
│   ├── Eval/            # Script đánh giá hiệu suất mô hình
│   │   ├── eval.py
│   │   └── predict_db.py
│   ├── Infer/           # Script thực hiện suy luận (Inference)
│   │   ├── app_dbnet.py # File code app DBNet
│   │   ├── app_yolo.py  # File code app YOLO
│   │   ├── predict_yolo.py # File predict YOLO
│   │   ├── predict.py   # File predict DBNet
│   │   ├── log_output/  # File log của quá trình suy luận
│   │   └── ppocr/           # Tích hợp thư viện DBNet
│   └── Training/        # Script huấn luyện mô hình
│       ├── DBNet/       
│       └── YOLOV8/
│
├── .gitignore           
└── README.md            # Hướng dẫn cài đặt và sử dụng
```

---

## ⚙️ **Chức năng**
### 1. **Huấn luyện mô hình**
- **DBNet:**
  ```bash
  cd src/Training/DBNet
  ```
  Thực hiện chạy file `dbnet-trainning.ipynb` với từng config ở folder `config`. 
- **YOLOv8:**
  ```bash
  cd src/Training/YOLOv8
  ```
  Thực hiện chạy các Folder `Method 1 - Only BKAI Dataset`, `Method 2 - Finetuning với 3 tập dữ liệu Total Icdar2015 BKAI`, `Xử lý dữ liệu YOLOv8`.

### 2. **Suy luận (Inference)**
- **Infer (DBNet):**
  ```bash
  cd src/Infer
  streamlit run app_dbnet.py
  ```

- **Infer (YOLO):**
  ```bash
  cd src/Infer
  streamlit run app_yolo.py
  ```

### 3. **Đánh giá hiệu suất**
- **Chạy script đánh giá DBNet:**
  ```bash
  cd src/Eval/Eval_DBNet
  python eval.py
  ```
- **Chạy script đánh giá YOLOv8:**
  ```bash
  cd src/Eval/Eval_YOLOv8
  ```
  Tiến hành chạy các file `Method 1 - Only BKAI Dataset`, `Method 2`.
---


