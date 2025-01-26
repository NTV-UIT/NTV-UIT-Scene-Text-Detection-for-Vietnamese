import json
import os
import numpy as np
from matplotlib import pyplot as plt
import cv2

def parse_predictions(pred_file):
    """
    Parse the predictions file into a dictionary.
    :param pred_file: Path to the predictions file.
    :return: Dictionary {image_name: list of predicted boxes}.
    """
    predictions = {}
    with open(pred_file, "r", encoding="utf-8") as f:
        for line in f:
            # Loại bỏ khoảng trắng thừa ở đầu và cuối chuỗi
            # print(line)
            line = line.strip()

            # Tìm vị trí đầu tiên của dấu cách giữa tên hình ảnh và hộp dự đoán
            image_name_end_index = line.index(" ")  # Chỉ có một dấu cách để phân tách
            image_name = line[:image_name_end_index]
            boxes_str = line[image_name_end_index + 1:]  # Lấy phần sau dấu cách

            # Chuyển chuỗi box thành danh sách
            boxes = json.loads(boxes_str)

            # Lưu kết quả vào từ điển
            predictions[image_name] = boxes
                
    return predictions

def load_ground_truths(gt_dir):
    """
    Load ground truth boxes from the directory.
    Each file contains the bounding boxes in the format:
    x1, y1, x2, y2, x3, y3, x4, y4.
    :param gt_dir: Directory containing ground truth files.
    :return: Dictionary {image_name: list of GT boxes}.
    """
    gt_data = {}
    for file in os.listdir(gt_dir):
        if file.endswith(".txt"):
            img_name = file.replace("gt_", "").replace(".txt", "")
            # print(img_name)
            with open(os.path.join(gt_dir, file), "r", encoding="utf-8") as f:
                boxes = []
                for line in f:
                    # print(line)
                    points = list(map(float, line.strip().split(",")[:8]))
                    boxes.append(np.array(points).reshape(-1, 2))  # Reshape to (4, 2)
                gt_data[img_name] = boxes
    return gt_data

def load_ground_truths_polygon(gt_dir):
    """
    Load ground truth boxes with labels from the directory.
    Each file contains the bounding boxes in the format:
    x1, y1, x2, y2, ..., xn, yn, label (optional).
    :param gt_dir: Directory containing ground truth files.
    :return: Dictionary {image_name: list of (box, label)}, where box is a numpy array of points.
    """
    gt_data = {}
    for file in os.listdir(gt_dir):
        if file.endswith(".txt"):
            img_name = file.replace(".jpg", "").replace(".txt", "")
            with open(os.path.join(gt_dir, file), "r", encoding="utf-8") as f:
                boxes = []
                for line in f:
                    parts = line.strip().split(",")
                    # print(parts)
                    if parts == ['547', '424', '669', '410', '672', '437', '552', '445', 'Breakfast', 'Lunch']:
                        continue
                    # Chuyển các phần tử thành float nếu có thể, bỏ qua phần tử cuối nếu là nhãn
                    try:
                        coords = [float(p) for p in parts[:-1]]
                        label = parts[-1]
                    except ValueError:
                        coords = [float(p) for p in parts]
                        label = None

                    # Kiểm tra nếu số lượng tọa độ là lẻ
                    if len(coords) % 2 != 0:
                        print(f"Warning: Invalid number of coordinates in line '{line.strip()}' in file '{file}'")
                        continue

                    # Chuyển tọa độ thành numpy array và reshape (-1, 2)
                    box = np.array(coords).reshape(-1, 2)
                    boxes.append((box, label))

                gt_data[img_name] = boxes
    return gt_data

def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) between two boxes.
    :param box1: Numpy array of shape (4, 2).
    :param box2: Numpy array of shape (4, 2).
    :return: IoU value.
    """
    # Convert polygon to bounding box
    x_min1, y_min1 = np.min(box1, axis=0)
    x_max1, y_max1 = np.max(box1, axis=0)

    x_min2, y_min2 = np.min(box2, axis=0)
    x_max2, y_max2 = np.max(box2, axis=0)

    # Intersection
    x_min_inter = max(x_min1, x_min2)
    y_min_inter = max(y_min1, y_min2)
    x_max_inter = min(x_max1, x_max2)
    y_max_inter = min(y_max1, y_max2)

    if x_min_inter >= x_max_inter or y_min_inter >= y_max_inter:
        return 0.0

    intersection_area = (x_max_inter - x_min_inter) * (y_max_inter - y_min_inter)
    box1_area = (x_max1 - x_min1) * (y_max1 - y_min1)
    box2_area = (x_max2 - x_min2) * (y_max2 - y_min2)

    union_area = box1_area + box2_area - intersection_area
    return intersection_area / union_area

def calculate_iou_polygon(box1, box2):
    """
    Calculate the Intersection over Union (IoU) between two boxes.
    :param box1: Numpy array of shape (4, 2).
    :param box2: Numpy array of shape (4, 2).
    :return: IoU value.
    """
    # Convert polygon to bounding box
    x_min1, y_min1 = np.min(box1, axis=0)
    x_max1, y_max1 = np.max(box1, axis=0)

    x_min2, y_min2 = np.min(box2, axis=0)
    x_max2, y_max2 = np.max(box2, axis=0)

    # Intersection
    x_min_inter = max(x_min1, x_min2)
    y_min_inter = max(y_min1, y_min2)
    x_max_inter = min(x_max1, x_max2)
    y_max_inter = min(y_max1, y_max2)

    if x_min_inter >= x_max_inter or y_min_inter >= y_max_inter:
        return 0.0

    intersection_area = (x_max_inter - x_min_inter) * (y_max_inter - y_min_inter)
    box1_area = (x_max1 - x_min1) * (y_max1 - y_min1)
    box2_area = (x_max2 - x_min2) * (y_max2 - y_min2)

    union_area = box1_area + box2_area - intersection_area
    return intersection_area / union_area

def calculate_metrics(gt_boxes, pred_boxes, flat = True, iou_threshold=0.1):
    """
    Calculate Precision, Recall, and F1-score for a single image.
    :param gt_boxes: List of ground truth boxes (list of (4, 2) arrays).
    :param pred_boxes: List of predicted boxes (list of (4, 2) arrays).
    :param iou_threshold: IoU threshold to consider a prediction as correct.
    :return: precision, recall, f1
    """
    if flat == False:
        gt_boxes = [np.array(gt_box[0]) for gt_box in gt_boxes]
        pred_boxes = [np.array(pred_box) for pred_box in pred_boxes]
    matched_gt = set()
    matched_pred = set()

    for i, gt_box in enumerate(gt_boxes):
        for j, pred_box in enumerate(pred_boxes):
            if flat == True:
                iou = calculate_iou(gt_box, pred_box)
            elif flat == False:
                iou = calculate_iou_polygon(gt_box, pred_box)

            if iou >= iou_threshold and i not in matched_gt and j not in matched_pred:
                matched_gt.add(i)
                matched_pred.add(j)

    tp = len(matched_gt)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1

def visualize_boxes(image_path, gt_boxes, pred_boxes):
    """
    Visualize the ground truth and predicted bounding boxes on the image.
    :param image_path: Path to the image file.
    :param gt_boxes: List of ground truth boxes (list of (4, 2) arrays).
    :param pred_boxes: List of predicted boxes (list of (4, 2) arrays).
    """
    # Load image using OpenCV
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for matplotlib

    # Draw ground truth boxes in green
    for box in gt_boxes:
        box = np.array(box).reshape(-1, 2)
        pts = box.astype(int)
        cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=1)

    # Draw predicted boxes in red
    for box in pred_boxes:
        print(box)
        box = np.array(box).reshape(-1, 2)
        pts = box.astype(int)
        cv2.polylines(image, [pts], isClosed=True, color=(255, 0, 0), thickness=1)

    # Display the image with bounding boxes
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def evaluate_predictions(pred_file, gt_dir, img_dir, flat = True):
    """
    Evaluate predictions against ground truth data.
    :param pred_file: Path to the predictions file.
    :param gt_dir: Path to the directory containing ground truth files.
    """
    # Load predictions and ground truths
    predictions = parse_predictions(pred_file)
    print(f"flat: {flat}")
    if flat == True:
        ground_truths = load_ground_truths(gt_dir)
    else:
        ground_truths = load_ground_truths_polygon(gt_dir)

    # ground_truths = load_ground_truths(gt_dir)
    all_precisions = []
    all_recalls = []
    all_f1s = []

    for img_name, gt_boxes in ground_truths.items():
        # print(img_name)
        pred_boxes = predictions.get(f"{img_name}.jpg", [])

        precision, recall, f1 = calculate_metrics(gt_boxes, pred_boxes, flat)
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1s.append(f1)

        # print(f"Image: {img_name}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        # Visualize the boxes
        img_path = os.path.join(img_dir, img_name + ".jpg")  # Assuming the images are in img_dir and have .jpg extension
        # visualize_boxes(img_path, gt_boxes, pred_boxes)

    # Calculate overall metrics
    avg_precision = np.mean(all_precisions)
    avg_recall = np.mean(all_recalls)
    # avg_f1 = np.mean(all_f1s)
    avg_f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0.0

    print("\n--- Overall Metrics ---")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1: {avg_f1:.4f}")


# Example usage
if __name__ == "__main__":
    prediction_file = r"E:\Docs\CVA\PaddleOCR\train_data\results\resnet18_bkai_only.txt"
    ground_truth_dir = r"E:\Docs\CVA\PaddleOCR\train_data\data_test\groundtruth"
    # ground_truth_dir = r"E:\Docs\CVA\PaddleOCR\train_data\total_text\test_gts"
    img_dir = r"E:\Docs\CVA\PaddleOCR\train_data\data_test\private_test_imgs"
    evaluate_predictions(prediction_file, ground_truth_dir, img_dir, flat = True)
