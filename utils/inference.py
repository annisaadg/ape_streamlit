import cv2
from ultralytics import solutions, YOLO
import os
import time
import numpy as np
import streamlit as st

def run_inference(video_path, model_path, output_dir):
    """Melakukan inference & counting objek, menyimpan video output dan mengembalikan path dan jumlah objek."""
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Gagal membaca video."

    success, frame = cap.read()
    if not success:
        raise ValueError("Frame pertama tidak dapat dibaca.")
    
    # Rotasi hanya jika frame awal adalah landscape
    if frame.shape[1] > frame.shape[0]:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    h, w = frame.shape[:2]
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    os.makedirs(output_dir, exist_ok=True)
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    input_name = os.path.basename(video_path)
    output_filename = f"{model_name}_{input_name}"
    output_path = os.path.join(output_dir, output_filename)

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    
    # Garis vertikal di tengah
    region_center_x = w // 2
    region_points = [(region_center_x, 0), (region_center_x, h)]

    counter = solutions.ObjectCounter(
        show=True,
        region=region_points,
        model=model_path,
        show_in=True,
        show_out=True,
        classes=[0]
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    prev_time = time.time()

    total_fps = 0
    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        # Rotasi semua frame agar portrait
        if frame.shape[1] > frame.shape[0]:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            
        results = counter(frame)
        output_frame = results.plot_im
        
        curr_time = time.time()
        # fps_text = f"FPS: {1 / (curr_time - prev_time):.2f}"
        fps = 1 / (curr_time - prev_time)
        total_fps = total_fps + fps
        frame_count += 1
        avg_fps = total_fps/frame_count
        avg_fps_str = float("{:.2f}".format(avg_fps))
        avg_fps_text = f"FPS: {avg_fps_str}"
        prev_time = curr_time

        (text_width, text_height), _ = cv2.getTextSize(avg_fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        x, y = 10, 40  # posisi kiri atas teks
        cv2.rectangle(output_frame, (x - 5, y - text_height - 5), (x + text_width + 5, y + 5), (255, 255, 255), -1)

        cv2.putText(
            output_frame, avg_fps_text, (x, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA
        )
        
        writer.write(output_frame)

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    return output_path, counter.in_count + counter.out_count, avg_fps_str


######## INFERENCE FOTO ########

def load_yolo_labels(label_path):
    """Membaca file label YOLO dan mengembalikan bounding box dalam format [x1,y1,x2,y2]"""
    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                # Format YOLO: class x_center y_center width height (all normalized)
                _, xc, yc, w, h = map(float, parts)
                x1 = (xc - w/2)
                y1 = (yc - h/2)
                x2 = (xc + w/2)
                y2 = (yc + h/2)
                boxes.append([x1, y1, x2, y2])
    return boxes

def denormalize_boxes(boxes, img_w, img_h):
    """Ubah koordinat normal ke pixel"""
    denorm_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box
        x1_px = int(x1 * img_w)
        y1_px = int(y1 * img_h)
        x2_px = int(x2 * img_w)
        y2_px = int(y2 * img_h)
        denorm_boxes.append([x1_px, y1_px, x2_px, y2_px])
    return denorm_boxes

def compute_iou(boxA, boxB):
    # Hitung intersection
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    # Hitung union
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def match_boxes(pred_boxes, gt_boxes, iou_threshold=0.5):
    """
    Cocokkan prediksi dengan ground truth berdasarkan IoU threshold.
    Return: matched GT indices, matched Pred indices, false positives, false negatives
    """
    matched_gt = set()
    matched_pred = set()
    for pred_i, pbox in enumerate(pred_boxes):
        for gt_i, gtbox in enumerate(gt_boxes):
            if gt_i in matched_gt:
                continue
            iou = compute_iou(pbox, gtbox)
            if iou >= iou_threshold:
                matched_gt.add(gt_i)
                matched_pred.add(pred_i)
                break
    false_positives = [pred_boxes[i] for i in range(len(pred_boxes)) if i not in matched_pred]
    false_negatives = [gt_boxes[i] for i in range(len(gt_boxes)) if i not in matched_gt]
    return matched_gt, matched_pred, false_positives, false_negatives

def draw_boxes(img, boxes, color, label=None, thickness=2):
    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (x1,y1), (x2,y2), color, thickness)
        if label:
            cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

# Fungsi untuk load semua model dari folder models/
@st.cache_resource
def load_models():
    model_names = [f for f in os.listdir("models") if f.endswith(".pt")]
    models = []
    for name in model_names:
        model_path = os.path.join("models", name)
        model = YOLO(model_path)
        models.append((model, name))  # return tuple, bukan cuma model
    return models

def run_inference_img(img_path, label_path, iou_threshold=0.5):
    # Baca gambar
    img = cv2.imread(img_path)
    img_h, img_w = img.shape[:2]

    # Load dan denormalisasi bounding box ground truth
    gt_boxes_norm = load_yolo_labels(label_path)
    gt_boxes = denormalize_boxes(gt_boxes_norm, img_w, img_h)

    # Load semua model [(model, model_name), ...]
    models = load_models()

    # Buat copy untuk gambar GT dan gambarkan bounding box hijau
    img_gt = img.copy()
    draw_boxes(img_gt, gt_boxes, (0,255,0))

    # Buat folder output berdasarkan nama file gambar (tanpa ekstensi)
    base_img_name = os.path.splitext(os.path.basename(img_path))[0]
    output_dir = os.path.join("images", "output", base_img_name)
    os.makedirs(output_dir, exist_ok=True)

    # Simpan gambar ground truth
    gt_path = os.path.join(output_dir, "ground_truth.jpg")
    cv2.imwrite(gt_path, img_gt)

    stats_all = []

    for model, model_name in models:
        img_pred = img.copy()
        results = model(img)

        pred_boxes = []
        boxes = results[0].boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            pred_boxes.append([int(x1), int(y1), int(x2), int(y2)])

        matched_gt, matched_pred, false_positives, false_negatives = match_boxes(pred_boxes, gt_boxes, iou_threshold)

        draw_boxes(img_pred, [gt_boxes[i] for i in matched_gt], (0,255,0))
        draw_boxes(img_pred, false_positives, (0,0,255))
        draw_boxes(img_pred, false_negatives, (255,0,0))

        # Simpan gambar hasil inferensi tiap model dengan nama modelnya
        pred_path = os.path.join(output_dir, f"{model_name}.jpg")
        cv2.imwrite(pred_path, img_pred)

        total_pred = len(pred_boxes)
        percentage_error = abs((total_pred - len(gt_boxes)) / len(gt_boxes)) * 100 if len(gt_boxes) > 0 else 0

        stats = {
            "model": model_name,
            "True Positives": len(matched_gt),
            "False Positives": len(false_positives),
            "False Negatives": len(false_negatives),
            "Jumlah Aktual": len(gt_boxes),
            "Jumlah Prediksi": total_pred,
            "Percentage Error (%)": round(percentage_error, 2)
        }
        stats_all.append(stats)
    
    total_gt = len(gt_boxes)

    return gt_path, output_dir, stats_all, total_gt

# def show_inference_results(img_gt, imgs_preds, model_names):
#     n_models = len(imgs_preds)
#     cols = 3
#     rows = (n_models + cols - 1) // cols  # hitung jumlah baris

#     st.markdown("### Foto Ground Truth")
#     gt_col1, gt_col2, gt_col3 = st.columns(3)
#     with gt_col2:
#         img_gt_rgb = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
#         st.image(img_gt_rgb, caption="Ground Truth", use_container_width=True)

#     st.markdown("### Hasil Inferensi Model")

#     idx = 0
#     for r in range(rows):
#         row_cols = st.columns(cols)
#         for c in range(cols):
#             if idx >= n_models:
#                 break
#             with row_cols[c]:
#                 img_rgb = cv2.cvtColor(imgs_preds[idx], cv2.COLOR_BGR2RGB)
#                 st.image(img_rgb, caption=model_names[idx], use_container_width=True)
#             idx += 1