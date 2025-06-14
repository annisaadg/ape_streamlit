from ultralytics import YOLO
import cv2

# Load model hasil training
model = YOLO('models/Baseline.pt')  # ganti dengan path model YOLO11 kamu

# Path ke gambar
image_path = 'images/uploaded/img/kendaraan-3.jpg'  # ganti dengan path gambar kamu

# Baca gambar
img = cv2.imread(image_path)

# Jalankan inferensi
results = model(img)[0]  # Ambil hasil pertama

# Ambil informasi prediksi
boxes = results.boxes
class_names = model.names

# Loop dan gambar bounding box + nama class
for box in boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])  # bounding box koordinat
    conf = box.conf[0].item()               # confidence
    cls_id = int(box.cls[0].item())         # class id
    label = f"{class_names[cls_id]} {conf:.2f}"  # label + confidence

    # Gambar box dan label
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2)

# Tampilkan hasil
cv2.imshow("Detection Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
