import cv2
from ultralytics import solutions
import os

def run_inference(video_path, model_path, output_dir):
    """Melakukan inference & counting objek, menyimpan video output dan mengembalikan path dan jumlah objek."""
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Gagal membaca video."

    success, frame = cap.read()
    if not success:
        raise ValueError("Frame pertama tidak dapat dibaca.")

    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    h, w = frame.shape[:2]
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    filename = os.path.basename(video_path)
    output_path = os.path.join(output_dir, f"inference_{filename}")
    os.makedirs(output_dir, exist_ok=True)

    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    region_points = [(100, 0), (100, h)]

    counter = solutions.ObjectCounter(
        show=True,
        region=region_points,
        model=model_path,
        show_in=True,
        show_out=True,
        classes=[0]
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        results = counter(frame)
        writer.write(results.plot_im)

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    return output_path, counter.in_count + counter.out_count
