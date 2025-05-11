import cv2
import sys
from ultralytics import solutions
import os
import time

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

    os.makedirs(output_dir, exist_ok=True)
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    input_name = os.path.basename(video_path)
    output_filename = f"{model_name}_{input_name}"
    output_path = os.path.join(output_dir, output_filename)

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
    prev_time = time.time()

    total_fps = 0
    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
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

    return output_path, counter.in_count + counter.out_count
