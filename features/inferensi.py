import streamlit as st
import os
from config.settings import UPLOAD_DIR, OUTPUT_DIR, ALLOWED_EXTENSIONS, MAX_FILE_SIZE_MB
from utils.inference import run_inference
from utils.db import save_inference_result, is_inference_exist
from utils.compress import compress
from utils.video_utils import render_video

def run():
    st.header("🔍 Inferensi Deteksi Video")

    # Upload Video
    uploaded_video = st.file_uploader("Unggah video", type=None)
    video_path = None
    print("EXTTT:", uploaded_video)
    if uploaded_video:
        if uploaded_video:
            ext = os.path.splitext(uploaded_video.name)[1].lower()
            if ext not in ALLOWED_EXTENSIONS:
                st.error(f"❌ Ekstensi {ext} tidak diizinkan!")
            else:
                os.makedirs(UPLOAD_DIR, exist_ok=True)
                video_path = os.path.join(UPLOAD_DIR, uploaded_video.name.lower())
                with open(video_path, "wb") as f:
                    f.write(uploaded_video.read())
                st.success(f"✅ {uploaded_video.name} berhasil diunggah.")
        elif uploaded_video.size > MAX_FILE_SIZE_MB * 1024 * 1024:
            st.error("❌ Ukuran file melebihi 200MB!")
            return

    model_names = [f for f in os.listdir("models") if f.endswith(".pt")]
    selected_model = st.selectbox("Pilih Model YOLO11", model_names)

    if st.button("Jalankan Inferensi"):
        if not video_path or not selected_model:
            st.error("❌ Silakan unggah video dan pilih model terlebih dahulu.")
            return
        
        filename = os.path.basename(video_path)
        if is_inference_exist(filename, selected_model):
            st.warning("⚠️ Video ini sudah pernah diuji dengan model yang sama. Inferensi tidak dijalankan ulang.")
            return

        with st.spinner("⏳ Proses inferensi sedang berjalan..."):
            output_path, total_objects = run_inference(video_path, f"models/{selected_model}", OUTPUT_DIR)
            save_inference_result(video_path, selected_model, output_path, total_objects)
            st.success(f"Inferensi selesai. Total objek terdeteksi: {total_objects}")
            output_path = compress(output_path, OUTPUT_DIR)
            render_video(output_path)

            try:
                for f in os.listdir(UPLOAD_DIR):
                    file_path = os.path.join(UPLOAD_DIR, f)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                st.info("📂 Semua file unggahan telah dihapus setelah inferensi.")
            except Exception as e:
                st.warning(f"⚠️ Gagal menghapus file di folder upload: {e}")
