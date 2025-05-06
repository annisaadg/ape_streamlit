import streamlit as st
from utils.inference import run_inference
from utils.db import load_model_info, load_evaluation_results, save_inference_result
import os
import shutil
from datetime import datetime

# Konfigurasi
st.set_page_config(page_title="APE - Aplikasi Pendukung Eksperimen", layout="wide")
UPLOAD_DIR = "videos/uploaded"
OUTPUT_DIR = "videos/output"
ALLOWED_EXTENSIONS = [".mp4", ".mov", ".avi"]
MAX_FILE_SIZE_MB = 200

st.title("🎥 APE - Aplikasi Pendukung Eksperimen Deteksi Jeruk")

# --- FR01: Upload Video ---
st.sidebar.header("1. Upload Video Uji")
uploaded_video = st.sidebar.file_uploader("Unggah video (.mp4, .mov, .avi)", type=['mp4', 'mov', 'avi'])
                                        
if uploaded_video:
    if uploaded_video.size > MAX_FILE_SIZE_MB * 1024 *1024:
        st.sidebar.error("Ukuran file melebihi 200MB!")
    else:
        video_path = os.path.join(UPLOAD_DIR, uploaded_video.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_video.read())
        st.sidebar.success(f"✅ {uploaded_video.name} berhasil diunggah.")

# --- FR02: Pilih Model ---
model_names = [f for f in os.listdir("models") if f.endswith(".pt")]
selected_model = st.sidebar.selectbox("2. Pilih Model YOLO11", model_names)

# --- FR03-FR06: Inferensi ---
if st.sidebar.button("🔍 Jalankan Inferensi"):
    if not uploaded_video or not selected_model:
        st.error("Silakan unggah video dan pilih model terlebih dahulu.")
    else:
        with st.spinner("Proses inferensi sedang berjalan..."):
            output_path, total_objects = run_inference(video_path, f"models/{selected_model}", OUTPUT_DIR)
            save_inference_result(video_path, selected_model, output_path, total_objects)
            st.success(f"Inferensi selesai. Total objek terdeteksi: {total_objects}")
            st.video(output_path)
            
# --- FR07: Perbandingan Model ---
st.subheader("📊 Perbandingan Performa Model")
evaluation_data = load_evaluation_results()
model1 = st.selectbox("Model A", model_names, key="model1")
model2 = st.selectbox("Model B", model_names, key="model2")

if st.button("Bandingkan Model"):
    if model1 == model2:
        st.warning("Pilih dua model yang berbeda.")
    else:
        eval1 = evaluation_data.get(model1, {})
        eval2 = evaluation_data.get(model2, {})
        
        st.write("📈 Evaluasi Performa")
        st.table({
            "Metrik": ["Precision", "Recall", "mAP50", "mAP50-95", "FPS", "Jumlah Parameter"],
            model1: [eval1.get(k, "-") for k in ["precision", "recall", "map50", "map5095", "fps", "params"]],
            model2: [eval2.get(k, "-") for k in ["precision", "recall", "map50", "map5095", "fps", "params"]],
        })
        
# --- FR08: Rekap Deteksi Semua Video ---
st.subheader("📋 Rekapitulasi Jumlah Objek Terdeteksi")
from utils.db import get_rekapitulasi
rekap_df = get_rekapitulasi()
st.dataframe(rekap_df)