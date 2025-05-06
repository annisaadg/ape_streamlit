import streamlit as st
import os
from utils.inference import run_inference
from utils.db import save_inference_result
from utils.compress import compress
import base64

# Perbaikan bug streamlit x torch
import torch._classes  # untuk menghindari error '__path__._path'

# Konfigurasi halaman
st.set_page_config(page_title="APE - Aplikasi Pendukung Eksperimen", layout="wide")
UPLOAD_DIR = "videos/uploaded"
OUTPUT_DIR = "videos/output"
ALLOWED_EXTENSIONS = [".mp4", ".mov", ".avi", ".mpeg4"]
MAX_FILE_SIZE_MB = 200

# Sidebar: pilih fitur
fitur_dipilih = st.sidebar.radio("Pilih Fitur", ["Inferensi", "Perbandingan Model", "Rekapitulasi Jumlah Objek"])

# Judul halaman utama
st.title("🎥 APE - Aplikasi Pendukung Eksperimen Deteksi Jeruk")

# --- FITUR 1: INFERENSI ---
if fitur_dipilih == "Inferensi":
    st.header("🔍 Inferensi Deteksi Video")  
    
    
    
    # Baca file video
    video_path = "videos/output/compressed/compressed_inference_img_0063.mp4"
    with open(video_path, "rb") as f:
        video_bytes = f.read()

    # Encode ke base64
    b64_video = base64.b64encode(video_bytes).decode()

    # Buat elemen HTML video
    video_html = f"""
        <div style="display: flex; justify-content: center; margin-top: 20px;">
            <video width="640" height="360" controls style="border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.2);">
                <source src="data:video/mp4;base64,{b64_video}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
        </div>
    """

    # Tampilkan di Streamlit
    st.markdown(video_html, unsafe_allow_html=True)
    
      

    # Upload Video
    uploaded_video = st.file_uploader("Unggah video (.mp4, .mov, .avi, .mpeg4)", type=['mp4', 'mov', 'avi', 'mpeg4'])

    video_path = None
    if uploaded_video:
        ext = os.path.splitext(uploaded_video.name)[1].lower()  # Pastikan ekstensi huruf kecil
        if ext not in ALLOWED_EXTENSIONS:
            st.error(f"❌ Format {ext} tidak diizinkan!")
        elif uploaded_video.size > MAX_FILE_SIZE_MB * 1024 * 1024:
            st.error("❌ Ukuran file melebihi 200MB!")
        else:
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            video_name = uploaded_video.name.lower()  # Ubah nama file menjadi huruf kecil
            video_path = os.path.join(UPLOAD_DIR, video_name)
            with open(video_path, "wb") as f:
                f.write(uploaded_video.read())
            st.success(f"✅ {uploaded_video.name} berhasil diunggah.")

    # Pilih Model
    model_names = [f for f in os.listdir("models") if f.endswith(".pt")]
    selected_model = st.selectbox("Pilih Model YOLO11", model_names)

    # Jalankan Inferensi
    if st.button("Jalankan Inferensi"):
        if not video_path or not selected_model:
            st.error("❌ Silakan unggah video dan pilih model terlebih dahulu.")
        else:
            with st.spinner("⏳ Proses inferensi sedang berjalan..."):
                output_path, total_objects = run_inference(video_path, f"models/{selected_model}", OUTPUT_DIR)
                save_inference_result(video_path, selected_model, output_path, total_objects)
                st.success(f"Inferensi selesai. Total objek terdeteksi: {total_objects}")
                
                # Compress video agar tidak terlalu besar
                output_path = compress(output_path, OUTPUT_DIR)
                
                # Coba putar video dengan path yang benar
                if output_path:
                    try:
                        # Pastikan format video cocok
                        st.video(output_path)
                    except Exception as e:
                        st.error(f"❌ Tidak dapat memutar video. Error: {e}")

# --- FITUR 2: PERBANDINGAN MODEL ---
elif fitur_dipilih == "Perbandingan Model":
    st.header("📊 Perbandingan Performa Model")

    model_names = [f for f in os.listdir("models") if f.endswith(".pt")]
    evaluation_data = load_evaluation_results()

    model1 = st.selectbox("Model A", model_names, key="model1")
    model2 = st.selectbox("Model B", model_names, key="model2")

    if st.button("Bandingkan Model"):
        if model1 == model2:
            st.warning("⚠️ Pilih dua model yang berbeda.")
        else:
            eval1 = next((item for item in evaluation_data if item['model_name'] == model1), {})
            eval2 = next((item for item in evaluation_data if item['model_name'] == model2), {})

            st.write("📈 Evaluasi Performa")
            st.table({
                "Metrik": ["Precision", "Recall", "mAP50", "mAP50-95", "FPS", "Jumlah Parameter"],
                model1: [eval1.get(k, "-") for k in ["precision", "recall", "map50", "map50_95", "fps", "params"]],
                model2: [eval2.get(k, "-") for k in ["precision", "recall", "map50", "map50_95", "fps", "params"]],
            })

# --- FITUR 3: REKAPITULASI OBJEK TERDETEKSI ---
elif fitur_dipilih == "Rekapitulasi Jumlah Objek":
    st.header("📋 Rekapitulasi Jumlah Objek Terdeteksi")

    rekap_df = get_rekapitulasi()
    st.dataframe(rekap_df)
