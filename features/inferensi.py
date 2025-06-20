import streamlit as st
import os
import pandas as pd
from config.settings import UPLOAD_DIR, OUTPUT_DIR, ALLOWED_EXTENSIONS, ALLOWED_EXTENSIONS_IMG, ALLOWED_EXTENSIONS_LABEL, MAX_FILE_SIZE_MB, UPLOAD_DIR_IMG, UPLOAD_DIR_LABEL
from utils.inference import run_inference, run_inference_img
from utils.db import save_inference_result, is_inference_exist
from utils.compress import compress
from utils.video_utils import render_video

def run():
    st.header("🔍📽️ Inferensi Deteksi Video")

    # Upload Video
    uploaded_video = st.file_uploader("Unggah video", type=None)
    video_path = None
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

    if st.button("Jalankan Inferensi Video"):
        if not video_path or not selected_model:
            st.error("❌ Silakan unggah video dan pilih model terlebih dahulu.")
            return
        
        filename = os.path.basename(video_path)
        if is_inference_exist(filename, selected_model):
            st.warning("⚠️ Video ini sudah pernah diuji dengan model yang sama. Inferensi tidak dijalankan ulang.")
            return

        with st.spinner("⏳ Proses inferensi sedang berjalan..."):
            output_path, total_objects, avg_fps = run_inference(video_path, f"models/{selected_model}", OUTPUT_DIR)
            save_inference_result(video_path, selected_model, output_path, total_objects, avg_fps)
            st.success(f"Inferensi selesai. Total objek terdeteksi: {total_objects}")
            output_path = compress(output_path, OUTPUT_DIR)
            st.markdown(render_video(output_path), unsafe_allow_html=True)

            try:
                for f in os.listdir(UPLOAD_DIR):
                    file_path = os.path.join(UPLOAD_DIR, f)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                st.info("📂 Semua file unggahan telah dihapus setelah inferensi.")
            except Exception as e:
                st.warning(f"⚠️ Gagal menghapus file di folder upload: {e}")
    
    st.header("🔍🖼️ Inferensi Deteksi Gambar")
    # Upload Video
    uploaded_img = st.file_uploader("Unggah gambar", type=None)
    uploaded_label = st.file_uploader("Unggah label (Opsional)", type=None)
    img_path = None
    label_path = None

    if uploaded_img:
        ext = os.path.splitext(uploaded_img.name)[1].lower()
        if ext not in ALLOWED_EXTENSIONS_IMG:
            st.error(f"❌ Ekstensi gambar {ext} tidak diizinkan!")
        elif uploaded_img.size > MAX_FILE_SIZE_MB * 1024 * 1024:
            st.error("❌ Ukuran gambar melebihi 200MB!")
        else:
            os.makedirs(UPLOAD_DIR_IMG, exist_ok=True)
            img_path = os.path.join(UPLOAD_DIR_IMG, uploaded_img.name.lower())
            try:
                with open(img_path, "wb") as f:
                    f.write(uploaded_img.read())
                st.success(f"✅ Gambar '{uploaded_img.name}' berhasil diunggah.")
                # st.image(img_path, caption="Gambar yang diunggah", use_container_width=True)
            except Exception as e:
                st.error(f"❌ Gagal menyimpan gambar: {e}")
    
    if uploaded_label:
        ext = os.path.splitext(uploaded_label.name)[1].lower()
        if ext not in ALLOWED_EXTENSIONS_LABEL:
            st.error(f"❌ Ekstensi label {ext} tidak diizinkan!")
        elif uploaded_label.size > MAX_FILE_SIZE_MB * 1024 * 1024:
            st.error("❌ Ukuran label melebihi 200MB!")
        else:
            os.makedirs(UPLOAD_DIR_LABEL, exist_ok=True)
            label_path = os.path.join(UPLOAD_DIR_LABEL, uploaded_label.name.lower())
            try:
                with open(label_path, "wb") as f:
                    f.write(uploaded_label.read())
                st.success(f"✅ Label '{uploaded_label.name}' berhasil diunggah.")
            except Exception as e:
                st.error(f"❌ Gagal menyimpan label: {e}")
    
    if st.button("Jalankan Inferensi Gambar"):
        if not img_path:
            st.error("❌ Silakan unggah gambar terlebih dahulu.")
            return
        
        with st.spinner("⏳ Proses inferensi sedang berjalan..."):
            label_path_input = label_path if label_path and os.path.exists(label_path) else None
            gt_path, output_dir, stats_all, total_gt = run_inference_img(img_path, label_path_input)

           # 🔻 Hapus file gambar dan label setelah inferensi
            try:
                # Hapus gambar
                if os.path.exists(UPLOAD_DIR_IMG):
                    for f in os.listdir(UPLOAD_DIR_IMG):
                        file_path = os.path.join(UPLOAD_DIR_IMG, f)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                
                # Hapus label jika foldernya ada
                if os.path.exists(UPLOAD_DIR_LABEL):
                    for f in os.listdir(UPLOAD_DIR_LABEL):
                        file_path = os.path.join(UPLOAD_DIR_LABEL, f)
                        if os.path.isfile(file_path):
                            os.remove(file_path)

                st.info("📂 Semua file unggahan telah dihapus setelah inferensi.")
            except Exception as e:
                st.warning(f"⚠️ Gagal menghapus file di folder upload: {e}")

            st.markdown("### Keterangan Bounding Box")
            st.markdown("""
            - 🟩 **Hijau**: True Positive (terdeteksi dan sesuai ground truth)  
            - 🟥 **Merah**: False Positive (terdeteksi tapi tidak ada pada ground truth)  
            - 🟦 **Biru**: False Negative (ada di ground truth tapi tidak terdeteksi)
            """)

            # === Tampilkan Ground Truth di tengah ===
            col1, col2, col3 = st.columns([1, 2, 1])

            if gt_path:
                with col2:
                    st.markdown(f"### Ground Truth: {total_gt}")
                    st.image(gt_path, caption="Ground Truth")  # Ukuran disesuaikan
            else:
                st.info("📂 Ground truth tidak tersedia. Hanya menampilkan hasil prediksi.")

            # === Tampilkan hasil semua model ===
            custom_order = ['baseline', 'm1', 'm2', 'm3', 'm4'] + [f'c{i}' for i in range(1, 12)]
            model_images = [f for f in os.listdir(output_dir) if f.endswith('.jpg') and f.lower() != 'ground_truth.jpg']

            def extract_prefix(filename):
                import re
                match = re.match(r'([a-zA-Z]+\d*)', filename.lower())
                if match:
                    return match.group(1)
                else:
                    return filename.lower()

            model_images.sort(key=lambda x: custom_order.index(extract_prefix(x)) if extract_prefix(x) in custom_order else 999)
            
            num_cols = 4
            for i in range(0, len(model_images), num_cols):
                cols = st.columns(num_cols)
                for j in range(num_cols):
                    if i + j < len(model_images):
                        model_img_name = model_images[i + j]
                        img_full_path = os.path.join(output_dir, model_img_name)
                        with cols[j]:
                            st.image(
                                img_full_path,
                                caption=model_img_name.replace(".jpg", ""),
                                use_container_width=True
                            )

            # Tampilkan stats
            st.header("📋 Tabel Hasil Inferensi")
            stats_all.sort(key=lambda d: custom_order.index(extract_prefix(d['model'])) if extract_prefix(d['model']) in custom_order else 999)

            st.dataframe(pd.DataFrame(stats_all))
        
