import streamlit as st
import os
import json
import pandas as pd
from PIL import Image

def run():
    st.header("🖼️ Hasil Inferensi Statis")
    base_output_dir = "images/output"

    if not os.path.exists(base_output_dir):
        st.warning("⚠️ Folder output tidak ditemukan.")
        return

    # Folder yang tersedia di direktori output
    available_folders = [f for f in os.listdir(base_output_dir) if os.path.isdir(os.path.join(base_output_dir, f))]

    # Urutan folder yang diinginkan
    custom_order = ['baseline', 'm01', 'm02', 'm03', 'm04'] + [f'c{i:02}' for i in range(1, 12)]

    # Urutkan folder sesuai urutan kustom
    folder_names = [f for f in custom_order if f in available_folders]

    # Tambahkan folder lain yang tidak ada dalam custom_order
    other_folders = sorted([f for f in available_folders if f not in custom_order])
    folder_names += other_folders

    if not folder_names:
        st.info("📂 Belum ada hasil inferensi yang tersimpan.")
        return

    for folder in folder_names:
        folder_path = os.path.join(base_output_dir, folder)
        st.subheader(f"📁 {folder}")

        # Tampilkan gambar hasil inferensi
        image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png'))])
        if image_files:
            num_cols = 4
            for i in range(0, len(image_files), num_cols):
                cols = st.columns(num_cols)
                for j in range(num_cols):
                    if i + j < len(image_files):
                        img_file = image_files[i + j]
                        img_path = os.path.join(folder_path, img_file)
                        try:
                            image = Image.open(img_path)
                            with cols[j]:
                                st.image(image, caption=img_file.replace(".jpg", "").replace(".png", ""), use_container_width=True)
                        except Exception as e:
                            st.warning(f"Gagal membuka gambar: {img_file} ({e})")

        # Tampilkan log hasil inferensi
        log_path = os.path.join(folder_path, "log_inferensi.json")
        if os.path.exists(log_path):
            try:
                with open(log_path, "r") as f:
                    log_data = json.load(f)
                st.markdown("### 📊 Log Hasil Inferensi")
                st.markdown(f"🕒 Timestamp: `{log_data.get('timestamp', '-')}`")
                st.markdown(f"📌 Total Ground Truth: **{log_data.get('total_ground_truth', 0)}**")
                st.dataframe(pd.DataFrame(log_data.get("results", [])))
            except Exception as e:
                st.error(f"Gagal membaca log inferensi: {e}")

        # Tampilkan atau tulis deskripsi
        st.markdown("### 📝 Deskripsi")
        deskripsi_path = os.path.join(folder_path, "deskripsi.txt")
        if os.path.exists(deskripsi_path):
            with open(deskripsi_path, "r") as f:
                st.info(f.read())
        else:
            deskripsi_input = st.text_area(f"Tulis deskripsi untuk {folder}", key=folder, height=100)
            if st.button(f"Simpan Deskripsi untuk {folder}"):
                try:
                    with open(deskripsi_path, "w") as f:
                        f.write(deskripsi_input)
                    st.success("Deskripsi berhasil disimpan.")
                except Exception as e:
                    st.error(f"Gagal menyimpan deskripsi: {e}")

        st.markdown("---")
