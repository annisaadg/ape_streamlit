import streamlit as st
from utils.db import get_rekapitulasi
from config.settings import VIDEO_OUTPUT_DIR
from utils.video_utils import render_video
import os

def run():
    st.header("📋 Rekapitulasi Jumlah Objek Terdeteksi")
    df = get_rekapitulasi()
    st.dataframe(df)
    
    st.markdown("### Pratinjau Video")

    video_list = ["-- Pilih Video --"] + df["output"].tolist()
    selected_video = st.selectbox("Pilih video output untuk ditampilkan:", video_list)

    if selected_video != "-- Pilih Video --":
        full_path = os.path.join(VIDEO_OUTPUT_DIR, selected_video)
        if os.path.exists(full_path):
            video_container = st.empty()
            video_html = render_video(full_path)
            video_container.markdown(video_html, unsafe_allow_html=True)
        else:
            st.error(f"❌ Video tidak ditemukan di `{full_path}`.")