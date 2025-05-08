import streamlit as st
import os
from config.settings import VIDEO_OUTPUT_DIR

def run():
    params = st.query_params
    file_name = params.get("file", None)

    if not file_name:
        st.error("Nama file video tidak diberikan.")
        return

    st.header("🎬 Hasil Video Inferensi")
    output_path = os.path.join(VIDEO_OUTPUT_DIR, file_name)

    if os.path.exists(output_path):
        st.video(output_path)
    else:
        st.error(f"Video tidak ditemukan di: `{output_path}`")
