import streamlit as st
from config.settings import PAGE_TITLE
from features import inferensi, perbandingan, rekapitulasi

st.set_page_config(page_title=PAGE_TITLE, layout="wide")
st.title("🎥 APE - Aplikasi Pendukung Eksperimen: Deteksi dan Penghitungan Jeruk")

# Sidebar: Pilih fitur
fitur_dipilih = st.sidebar.radio("Pilih Fitur", ["Inferensi", "Perbandingan Model", "Rekap Inferensi"])

if fitur_dipilih == "Inferensi":
    inferensi.run()
elif fitur_dipilih == "Perbandingan Model":
    perbandingan.run()
elif fitur_dipilih == "Rekap Inferensi":
    rekapitulasi.run()
