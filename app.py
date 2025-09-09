import streamlit as st
from config.settings import PAGE_TITLE
from features import inferensi, perbandingan, rekapitulasi, hasil_statistik
import os
import sys
sys.path.insert(0, os.path.abspath(".")) 

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
elif fitur_dipilih == "Hasil Inferensi Objek Non Dekopon":
    hasil_statistik.run()
