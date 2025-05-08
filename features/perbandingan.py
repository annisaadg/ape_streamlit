import streamlit as st
import os
from utils.evaluation import load_evaluation_results

def run():
    st.header("📊 Perbandingan Performa Model")
    model_names = [f for f in os.listdir("models") if f.endswith(".pt")]
    evaluation_data = load_evaluation_results()

    model1 = st.selectbox("Model A", model_names, key="model1")
    model2 = st.selectbox("Model B", model_names, key="model2")

    if st.button("Bandingkan Model"):
        if model1 == model2:
            st.warning("⚠️ Pilih dua model yang berbeda.")
            return
        eval1 = next((item for item in evaluation_data if item['model_name'] == model1), {})
        eval2 = next((item for item in evaluation_data if item['model_name'] == model2), {})

        st.write("📈 Evaluasi Performa")
        st.table({
            "Metrik": ["Precision", "Recall", "mAP50", "mAP50-95", "FPS", "Jumlah Parameter"],
            model1: [eval1.get(k, "-") for k in ["precision", "recall", "map50", "map50_95", "fps", "params"]],
            model2: [eval2.get(k, "-") for k in ["precision", "recall", "map50", "map50_95", "fps", "params"]],
        })
