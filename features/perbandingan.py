import streamlit as st
import os
from utils.evaluation import load_evaluation_results
import altair as alt
import pandas as pd

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

        # Format nilai parameter dan metrik untuk model1 dan model2
        def format_params(val):
            return f"{int(val):,}" if isinstance(val, (int, float)) else val

        st.write("📈 Evaluasi Performa")
        st.table({
            "Metrik": ["Precision", "Recall", "mAP50", "mAP50-95", "Jumlah Parameter"],
            model1: [
            f"{eval1.get(k, '-'):.4f}" if isinstance(eval1.get(k, "-"), float) else format_params(eval1.get(k, "-"))
            for k in ["precision", "recall", "map50", "map50_95", "params"]
            ],
            model2: [
                f"{eval2.get(k, '-'):.4f}" if isinstance(eval2.get(k, "-"), float) else format_params(eval2.get(k, "-"))
                for k in ["precision", "recall", "map50", "map50_95", "params"]
            ]
        })

    df = pd.DataFrame(evaluation_data)

    # List metrik yang ingin divisualisasikan
    metrics = ['precision', 'recall', 'map50', 'map50_95']
    metric_titles = {
        'precision': 'Precision',
        'recall': 'Recall',
        'map50': 'mAP@0.5',
        'map50_95': 'mAP@0.5:0.95'
    }

    # Loop dan tampilkan satu grafik per metrik
    for metric in metrics:
        st.subheader(f"{metric_titles[metric]}")
        
        chart = alt.Chart(df).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X('model_name:N', title='Model'),
            y=alt.Y(f'{metric}:Q', title='Nilai'),
            color=alt.Color('model_name:N', legend=None),
            tooltip=['model_name', f'{metric}']
        ).properties(height=300).configure_axis(labelAngle=0)

        st.altair_chart(chart, use_container_width=True)