from pathlib import Path
import streamlit as st
from utils.evaluation import load_evaluation_results
from config.settings import EVALUATION_DIR
import altair as alt
import pandas as pd
import matplotlib.pyplot as plt
import os

def run():
    st.header("📊 Perbandingan Hasil Train Dua Model")
    evaluation_data = load_evaluation_results()
    model_names = list({item['model_name'] for item in evaluation_data})
    model_names = ["-- Pilih Model --"] + sorted(model_names)  # tambahkan opsi default

    model1 = st.selectbox("Model A", model_names, index=0, key="model1")
    model2 = st.selectbox("Model B", model_names, index=0, key="model2")

    compare_clicked = st.button("Bandingkan Model")

    if compare_clicked:
        if model1 == "---pilih model---" or model2 == "---pilih model---":
            st.warning("⚠️ Silakan pilih kedua model untuk membandingkan.")
            return
        if model1 == model2:
            st.warning("⚠️ Pilih dua model yang berbeda.")
            return

        # Lanjut tampilkan tabel dan grafik seperti biasa
        eval1 = next((item for item in evaluation_data if item['model_name'] == model1), {})
        eval2 = next((item for item in evaluation_data if item['model_name'] == model2), {})

        def format_params(val):
            return f"{int(val):,}" if isinstance(val, (int, float)) else val

        st.subheader("📈 Evaluasi Performa")
        st.table({
            "Metrik": ["Precision", "Recall", "mAP50", "mAP50-95", "Jumlah Parameter"],
            model1: [
                f"{eval1.get(k, '-') * 100:.2f}%".replace('.', ',') if isinstance(eval1.get(k, "-"), float) else format_params(eval1.get(k, "-"))
                for k in ["precision", "recall", "map50", "map50_95", "params"]
            ],
            model2: [
                f"{eval2.get(k, '-') * 100:.2f}%".replace('.', ',') if isinstance(eval2.get(k, "-"), float) else format_params(eval2.get(k, "-"))
                for k in ["precision", "recall", "map50", "map50_95", "params"]
            ]
        })

        # --- PR Curve ---
        st.subheader("📉 Precision-Recall Curve")
        cols = st.columns(2)
        for idx, model in enumerate([model1, model2]):
            model_folder = Path(model).stem
            pr_path = os.path.join(EVALUATION_DIR, model_folder, "PR_curve.png")
            if os.path.exists(pr_path):
                cols[idx].image(pr_path, caption=f"{model} - PR Curve", use_container_width=True)
            else:
                cols[idx].warning(f"PR Curve untuk {model} tidak ditemukan.")

        # --- Loss Curve ---
        st.subheader("📉 Grafik Loss dan Metrics per Epoch")

        metrics_row1 = ['train/box_loss', 'train/cls_loss', 'train/dfl_loss', 'metrics/precision(B)', 'metrics/recall(B)']
        metrics_row2 = ['val/box_loss', 'val/cls_loss', 'val/dfl_loss', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']

        for model in [model1, model2]:
            result_path = os.path.join(EVALUATION_DIR, Path(model).stem, "results.csv")
            if os.path.exists(result_path):
                df_result = pd.read_csv(result_path)
                if "epoch" not in df_result.columns:
                    df_result["epoch"] = df_result.index + 1

                st.markdown(f"### {model}")

                # Baris pertama
                cols = st.columns(5)
                for i, metric in enumerate(metrics_row1):
                    if metric in df_result.columns:
                        chart = alt.Chart(df_result).mark_line().encode(
                            x=alt.X('epoch:Q', title='Epoch'),
                            y=alt.Y(f'{metric}:Q', title=metric),
                            tooltip=[alt.Tooltip('epoch:Q', title='Epoch'), alt.Tooltip(f'{metric}:Q', title=metric)],
                        ).properties(
                            height=200,
                            width=200,
                            title=metric
                        )
                        cols[i].altair_chart(chart, use_container_width=True)
                    else:
                        cols[i].write(f"**{metric}** tidak tersedia")

                # Baris kedua
                cols = st.columns(5)
                for i, metric in enumerate(metrics_row2):
                    if metric in df_result.columns:
                        chart = alt.Chart(df_result).mark_line().encode(
                            x=alt.X('epoch:Q', title='Epoch'),
                            y=alt.Y(f'{metric}:Q', title=metric),
                            tooltip=[alt.Tooltip('epoch:Q', title='Epoch'), alt.Tooltip(f'{metric}:Q', title=metric)],
                        ).properties(
                            height=200,
                            width=200,
                            title=metric
                        )
                        cols[i].altair_chart(chart, use_container_width=True)
                    else:
                        cols[i].write(f"**{metric}** tidak tersedia")
            else:
                st.warning(f"results.csv untuk {model} tidak ditemukan.")
        
    # --- Bar Chart Komparatif Semua Model (tidak berubah) ---
    st.header("📊 Perbandingan Performa Seluruh Model")

    df = pd.DataFrame(evaluation_data)
    model_names = df['model_name'].unique().tolist()

    # Multiselect dengan default semua model terpilih
    selected_models = st.multiselect(
        "Pilih model yang ingin dibandingkan (biarkan kosong untuk semua model)", 
        options=model_names, 
        default=model_names
    )
    
    plot_metrics_per_epoch(selected_models, EVALUATION_DIR)

    # Filter dataframe sesuai pilihan user, kalau kosong berarti semua model
    if selected_models:
        df_filtered = df[df['model_name'].isin(selected_models)]
    else:
        df_filtered = df

    # --- Tabel Sortable ---
    st.subheader("📋 Tabel Ringkasan Performa")
    df_display = df_filtered[['model_name', 'precision', 'recall', 'map50', 'map50_95', 'params']].copy()
    df_display.columns = ['Model', 'Precision(%)', 'Recall(%)', 'mAP50(%)', 'mAP50-95(%)', 'Jumlah Parameter']
    df_display['Precision(%)'] = (df_display['Precision(%)'] * 100).round(2)
    df_display['Recall(%)'] = (df_display['Recall(%)'] * 100).round(2)
    df_display['mAP50(%)'] = (df_display['mAP50(%)'] * 100).round(2)
    df_display['mAP50-95(%)'] = (df_display['mAP50-95(%)'] * 100).round(2)
    df_display['Jumlah Parameter'] = df_display['Jumlah Parameter'].apply(lambda x: f"{int(x):,}")
    
    # Tambahkan kolom indeks khusus (0, M1, ..., C11)
    custom_indices = ['0', 'M1', 'M2', 'M3', 'M4', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11']
    # Reset index dan assign ulang berdasarkan urutan data yang tersaring
    df_display.insert(0, 'ID', custom_indices[:len(df_display)])

    st.dataframe(df_display, use_container_width=True, hide_index=True)

    # --- Grafik Per Metrik ---
    metrics = ['precision', 'recall', 'map50', 'map50_95']
    metric_titles = {
        'precision': 'Precision',
        'recall': 'Recall',
        'map50': 'mAP@0.5',
        'map50_95': 'mAP@0.5:0.95'
    }

    for metric in metrics:
        st.subheader(f"{metric_titles[metric]}")

        chart = alt.Chart(df_filtered).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X('model_name:N', title='Model'),
            y=alt.Y(f'{metric}:Q', title='Nilai'),
            color=alt.Color('model_name:N', legend=None),
            tooltip=['model_name', f'{metric}']
        ).properties(height=300).configure_axis(labelAngle=0)

        st.altair_chart(chart, use_container_width=True)

def plot_metrics_per_epoch(selected_models, evaluation_dir):
    import matplotlib.pyplot as plt
    from pathlib import Path
    import os
    import streamlit as st

    colors = [
        'blue', 'green', 'orange', 'red', 'purple', 'brown', 'pink', 'gray',
        'olive', 'cyan', 'magenta', 'navy', 'teal', 'gold', 'coral', 'black'
    ]

    fig, axs = plt.subplots(2, 2, figsize=(18, 10))
    axs = axs.ravel()

    titles = ['Precision', 'Recall', 'mAP at IoU=0.5', 'mAP for IoU Range 0.5–0.95']
    y_columns = [
        'metrics/precision(B)',
        'metrics/recall(B)',
        'metrics/mAP50(B)',
        'metrics/mAP50-95(B)'
    ]

    for idx, model in enumerate(selected_models):
        model_folder = Path(model).stem
        csv_path = os.path.join(evaluation_dir, model_folder, "results.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if 'epoch' not in df.columns:
                df['epoch'] = df.index + 1

            color = colors[idx % len(colors)]
            for i, y_col in enumerate(y_columns):
                if y_col in df.columns:
                    axs[i].plot(df['epoch'], df[y_col], label=model, color=color)
                else:
                    axs[i].text(0.5, 0.5, f"'{y_col}' tidak ada pada {model}", 
                                transform=axs[i].transAxes,
                                ha='center', va='center', fontsize=10, color='red')
        else:
            st.warning(f"File results.csv untuk model {model} tidak ditemukan.")

    for i in range(4):
        axs[i].set_title(titles[i])
        axs[i].set_xlabel('Epoch')
        axs[i].set_ylabel(titles[i])
        axs[i].legend(fontsize='small')
        axs[i].grid(True)

    plt.tight_layout()
    st.pyplot(fig)
