import pandas as pd
import os
import json

DATA_FILE = "db/inference_log.json"
EVAL_FILE = "db/evaluation_results.json"
os.makedirs("db", exist_ok=True)

def save_inference_result(video_path, model_name, output_path, total_objects):
    """Simpan hasil inferensi ke file JSON."""
    try:
        with open(DATA_FILE, "r") as f:
            data = json.load(f)
    except:
        data = []

    data.append({
        "video": os.path.basename(video_path),
        "model": model_name,
        "output": os.path.basename(output_path),
        "total_objects": total_objects
    })

    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)

def get_rekapitulasi():
    """Ambil rekap deteksi dari file JSON sebagai DataFrame."""
    if not os.path.exists(DATA_FILE):
        return pd.DataFrame(columns=["video", "model", "output", "total_objects"])
    with open(DATA_FILE, "r") as f:
        data = json.load(f)
    return pd.DataFrame(data)

def load_evaluation_results():
    """Muat hasil evaluasi dari file JSON."""
    if not os.path.exists(EVAL_FILE):
        return {}
    with open(EVAL_FILE, "r") as f:
        return json.load(f)

def load_model_info():
    """(Opsional) Muat info tambahan model, jika diperlukan."""
    # Placeholder
    return {}
