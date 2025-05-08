import json
import os

def load_evaluation_results(json_path="db/evaluation_results.json"):
    if not os.path.exists(json_path):
        return []
    with open(json_path, "r") as f:
        return json.load(f)
