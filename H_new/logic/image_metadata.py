# logic/image_metadata.py
import json
from pathlib import Path

def get_metadata_path(img_path, dataset_dir):
    return Path(dataset_dir) / "metadata" / (Path(img_path).stem + ".json")

def metadata_exists(img_path, dataset_dir):
    return get_metadata_path(img_path, dataset_dir).exists()

def load_metadata(img_path, dataset_dir):
    path = get_metadata_path(img_path, dataset_dir)
    if path.exists():
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_metadata(img_path, dataset_dir, data):
    path = get_metadata_path(img_path, dataset_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    return path

def build_metadata_from_ocr(img_path, img, ocr_seeds, room_names, label_lines, labelled, h, w):
    return {"source": "ocr_stub", "notes": "Stubbed metadata"}

def build_metadata_from_gemini(img_path, gemini_data):
    return {"source": "gemini_stub"}

def list_all_metadata(dataset_dir):
    meta_dir = Path(dataset_dir) / "metadata"
    if not meta_dir.exists():
        return []
    res = []
    for p in meta_dir.glob("*.json"):
        res.append({"file": p.name})
    return res