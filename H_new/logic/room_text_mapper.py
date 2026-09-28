# logic/room_text_mapper.py

def analyse_image(img, room_cnts=None):
    # Placeholder response to satisfy server.py
    return {
        "mappings": [],
        "assigned": [],
        "summary": "OCR not implemented (stub).",
        "pre_label_img": img.copy() if img is not None else None,
        "post_label_img": img.copy() if img is not None else None,
        "regions": []
    }

def draw_text_mapping_overlay(img, mappings):
    return img.copy() if img is not None else None