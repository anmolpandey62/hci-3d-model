# logic/floor_plan_analyzer.py

def analyse_floor_plan(img, labelled_or_heur):
    enhanced = dict(labelled_or_heur) if labelled_or_heur else {}
    enhanced["_room_names"] = {}
    enhanced["_ocr_seeds"] = []
    enhanced["_analyzer_used"] = False
    return enhanced

def draw_analysis_overlay(img, analysis_data):
    return img

def extract_text_seeds(img):
    return []