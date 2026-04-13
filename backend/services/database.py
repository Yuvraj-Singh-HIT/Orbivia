import os
import json

DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "db", "database.json"
)


def load_db():
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "r") as f:
            return json.load(f)
    return {
        "users": [],
        "sessions": [],
        "analysis_history": [],
        "model_info": {
            "name": "Segformer MIT-B0",
            "num_classes": 10,
            "trained": True,
            "model_file": "segformer.pth",
            "accuracy": 0.51,
            "iou_score": 0.015,
        },
    }


def save_db(data):
    with open(DB_PATH, "w") as f:
        json.dump(data, f, indent=4)


def add_analysis_record(filename, traversability_score, danger_level):
    db = load_db()
    db["analysis_history"].append(
        {
            "filename": filename,
            "traversability_score": float(traversability_score),
            "danger_level": danger_level,
        }
    )
    save_db(db)


def get_analysis_history():
    db = load_db()
    return db.get("analysis_history", [])


def get_model_info():
    db = load_db()
    return db.get("model_info", {})


def update_model_info(**kwargs):
    db = load_db()
    if "model_info" not in db:
        db["model_info"] = {}
    db["model_info"].update(kwargs)
    save_db(db)
