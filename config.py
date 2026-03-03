# config.py

# -----------------------------
# Global default hyperparameters
# -----------------------------
DEFAULTS = {
    "RCNN": {
        "epochs": 20,
        "batch_size_train": 2,
        "batch_size_val": 1,
        "learning_rate": 0.005,
        "patience": 10,
        "num_classes": 2
    },
    "YOLO": {
        "epochs": 15,
        "batch_size": 8,
        "learning_rate": 0.001,
        "patience": None
    }
}

# -----------------------------
# Dataset-specific overrides
# -----------------------------
DATASETS = {
    "Oxford Pet": {
        "RCNN": {
            "root": "Oxford-IIIT Pet",
            "json_folder": "Oxford-IIIT Pet/OxfordPetSubset_lists",
            "num_classes": 11,         # 10 breeds + background
            "epochs": 20,
            "batch_size_train": 2,
            "batch_size_val": 1,
            "learning_rate": 0.005,
            "checkpoint_prefix": "oxford_pet_model"
        },
        "YOLO": {
            "data_yaml": "data_oxford.yaml",
            "save_path": "yolov8n_oxford.pt",
            "epochs": 20,
            "batch_size": 8,
            "learning_rate": 0.001
        }
    },
    "Penn-Fudan": {
        "RCNN": {
            "root": "PennFudanPed",
            "json_folder": "PennFudanPed/PennFudan_lists",
            "num_classes": 2,          # 1 class + background
            "epochs": 15,
            "batch_size_train": 2,
            "batch_size_val": 1,
            "learning_rate": 0.005,
            "checkpoint_prefix": "pennfudan_model"
        },
        "YOLO": {
            "data_yaml": "data_fudan.yaml",
            "save_path": "yolov8n_fudan.pt",
            "epochs": 15,
            "batch_size": 8,
            "learning_rate": 0.001
        }
    }
}
