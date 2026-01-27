LOCATIONS = {
    "0": {
        "name": "",
        "video_source": "",
        "road_capacity": 0,
        "latitude": 0,
        "longitude": 0
    },
}

MODEL_CONFIG = {
    "model_path": "yolov8n.pt",
    "confidence_threshold": 0.5,
    "vehicle_classes": [2, 3, 5, 7],
    "class_names": {
        2: "car",
        3: "motorcycle", 
        5: "bus",
        7: "truck"
    }
}

CONGESTION_THRESHOLDS = {
    "sangat_padat": 1.0,
    "padat": 0.7,
    "normal": 0.4,
    "lengang": 0.2
}

API_CONFIG = {
    "webhook_url": "http://0.0.0.0:5000/webhook",
    "host": "0.0.0.0",
    "port": 5000,
    "debug": True
}

PROCESSING_CONFIG = {
    "interval_seconds": 1,
    "show_window": False,
    "batch_size": 1,
    "save_detections": False,
    "output_dir": "detections/",
    "dataset_mode": True,
    "dataset_dir": "dataset",
    "frame_rate": 1
}