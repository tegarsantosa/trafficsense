import cv2
import numpy as np
import requests
import json
import yt_dlp
import os
import csv

from ultralytics import YOLO
from datetime import datetime
from multiprocessing import Process


class VehicleDetector:
    def __init__(
        self,
        model_path,
        webhook_url,
        confidence_threshold=0.5,
        vehicle_classes=None,
        class_names=None,
        show_window=False,
        dataset_mode=False,
        dataset_dir="dataset"
    ):
        self.model = YOLO(model_path)
        self.webhook_url = webhook_url
        self.confidence_threshold = confidence_threshold
        self.vehicle_classes = vehicle_classes or [2, 3, 5, 7]
        self.class_names = class_names or {
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck"
        }
        self.show_window = show_window
        self.dataset_mode = dataset_mode
        self.dataset_dir = dataset_dir
        
        if self.dataset_mode:
            os.makedirs(self.dataset_dir, exist_ok=True)

    def get_youtube_stream_url(self, youtube_url):
        ydl_opts = {
            "format": "best[height<=720]",
            "quiet": True,
            "no_warnings": True,
            "nocheckcertificate": True,
            "source_address": "0.0.0.0",
            "extractor_args": {"youtube": {"player_client": ["android", "web"]}},
            "http_headers": {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                return info["url"]
        except Exception as e:
            print(f"[YouTube] Failed to extract stream: {e}")
            return None

    def detect_vehicles(self, frame):
        results = self.model(
            frame,
            classes=self.vehicle_classes,
            conf=self.confidence_threshold,
            verbose=False
        )

        detections = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())

                detections.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "class": self.class_names.get(cls, str(cls)),
                    "confidence": conf
                })

        return detections, len(detections)

    def calculate_congestion_index(self, vehicle_count, road_capacity):
        return round(vehicle_count / road_capacity, 2)

    def classify_congestion(self, ci):
        if ci >= 1.0:
            return "Sangat Padat"
        elif ci >= 0.7:
            return "Padat"
        elif ci >= 0.4:
            return "Normal"
        elif ci >= 0.2:
            return "Lengang"
        return "Sangat Lengang"

    def send_to_webhook(self, data):
        try:
            requests.post(self.webhook_url, json=data, timeout=5)
        except Exception as e:
            print(f"[Webhook] Error: {e}")

    def save_to_dataset(self, location_name, timestamp, vehicle_count, ci, status, detections):
        dataset_file = os.path.join(self.dataset_dir, f"{location_name.replace(' ', '_')}_dataset.csv")
        
        file_exists = os.path.exists(dataset_file)
        
        with open(dataset_file, 'a', newline='') as f:
            writer = csv.writer(f)
            
            if not file_exists:
                writer.writerow([
                    "timestamp",
                    "vehicle_count",
                    "congestion_index",
                    "status",
                    "cars",
                    "motorcycles",
                    "buses",
                    "trucks"
                ])
            
            vehicle_types = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}
            for det in detections:
                vehicle_types[det["class"]] += 1
            
            writer.writerow([
                timestamp,
                vehicle_count,
                ci,
                status,
                vehicle_types["car"],
                vehicle_types["motorcycle"],
                vehicle_types["bus"],
                vehicle_types["truck"]
            ])

    def process_video(
        self,
        video_source,
        location_name,
        road_capacity,
        interval_seconds,
        frame_rate=1
    ):
        if isinstance(video_source, str) and "youtube.com" in video_source:
            video_source = self.get_youtube_stream_url(video_source)
            if not video_source:
                return

        cap = cv2.VideoCapture(video_source)

        if not cap.isOpened():
            print(f"[ERROR] Cannot open stream: {video_source}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        capture_interval = int(fps / frame_rate)
        frame_interval = int(fps * interval_seconds)

        frame_count = 0
        vehicle_counts = []
        all_detections = []

        print(f"[START] {location_name} | FPS: {fps} | Frame Rate: {frame_rate} fps")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % capture_interval == 0:
                detections, count = self.detect_vehicles(frame)
                vehicle_counts.append(count)
                all_detections.extend(detections)

                if self.show_window:
                    for det in detections:
                        x1, y1, x2, y2 = det["bbox"]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    cv2.putText(
                        frame,
                        f"{location_name} | Vehicles: {count}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )

                    cv2.imshow(f"TrafficSense - {location_name}", frame)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            if frame_count % frame_interval == 0 and vehicle_counts:
                avg_count = int(np.mean(vehicle_counts))
                ci = self.calculate_congestion_index(avg_count, road_capacity)
                status = self.classify_congestion(ci)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                payload = {
                    "lokasi": location_name,
                    "waktu": timestamp,
                    "jumlah_kendaraan": avg_count,
                    "congestion_index": ci,
                    "status": status
                }

                if self.dataset_mode:
                    self.save_to_dataset(
                        location_name,
                        timestamp,
                        avg_count,
                        ci,
                        status,
                        all_detections
                    )
                    print(f"[DATASET] {location_name} | Count: {avg_count} | CI: {ci} | Status: {status}")
                else:
                    self.send_to_webhook(payload)
                    print(f"[WEBHOOK] {location_name} | Count: {avg_count} | CI: {ci} | Status: {status}")

                vehicle_counts.clear()
                all_detections.clear()

            frame_count += 1

        cap.release()
        if self.show_window:
            cv2.destroyAllWindows()

        print(f"[STOP] {location_name}")


def run_camera(location, model_cfg, api_cfg, processing_cfg):
    detector = VehicleDetector(
        model_path=model_cfg["model_path"],
        webhook_url=api_cfg["webhook_url"],
        confidence_threshold=model_cfg["confidence_threshold"],
        vehicle_classes=model_cfg["vehicle_classes"],
        class_names=model_cfg["class_names"],
        show_window=processing_cfg.get("show_window", False),
        dataset_mode=processing_cfg.get("dataset_mode", False),
        dataset_dir=processing_cfg.get("dataset_dir", "dataset")
    )

    detector.process_video(
        video_source=location["video_source"],
        location_name=location["name"],
        road_capacity=location["road_capacity"],
        interval_seconds=processing_cfg["interval_seconds"],
        frame_rate=processing_cfg.get("frame_rate", 1)
    )


if __name__ == "__main__":
    from .config import (
        LOCATIONS,
        MODEL_CONFIG,
        API_CONFIG,
        PROCESSING_CONFIG
    )

    processes = []

    for location in LOCATIONS.values():
        p = Process(
            target=run_camera,
            args=(location, MODEL_CONFIG, API_CONFIG, PROCESSING_CONFIG),
            daemon=True
        )
        p.start()
        processes.append(p)
        print(f"[BOOT] {location['name']}")

    for p in processes:
        p.join()