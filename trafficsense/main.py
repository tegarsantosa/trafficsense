import cv2
import numpy as np
import requests
import json
import yt_dlp

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
        show_window=False
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

    def get_youtube_stream_url(self, youtube_url):
        ydl_opts = {
            "format": "best[height<=720]",
            "quiet": True,
            "no_warnings": True
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

    def process_video(
        self,
        video_source,
        location_name,
        road_capacity,
        interval_seconds
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
        frame_interval = int(fps * interval_seconds)

        frame_count = 0
        vehicle_counts = []

        print(f"[START] {location_name}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections, count = self.detect_vehicles(frame)
            vehicle_counts.append(count)

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

                payload = {
                    "lokasi": location_name,
                    "waktu": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "jumlah_kendaraan": avg_count,
                    "congestion_index": ci,
                    "status": status
                }

                self.send_to_webhook(payload)
                vehicle_counts.clear()

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
        show_window=processing_cfg.get("show_window", False)
    )

    detector.process_video(
        video_source=location["video_source"],
        location_name=location["name"],
        road_capacity=location["road_capacity"],
        interval_seconds=processing_cfg["interval_seconds"]
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
