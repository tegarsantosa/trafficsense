# TrafficSense

Real-time traffic monitoring system using YOLO object detection on YouTube live streams.

## Prerequisites

- Python 3.8+
- pip

## Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
```

### 2. Activate Virtual Environment

**Linux/Mac:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Configuration

Edit `trafficsense/config.py` to configure your monitoring locations and settings.

### Example Configuration

```python
# trafficsense/config.py

LOCATIONS = {
    "one": {
        "name": "Traffic Light Cibinong",
        "video_source": "https://www.youtube.com/watch?v=VM9G7KRyWBQ",
        "road_capacity": 10,
        "latitude": -6.908540,
        "longitude": 110.583330
    },
    "two": {
        "name": "Traffic Light Cikampek",
        "video_source": "https://www.youtube.com/watch?v=j12j3bi3zAI",
        "road_capacity": 10,
        "latitude": -8.568540,
        "longitude": 111.143330
    },
}

MODEL_CONFIG = {
    "model_path": "yolov8n.pt",
    "confidence_threshold": 0.5,
    "vehicle_classes": [2, 3, 5, 7],  # car, motorcycle, bus, truck
    "class_names": {
        2: "car",
        3: "motorcycle", 
        5: "bus",
        7: "truck"
    }
}

API_CONFIG = {
    "webhook_url": "http://0.0.0.0:5000/webhook",
    "host": "0.0.0.0",
    "port": 5000,
    "debug": True
}

PROCESSING_CONFIG = {
    "interval_seconds": 30,  # Send data every 30 seconds
    "show_window": True,     # Show detection window
    "batch_size": 1,
    "save_detections": False,
    "output_dir": "detections/"
}
```

### What to Change

- **LOCATIONS**: Add more locations by copying the pattern. Each needs a unique key ("one", "two", etc.)
- **road_capacity**: Adjust based on actual road capacity for accurate congestion calculation
- **interval_seconds**: Change how often data is sent (in seconds)
- **show_window**: Set to `False` to run headless (no GUI)

## Running the Application

```bash
honcho start
```

This starts both:
- Traffic detection service (processes video streams)
- API server (receives and stores data)

## API Access

Once running, access the API at: `http://localhost:5000`

### Available Endpoints

- `GET /traffic/latest` - Get all latest traffic data
- `GET /traffic/location/<name>` - Get data for specific location
- `GET /traffic/status/<status>` - Get locations by congestion status
- `GET /health` - Check API health
- `DELETE /traffic/clear` - Clear all data (testing)

### Example

```bash
curl http://localhost:5000/traffic/latest
```

## Stopping the Application

Press `Ctrl+C` in the terminal running Honcho.

## Database

Traffic data is stored in `db/traffic.db` (SQLite).