from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import httpx
import os
import threading
import time
from typing import Optional, List
from sqlalchemy import create_engine

app = FastAPI(title="TrafficSense Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# PostgreSQL connection settings
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "trafficsense")
DB_USER = os.getenv("DB_USER", "trafficsense")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

MODEL_TRANSFORMER_URL = os.getenv("MODEL_TRANSFORMER_URL", "http://localhost:8001")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
PLAYBACK_INITIAL_ROWS = int(os.getenv("PLAYBACK_INITIAL_ROWS", "10"))
PLAYBACK_INTERVAL_SECONDS = int(os.getenv("PLAYBACK_INTERVAL_SECONDS", "5"))


class PlaybackState:
    def __init__(self):
        self.lock = threading.Lock()
        self.all_data: pd.DataFrame = pd.DataFrame()
        self.visible_count: int = 0
        self.is_playing: bool = False
        self.is_finished: bool = False
        self.tick_thread: Optional[threading.Thread] = None

    def load(self):
        engine = create_engine(DATABASE_URL)
        query = "SELECT * FROM traffic ORDER BY timestamp"
        df = pd.read_sql(query, engine, parse_dates=["timestamp"])
        engine.dispose()
        with self.lock:
            self.all_data = df
            self.visible_count = min(PLAYBACK_INITIAL_ROWS, len(df))
            self.is_playing = False
            self.is_finished = False

    def get_visible(self) -> pd.DataFrame:
        with self.lock:
            return self.all_data.iloc[: self.visible_count].copy()

    def _num_toll_roads(self) -> int:
        if self.all_data.empty:
            return 1
        return self.all_data["nama_tol"].nunique()

    def advance(self) -> bool:
        with self.lock:
            total = len(self.all_data)
            step = self._num_toll_roads()
            next_count = self.visible_count + step
            if next_count >= total:
                self.visible_count = total
                self.is_finished = True
                self.is_playing = False
                return False
            self.visible_count = next_count
            return True

    def play(self):
        with self.lock:
            if self.is_playing or self.is_finished:
                return
            self.is_playing = True

        def tick():
            while True:
                with self.lock:
                    if not self.is_playing:
                        return
                can_continue = self.advance()
                if not can_continue:
                    return
                time.sleep(PLAYBACK_INTERVAL_SECONDS)

        self.tick_thread = threading.Thread(target=tick, daemon=True)
        self.tick_thread.start()

    def pause(self):
        with self.lock:
            self.is_playing = False

    def reset(self):
        with self.lock:
            self.is_playing = False
            self.visible_count = min(PLAYBACK_INITIAL_ROWS, len(self.all_data))
            self.is_finished = False

    def status_dict(self) -> dict:
        with self.lock:
            total = len(self.all_data)
            step = self._num_toll_roads()
            latest_ts = (
                self.all_data.iloc[self.visible_count - 1]["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
                if self.visible_count > 0 else None
            )
            return {
                "visible_rows": self.visible_count,
                "total_rows": total,
                "current_timestamp_index": self.visible_count // step,
                "total_timestamps": total // step,
                "is_playing": self.is_playing,
                "is_finished": self.is_finished,
                "latest_timestamp": latest_ts,
                "interval_seconds": PLAYBACK_INTERVAL_SECONDS,
            }


playback = PlaybackState()


@app.on_event("startup")
def startup():
    playback.load()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]


class PredictRequest(BaseModel):
    nama_tol: str
    minutes_ahead: int = 5


def get_data_context(df: pd.DataFrame) -> str:
    if df.empty:
        return "No data currently visible."
    lines = []
    for tol, group in df.groupby("nama_tol"):
        latest = group.sort_values("timestamp").iloc[-1]
        avg_ci = group["congestion_index"].mean()
        status_counts = group["status"].value_counts().to_dict()
        lines.append(
            f"- {tol}: latest CI={latest['congestion_index']:.3f}, "
            f"status={latest['status']}, avg CI={avg_ci:.3f}, "
            f"distribution={status_counts}"
        )
    return (
        f"TrafficSense Live Data (playback):\n"
        f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}\n"
        f"Visible records: {len(df)}\n"
        + "\n".join(lines)
        + "\nThresholds: Low<0.65, Medium 0.65-0.80, High>=0.80"
    )


@app.get("/health")
def health():
    return {"status": "ok", "service": "backend"}


@app.get("/playback/status")
def playback_status():
    return playback.status_dict()


@app.post("/playback/play")
def playback_play():
    playback.play()
    return playback.status_dict()


@app.post("/playback/pause")
def playback_pause():
    playback.pause()
    return playback.status_dict()


@app.post("/playback/reset")
def playback_reset():
    playback.reset()
    return playback.status_dict()


@app.post("/playback/advance")
def playback_advance():
    playback.advance()
    return playback.status_dict()


@app.get("/data")
def get_data(nama_tol: Optional[str] = None):
    df = playback.get_visible()
    if nama_tol:
        df = df[df["nama_tol"] == nama_tol]
    df = df.sort_values("timestamp")
    records = df.copy()
    records["timestamp"] = records["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return {"data": records.to_dict(orient="records")}


@app.get("/data/full")
def get_data_full(nama_tol: Optional[str] = None):
    """Get all data from database (not respecting playback state)"""
    engine = create_engine(DATABASE_URL)
    query = "SELECT * FROM traffic ORDER BY timestamp"
    df = pd.read_sql(query, engine, parse_dates=["timestamp"])
    engine.dispose()
    
    if nama_tol:
        df = df[df["nama_tol"] == nama_tol]
    
    df = df.sort_values("timestamp")
    records = df.copy()
    records["timestamp"] = records["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return {"data": records.to_dict(orient="records")}


@app.get("/toll-roads")
def get_toll_roads():
    df = playback.get_visible()
    return {"toll_roads": sorted(df["nama_tol"].unique().tolist())}


@app.get("/summary")
def get_summary():
    df = playback.get_visible()
    result = []
    for tol, group in df.groupby("nama_tol"):
        latest = group.sort_values("timestamp").iloc[-1]
        result.append({
            "nama_tol": tol,
            "latest_congestion_index": round(float(latest["congestion_index"]), 4),
            "latest_status": str(latest["status"]),
            "avg_congestion_index": round(float(group["congestion_index"].mean()), 4),
            "total_vehicles": int(group["jumlah_kendaraan"].sum()),
            "total_records": len(group),
        })
    return {"summary": result}


@app.get("/status-distribution")
def get_status_distribution(nama_tol: Optional[str] = None):
    df = playback.get_visible()
    if nama_tol:
        df = df[df["nama_tol"] == nama_tol]
    dist = df.groupby(["nama_tol", "status"]).size().reset_index(name="count")
    return {"distribution": dist.to_dict(orient="records")}


@app.post("/predict")
async def predict(req: PredictRequest):
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.post(
                f"{MODEL_TRANSFORMER_URL}/predict",
                json={"nama_tol": req.nama_tol, "minutes_ahead": req.minutes_ahead}
            )
            return resp.json()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Model transformer unavailable: {str(e)}")


@app.post("/predict-multi-horizon")
async def predict_multi_horizon(req: PredictRequest):
    """Get multi-horizon predictions (5, 15, 30 minutes)"""
    async with httpx.AsyncClient(timeout=60) as client:
        try:
            resp = await client.post(
                f"{MODEL_TRANSFORMER_URL}/predict-multi-horizon",
                json={"nama_tol": req.nama_tol}
            )
            return resp.json()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Model transformer unavailable: {str(e)}")


@app.get("/alerts")
def get_alerts():
    """Get traffic alerts based on current status and predictions"""
    df = playback.get_visible()
    alerts = []
    
    # Current HIGH status alerts
    for _, row in df[df["status"] == "High"].iterrows():
        alerts.append({
            "type": "current",
            "severity": "high",
            "title": f"Warning: {row['nama_tol']} is currently in High congestion",
            "message": f"{row['nama_tol']} KM {row.get('lokasi', 'Unknown')} - DS = {row['congestion_index']:.3f}",
            "toll_name": row['nama_tol'],
            "congestion_index": float(row['congestion_index']),
            "timestamp": row['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
        })
    
    return {"alerts": alerts}


@app.get("/current-time")
def get_current_time():
    """Get current simulation time from playback"""
    df = playback.get_visible()
    if df.empty:
        return {"current_time": None}
    latest = df.sort_values("timestamp").iloc[-1]
    return {"current_time": latest["timestamp"].strftime("%Y-%m-%d %H:%M:%S")}


@app.get("/statistics")
def get_statistics():
    """Get comprehensive statistics"""
    df = playback.get_visible()
    
    if df.empty:
        return {
            "total_toll_roads": 0,
            "average_congestion_index": 0,
            "high_status_count": 0,
            "medium_status_count": 0,
            "low_status_count": 0,
            "highest_congestion_toll": None
        }
    
    # Count total unique toll roads
    total_toll_roads = df["nama_tol"].nunique()
    
    # Average congestion index
    avg_ci = float(df["congestion_index"].mean())
    
    # Status counts
    status_counts = df["status"].value_counts()
    high_count = int(status_counts.get("High", 0))
    medium_count = int(status_counts.get("Medium", 0))
    low_count = int(status_counts.get("Low", 0))
    
    # Highest congestion toll (latest)
    latest_per_tol = df.sort_values("timestamp").groupby("nama_tol").last()
    highest_tol = latest_per_tol["congestion_index"].idxmax()
    
    return {
        "total_toll_roads": total_toll_roads,
        "average_congestion_index": round(avg_ci, 4),
        "high_status_count": high_count,
        "medium_status_count": medium_count,
        "low_status_count": low_count,
        "highest_congestion_toll": highest_tol,
        "last_updated": df["timestamp"].max().strftime("%Y-%m-%d %H:%M:%S")
    }


@app.post("/chat")
async def chat(req: ChatRequest):
    df = playback.get_visible()
    data_context = get_data_context(df)
    system_prompt = (
        "You are TrafficSense AI, an intelligent traffic analyst for Indonesian toll roads. "
        "You have access to the current playback snapshot of real-time traffic data.\n\n"
        f"CURRENT DATA CONTEXT:\n{data_context}\n\n"
        "Answer questions about traffic conditions, trends, predictions, and travel recommendations. "
        "Be concise, data-driven, and specific. Use the data context for accurate answers."
    )
    ollama_messages = [{"role": "system", "content": system_prompt}]
    for msg in req.messages:
        ollama_messages.append({"role": msg.role, "content": msg.content})
    async with httpx.AsyncClient(timeout=60) as client:
        try:
            resp = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": OLLAMA_MODEL,
                    "messages": ollama_messages,
                    "stream": False,
                    "options": {"thinking": False}
                }
            )
            result = resp.json()
            reply = result.get("message", {}).get("content", "No response.")
            return {"reply": reply, "model": OLLAMA_MODEL}
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Ollama unavailable: {str(e)}")
