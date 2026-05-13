# TrafficSense

Real-time toll road traffic analytics platform with ML prediction and AI chatbot.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FRONTEND (Streamlit)                      │
│         Web Dashboard with 4 tabs, Charts, AI Chat           │
│                    Port: 8501                                │
└────────────────────────┬────────────────────────────────────┘
                         │
        HTTP Requests (REST API)
                         │
    ┌────────────────────┴───────────────────┐
    ↓                                        ↓
┌──────────────────────┐        ┌──────────────────────┐
│  BACKEND (FastAPI)   │        │ MODEL SERVICE        │
│   Port: 8000         │        │ (Transformer)        │
│ Routes HTTP requests │        │ Port: 8001           │
│ to model service     │        │ Runs ML predictions  │
│ & Ollama chatbot     │        │ (MLPRegressor)       │
└──────┬───────────────┘        └────────┬─────────────┘
       │                                 │
       └──────────────────┬──────────────┘
                          │
                ┌─────────┴──────────┐
                ↓                    ↓
        ┌───────────────┐   ┌──────────────────┐
        │ Ollama LLM    │   │ traffic.csv Data │
        │ (AI Chatbot)  │   │ (Playback)       │
        └───────────────┘   └──────────────────┘
```

## Services

| Service | Port | Description |
|---|---|---|
| `frontend` | 8501 | Streamlit dashboard |
| `backend` | 8000 | FastAPI data + chat proxy |
| `model_transformer` | 8001 | ML prediction service |
| Ollama (external) | 11434 | LLM for chatbot |

## Prerequisites

- Docker & Docker Compose
- [Ollama](https://ollama.com) installed and running locally

## Setup

### 1. Install and start Ollama

```bash
# Install Ollama (Linux/Mac)
curl -fsSL https://ollama.com/install.sh | sh

# Pull the model
ollama pull llama3.2

# Ollama runs automatically on port 11434
```

### 2. Prepare your data

Place your `traffic.csv` file in the `data/` folder.  
The sample file included has the expected schema:

```
id, nama_tol, lokasi, timestamp, jumlah_mobil, jumlah_bus,
jumlah_truck, jumlah_kendaraan, jumlah_bobot_kendaraan,
congestion_index, status
```

**Status thresholds:**
- `Low` — congestion_index < 0.65
- `Medium` — 0.65 ≤ congestion_index < 0.80
- `High` — congestion_index ≥ 0.80

### 3. Run all services

```bash
docker-compose up --build
```

Open your browser at **http://localhost:8501**

## API Reference

### Backend (http://localhost:8000)

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/data` | Get traffic records (`?nama_tol=` optional) |
| GET | `/toll-roads` | List all toll roads |
| GET | `/summary` | Latest stats per toll road |
| GET | `/status-distribution` | Status counts per toll road |
| POST | `/predict` | Run prediction via model transformer |
| POST | `/chat` | Chat with Ollama AI assistant |

### Model Transformer (http://localhost:8001)

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/train` | Train the regression model |
| POST | `/predict` | Predict congestion index |
| GET | `/toll-roads` | List toll roads in model |
| GET | `/summary` | Latest summary from data |

## Dashboard Tabs

1. **Dashboard** — Live metrics, congestion line chart, stacked vehicle bar chart, status pie chart, latest records table
2. **Trend Analysis** — Per-road faceted trend, weight load area chart, per-road status donut charts
3. **Prediction** — Single toll road prediction with slider, batch prediction chart across all roads
4. **AI Chatbot** — Ollama-powered assistant with full data context injected into every prompt

## Environment Variables

### backend
| Variable | Default | Description |
|---|---|---|
| `DATA_PATH` | `../data/traffic.csv` | CSV data file |
| `MODEL_TRANSFORMER_URL` | `http://localhost:8001` | Model service URL |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama URL |
| `OLLAMA_MODEL` | `llama3.2` | Model name |

### model_transformer
| Variable | Default | Description |
|---|---|---|
| `DATA_PATH` | `../data/traffic.csv` | CSV data file |

### frontend
| Variable | Default | Description |
|---|---|---|
| `BACKEND_URL` | `http://localhost:8000` | Backend URL |

## Development (without Docker)

```bash
# Terminal 1 - Model Transformer
cd model_transformer
pip install -r requirements.txt
DATA_PATH=../data/traffic.csv uvicorn main:app --port 8001

# Terminal 2 - Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --port 8000

# Terminal 3 - Frontend
cd frontend
pip install -r requirements.txt
BACKEND_URL=http://localhost:8000 streamlit run app.py
```
