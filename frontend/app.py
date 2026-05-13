import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import os
from datetime import datetime

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="TrafficSense Dashboard",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f, #2d6a9f);
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
        margin: 5px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .metric-card-title {
        font-size: 14px;
        opacity: 0.9;
        margin-bottom: 10px;
    }
    .metric-card-value {
        font-size: 28px;
        font-weight: bold;
        margin: 10px 0;
    }
    .playback-bar {
        background: #0f2035;
        border-radius: 10px;
        padding: 12px 20px;
        margin-bottom: 12px;
        border: 1px solid #2d6a9f;
    }
    .chat-user {
        background-color: #2d6a9f;
        border-radius: 10px;
        padding: 10px;
        margin: 4px 0;
        color: white;
    }
    .chat-bot {
        background-color: #1e3a5f;
        border-radius: 10px;
        padding: 10px;
        margin: 4px 0;
        color: #ecf0f1;
    }
    .alert-box {
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        border-left: 4px solid;
    }
    .alert-high {
        background-color: #ffe6e6;
        border-left-color: #e74c3c;
        color: #c0392b;
    }
    .alert-medium {
        background-color: #fff3e6;
        border-left-color: #f39c12;
        color: #d68910;
    }
    .table-container {
        overflow-x: auto;
    }
    .status-low { color: #2ecc71; font-weight: bold; }
    .status-medium { color: #f39c12; font-weight: bold; }
    .status-high { color: #e74c3c; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

CHART_LAYOUT = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font_color="#ecf0f1",
    legend_bgcolor="rgba(0,0,0,0)",
)

TOLL_COLORS = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]


def fetch(endpoint, params=None):
    try:
        r = requests.get(f"{BACKEND_URL}{endpoint}", params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.warning(f"Error fetching {endpoint}: {str(e)}")
        return None


def post(endpoint, payload=None):
    try:
        r = requests.post(f"{BACKEND_URL}{endpoint}", json=payload or {}, timeout=240)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def status_color(status):
    return {"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}.get(status, "#95a5a6")


def status_html_class(status):
    return {"Low": "status-low", "Medium": "status-medium", "High": "status-high"}.get(status, "")


def render_playback_bar(key: str = "default"):
    pb = fetch("/playback/status")
    if not pb:
        st.error("Backend unavailable.")
        return

    current = pb.get("current_timestamp_index", 0)
    total = pb.get("total_timestamps", 1)
    is_playing = pb.get("is_playing", False)
    is_finished = pb.get("is_finished", False)
    latest_ts = pb.get("latest_timestamp", "—")
    interval = pb.get("interval_seconds", 5)

    with st.container():
        st.markdown('<div class="playback-bar">', unsafe_allow_html=True)
        col_info, col_progress, col_controls = st.columns([2, 4, 3])

        with col_info:
            st.markdown(f"**🕐 Timestamp:** `{latest_ts}`")
            st.markdown(f"**Frame:** {current} / {total}")

        with col_progress:
            progress = current / total if total > 0 else 0
            st.progress(progress, text=f"Playback {int(progress * 100)}%")
            if is_finished:
                st.caption("✅ Playback complete")
            elif is_playing:
                st.caption(f"▶ Playing — advancing every {interval}s")
            else:
                st.caption("⏸ Paused")

        with col_controls:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                if st.button("▶", help="Play", disabled=is_playing or is_finished, use_container_width=True, key=f"play_{key}"):
                    post("/playback/play")
                    st.rerun()
            with c2:
                if st.button("⏸", help="Pause", disabled=not is_playing, use_container_width=True, key=f"pause_{key}"):
                    post("/playback/pause")
                    st.rerun()
            with c3:
                if st.button("⏭", help="Step forward", disabled=is_finished, use_container_width=True, key=f"advance_{key}"):
                    post("/playback/advance")
                    st.rerun()
            with c4:
                if st.button("↺", help="Reset", use_container_width=True, key=f"reset_{key}"):
                    post("/playback/reset")
                    st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    if is_playing:
        st.empty()
        import time
        time.sleep(pb.get("interval_seconds", 5))
        st.rerun()


# ============ SIDEBAR ============
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/traffic-jam.png", width=80)
    st.title("TrafficSense")
    st.caption("Real-time Toll Road Analytics v2.0")
    st.divider()

    toll_data = fetch("/toll-roads")
    toll_roads = toll_data["toll_roads"] if toll_data else []
    selected_tol = st.selectbox("Filter Toll Road", ["All"] + toll_roads)

    st.divider()
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.rerun()

    st.markdown("---")
    st.caption("TrafficSense v2.0")


# ============ AUTO-PLAY INITIALIZATION ============
if "playback_started" not in st.session_state:
    st.session_state.playback_started = False

if not st.session_state.playback_started:
    try:
        requests.post(f"{BACKEND_URL}/playback/play")
        st.session_state.playback_started = True
    except:
        pass

# ============ MAIN TABS ============
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📈 Analytics", "🔮 Prediction", "🤖 AI Chat"])

# ============ TAB 1: DASHBOARD ============
with tab1:
    st.header("Traffic Dashboard")

    # Summary Cards
    st.subheader("📋 Summary Cards")
    stats = fetch("/statistics")
    
    if stats:
        summary = fetch("/summary")
        current_time = fetch("/current-time")
        
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-title">Total Toll Roads</div>
                <div class="metric-card-value">{stats.get('total_toll_roads', 0)}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            current_ts = current_time.get('current_time', 'N/A') if current_time else 'N/A'
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-title">Current Time</div>
                <div class="metric-card-value" style="font-size: 16px;">{current_ts}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            avg_ds = stats.get('average_congestion_index', 0)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-title">Average DS</div>
                <div class="metric-card-value">{avg_ds:.3f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            highest_tol = stats.get('highest_congestion_toll', 'N/A')
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-title">Highest Congestion</div>
                <div class="metric-card-value" style="font-size: 14px;">{highest_tol}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            high_count = stats.get('high_status_count', 0)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-title">High Status Count</div>
                <div class="metric-card-value" style="color: #e74c3c;">{high_count}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col6:
            last_updated = stats.get('last_updated', 'N/A')
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-card-title">Last Updated</div>
                <div class="metric-card-value" style="font-size: 12px;">{last_updated}</div>
            </div>
            """, unsafe_allow_html=True)

    # Traffic Alerts
    st.divider()
    st.subheader("⚠️ Traffic Alerts")
    alerts = fetch("/alerts")
    if alerts and alerts.get("alerts"):
        for alert in alerts["alerts"]:
            severity_class = "alert-high" if alert["severity"] == "high" else "alert-medium"
            st.markdown(f"""
            <div class="alert-box {severity_class}">
                <strong>{alert['title']}</strong><br>
                {alert['message']}<br>
                <small>{alert['timestamp']}</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("✅ No active traffic alerts")

    # Current Traffic Status Table
    st.divider()
    st.subheader("🚗 Current Traffic Status Per Toll")
    
    params = {} if selected_tol == "All" else {"nama_tol": selected_tol}
    raw = fetch("/data", params=params)
    
    if raw and raw["data"]:
        df = pd.DataFrame(raw["data"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Get latest data per toll road
        latest_df = df.sort_values("timestamp").groupby("nama_tol").last().reset_index()
        
        display_cols = ["nama_tol", "lokasi", "timestamp", "jumlah_kendaraan", 
                       "jumlah_bobot_kendaraan", "congestion_index", "status"]
        display_df = latest_df[display_cols].copy()
        display_df["timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        display_df["congestion_index"] = display_df["congestion_index"].round(4)
        display_df = display_df.rename(columns={
            "nama_tol": "Toll Road",
            "lokasi": "Location",
            "timestamp": "Timestamp",
            "jumlah_kendaraan": "Vehicle Count",
            "jumlah_bobot_kendaraan": "Weight Load",
            "congestion_index": "DS",
            "status": "Status"
        })
        
        # Color the status column
        status_colors = []
        for status in display_df["Status"]:
            status_colors.append(f"color: {status_color(status)};")
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Charts
    st.divider()
    st.subheader("📉 Analytics Charts")
    
    if raw and raw["data"]:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Congestion Index Line Chart**")
            fig = px.line(
                df, x="timestamp", y="congestion_index", color="nama_tol",
                markers=True, color_discrete_sequence=TOLL_COLORS,
                labels={"congestion_index": "Congestion Index (DS)", "timestamp": "Time", "nama_tol": "Toll Road"}
            )
            fig.add_hline(y=0.65, line_dash="dash", line_color="#f39c12", annotation_text="Medium", annotation_position="right")
            fig.add_hline(y=0.80, line_dash="dash", line_color="#e74c3c", annotation_text="High", annotation_position="right")
            fig.update_layout(**CHART_LAYOUT, height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Vehicle Volume Chart**")
            agg = df.groupby("nama_tol").agg(
                jumlah_mobil=("jumlah_mobil", "sum"),
                jumlah_bus=("jumlah_bus", "sum"),
                jumlah_truck=("jumlah_truck", "sum"),
            ).reset_index()
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(name="Car", x=agg["nama_tol"], y=agg["jumlah_mobil"], marker_color="#3498db"))
            fig2.add_trace(go.Bar(name="Bus", x=agg["nama_tol"], y=agg["jumlah_bus"], marker_color="#9b59b6"))
            fig2.add_trace(go.Bar(name="Truck", x=agg["nama_tol"], y=agg["jumlah_truck"], marker_color="#e67e22"))
            fig2.update_layout(barmode="stack", **CHART_LAYOUT, height=400)
            st.plotly_chart(fig2, use_container_width=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("**Status Distribution**")
            dist_data = fetch("/status-distribution", params=params if params else None)
            if dist_data and dist_data.get("distribution"):
                dist_df = pd.DataFrame(dist_data["distribution"])
                fig3 = px.pie(
                    dist_df, names="status", values="count",
                    color="status",
                    color_discrete_map={"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}
                )
                fig3.update_layout(**CHART_LAYOUT, height=400)
                st.plotly_chart(fig3, use_container_width=True)
        
        with col4:
            st.markdown("**Status Count Over Time**")
            status_time = df.groupby(["timestamp", "status"]).size().reset_index(name="count")
            fig4 = px.bar(
                status_time, x="timestamp", y="count", color="status",
                color_discrete_map={"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"},
                labels={"count": "Count", "timestamp": "Time", "status": "Status"}
            )
            fig4.update_layout(**CHART_LAYOUT, height=400, barmode="group")
            st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("No data visible yet. Press ▶ to start playback.")


# ============ TAB 2: ANALYTICS ============
with tab2:
    st.header("Trend Analysis")

    raw = fetch("/data")
    if raw and raw["data"]:
        df = pd.DataFrame(raw["data"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Congestion Index per Toll Road")
            fig = px.line(
                df, x="timestamp", y="congestion_index", color="nama_tol",
                facet_row="nama_tol", markers=True,
                color_discrete_sequence=TOLL_COLORS,
                labels={"congestion_index": "CI", "timestamp": "Time"}
            )
            fig.update_layout(height=600, showlegend=False, **CHART_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Vehicle Weight Load Over Time")
            fig2 = px.area(
                df, x="timestamp", y="jumlah_bobot_kendaraan", color="nama_tol",
                color_discrete_sequence=TOLL_COLORS,
                labels={"jumlah_bobot_kendaraan": "Weight Load", "timestamp": "Time"}
            )
            fig2.update_layout(**CHART_LAYOUT, height=600)
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Status Distribution per Toll Road")
        dist_data = fetch("/status-distribution")
        if dist_data and dist_data.get("distribution"):
            dist_df = pd.DataFrame(dist_data["distribution"])
            unique_tols = sorted(dist_df["nama_tol"].unique())
            pie_cols = st.columns(len(unique_tols))
            for i, tol in enumerate(unique_tols):
                with pie_cols[i]:
                    tol_dist = dist_df[dist_df["nama_tol"] == tol]
                    fig_pie = px.pie(
                        tol_dist, names="status", values="count", title=tol,
                        color="status", hole=0.4,
                        color_discrete_map={"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"}
                    )
                    fig_pie.update_layout(**CHART_LAYOUT)
                    st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No data visible yet. Press ▶ to start playback.")


# ============ TAB 3: MULTI-HORIZON PREDICTION ============
with tab3:
    st.header("Multi-Horizon Prediction")
    st.caption("Transformer model with 5, 15, and 30-minute prediction horizons")

    col_pred_tol, col_pred_btn = st.columns([3, 1])
    
    with col_pred_tol:
        pred_tol = st.selectbox("Select Toll Road", toll_roads if toll_roads else ["—"], key="pred_tol_select")
    
    with col_pred_btn:
        predict_btn = st.button("🔮 Predict", use_container_width=True, key="predict_multi_btn")

    if predict_btn or "last_prediction" in st.session_state:
        if predict_btn:
            with st.spinner("Generating predictions..."):
                pred_result = post("/predict-multi-horizon", {"nama_tol": pred_tol})
                st.session_state.last_prediction = pred_result
        else:
            pred_result = st.session_state.last_prediction

        if "error" in pred_result:
            st.error(f"Prediction failed: {pred_result['error']}")
        elif not pred_result.get("has_sufficient_data"):
            st.warning(f"⏳ {pred_result.get('message', 'Insufficient data for prediction')}")
        else:
            # Display current status
            st.divider()
            st.subheader("📍 Current Status")
            
            col_curr1, col_curr2, col_curr3, col_curr4 = st.columns(4)
            
            with col_curr1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-card-title">Toll Road</div>
                    <div class="metric-card-value" style="font-size: 16px;">{pred_result.get('nama_tol')}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_curr2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-card-title">Timestamp</div>
                    <div class="metric-card-value" style="font-size: 12px;">{pred_result.get('current_timestamp')}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_curr3:
                curr_ds = pred_result.get('current_congestion_index', 0)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-card-title">Current DS</div>
                    <div class="metric-card-value">{curr_ds:.4f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_curr4:
                curr_status = pred_result.get('current_status', 'Low')
                status_color_val = status_color(curr_status)
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-card-title">Status</div>
                    <div class="metric-card-value" style="color: {status_color_val};">{curr_status}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Display predictions
            st.divider()
            st.subheader("🔮 Predictions")
            
            predictions = pred_result.get('predictions', [])
            if predictions:
                pred_df = pd.DataFrame(predictions)
                pred_df_display = pred_df.rename(columns={
                    'horizon_minutes': 'Horizon (min)',
                    'predicted_congestion_index': 'Predicted DS',
                    'predicted_status': 'Predicted Status'
                })
                
                col_table, col_chart = st.columns([1, 2])
                
                with col_table:
                    st.dataframe(pred_df_display, use_container_width=True, hide_index=True)
                
                with col_chart:
                    fig_pred = go.Figure()
                    
                    for i, row in pred_df.iterrows():
                        status = row['predicted_status']
                        color = status_color(status)
                        fig_pred.add_trace(go.Bar(
                            x=[row['horizon_minutes']],
                            y=[row['predicted_congestion_index']],
                            name=f"{row['horizon_minutes']}min - {status}",
                            marker_color=color,
                            text=f"{row['predicted_congestion_index']:.4f}",
                            textposition="auto"
                        ))
                    
                    fig_pred.add_hline(y=0.65, line_dash="dash", line_color="#f39c12", annotation_text="Medium")
                    fig_pred.add_hline(y=0.80, line_dash="dash", line_color="#e74c3c", annotation_text="High")
                    fig_pred.update_layout(
                        title="Predicted Congestion Index by Horizon",
                        xaxis_title="Minutes Ahead",
                        yaxis_title="Congestion Index (DS)",
                        **CHART_LAYOUT,
                        height=400,
                        showlegend=True
                    )
                    st.plotly_chart(fig_pred, use_container_width=True)

    # Batch predictions
    st.divider()
    st.subheader("📊 Batch Multi-Horizon Predictions")
    
    if st.button("Run Predictions for All Toll Roads", use_container_width=True):
        with st.spinner("Predicting for all toll roads..."):
            all_predictions = []
            for tol in (toll_roads if toll_roads else []):
                pred = post("/predict-multi-horizon", {"nama_tol": tol})
                if pred.get("has_sufficient_data"):
                    for p in pred.get("predictions", []):
                        all_predictions.append({
                            "Toll Road": tol,
                            "Horizon": f"{p['horizon_minutes']}min",
                            "Predicted DS": p['predicted_congestion_index'],
                            "Status": p['predicted_status']
                        })
            
            if all_predictions:
                batch_df = pd.DataFrame(all_predictions)
                st.dataframe(batch_df, use_container_width=True, hide_index=True)
                
                # Visualization
                fig_batch = px.scatter(
                    batch_df, x="Horizon", y="Predicted DS", color="Status", facet_col="Toll Road",
                    color_discrete_map={"Low": "#2ecc71", "Medium": "#f39c12", "High": "#e74c3c"},
                    height=400
                )
                fig_batch.update_layout(**CHART_LAYOUT)
                st.plotly_chart(fig_batch, use_container_width=True)


# ============ TAB 4: AI CHATBOT ============
with tab4:
    st.header("AI Traffic Assistant")
    st.caption("Powered by Ollama LLM · Contextual insights from traffic data")

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    chat_container = st.container(height=420)
    with chat_container:
        if not st.session_state.chat_messages:
            st.markdown("""
            <div class="chat-bot">
            👋 Hi! I'm your TrafficSense AI Assistant. I analyze real-time traffic data to provide insights, predictions, and recommendations.<br><br>
            <strong>Try asking me:</strong><br>
            • Which toll road is most congested right now?<br>
            • What's the traffic trend on Tol JORR?<br>
            • When is the best time to travel?<br>
            • What are the predicted congestion levels?<br>
            • Which road should I avoid?
            </div>
            """, unsafe_allow_html=True)

        for msg in st.session_state.chat_messages:
            css = "chat-user" if msg["role"] == "user" else "chat-bot"
            icon = "👤" if msg["role"] == "user" else "🤖"
            st.markdown(f'<div class="{css}">{icon} {msg["content"]}</div>', unsafe_allow_html=True)

    col_input, col_btn = st.columns([5, 1])
    with col_input:
        user_input = st.text_input(
            "Ask about traffic...", key="chat_input_tab4",
            label_visibility="collapsed",
            placeholder="e.g., Which road has the highest congestion?"
        )
    with col_btn:
        send = st.button("Send", use_container_width=True, key="chat_send_btn")

    if send and user_input.strip():
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        with st.spinner("Thinking..."):
            response = post("/chat", {"messages": st.session_state.chat_messages})
        reply = response.get("reply", f"Error: {response.get('error', 'Unknown')}")
        st.session_state.chat_messages.append({"role": "assistant", "content": reply})
        st.rerun()

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.chat_messages = []
        st.rerun()


    