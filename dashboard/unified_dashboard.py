"""
AxiPulseAI - Unified Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import requests
from datetime import datetime, timedelta
import json
import time

# Page config
st.set_page_config(
    page_title="AxiPulseAI - Unified Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .status-badge {
        padding: 0.25rem 1rem;
        border-radius: 20px;
        font-weight: bold;
    }
    .status-online {
        background-color: #28a745;
        color: white;
    }
    .status-offline {
        background-color: #dc3545;
        color: white;
    }
    .status-warning {
        background-color: #ffc107;
        color: black;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1f77b4;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🎯 AxiPulseAI Unified Dashboard</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚙️ Control Panel")
st.sidebar.markdown("---")

# API Configuration
API_URL = st.sidebar.text_input("🌐 API URL", "http://localhost:8000")
st.sidebar.markdown("---")

# Auto-refresh
auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh", value=False)
refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 5, 60, 10) if auto_refresh else 0

# Check API connection
def check_api():
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

api_connected = check_api()

# Status indicator
if api_connected:
    st.sidebar.success("✅ API Connected")
    st.sidebar.markdown(f"**Server:** `{API_URL}`")
else:
    st.sidebar.error("❌ API Disconnected")
    st.sidebar.info("Make sure API is running:\n```bash\nuvicorn api.main:app --host 0.0.0.0 --port 8000\n```")

st.sidebar.markdown("---")

# Navigation
st.sidebar.subheader("📋 Navigation")
page = st.sidebar.radio(
    "Select Dashboard",
)

st.sidebar.markdown("---")

# Quick actions
st.sidebar.subheader("🚀 Quick Actions")
if st.sidebar.button("🔄 Test All Endpoints"):
    st.sidebar.info("Testing endpoints...")

# ============================================================================
# PAGE: OVERVIEW
# ============================================================================

if page == "🏠 Overview":
    st.markdown('<h2 class="section-header">📊 System Overview</h2>', unsafe_allow_html=True)
    
    # System status cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-label">🏥 Engine 1</div>
                <div class="metric-value">Operational</div>
                <div class="metric-label">Health Predictor</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-label">⭐ Engine 2</div>
                <div class="metric-value">Operational</div>
                <div class="metric-label">NPS Predictor</div>
            </div>
        """, unsafe_allow_html=True)
    
        st.markdown("""
            <div class="metric-card">
                <div class="metric-value">Ready</div>
                <div class="metric-label">Data Generator</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
            <div class="metric-card">
                <div class="metric-label">🎯 System</div>
                <div class="metric-value">Online</div>
                <div class="metric-label">v1.0.0</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick access grid
    st.markdown("### 🚀 Quick Access")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**🏥 Health Predictor**\n\nPredict operational health index (0-100) from KPIs.")
        if st.button("Go to Health →", key="health_btn", use_container_width=True):
            page = "🏥 Health Predictor"
    
    with col2:
        st.info("**⭐ NPS Predictor**\n\nPredict NPS distribution and scores for tomorrow.")
        if st.button("Go to NPS →", key="nps_btn", use_container_width=True):
            page = "⭐ NPS Predictor"
    
    
    # API Status
    st.markdown("---")
    st.markdown("### 🔌 API Status")
    
    if api_connected:
        try:
            status = requests.get(f"{API_URL}/api/v1/system/status").json()
            st.json(status)
        except:
            st.warning("Could not fetch system status")
    else:
        st.error("API not connected. Start the API with:\n```bash\nuvicorn api.main:app --host 0.0.0.0 --port 8000\n```")

# ============================================================================
# PAGE: HEALTH PREDICTOR
# ============================================================================

elif page == "🏥 Health Predictor":
    st.markdown('<h2 class="section-header">🏥 Operational Health Predictor</h2>', unsafe_allow_html=True)
    
    if not api_connected:
        st.error("❌ API not connected. Please start the API first.")
        st.stop()
    
    st.info("Enter your KPIs to predict operational health index (0-100).")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📝 Input", "📊 Results", "📈 History"])
    
    with tab1:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🎯 Target Metrics")
            target_quality = st.number_input("Target Quality", 60.0, 100.0, 85.0, 1.0)
            target_competency = st.number_input("Target Competency", 60.0, 100.0, 80.0, 1.0)
            target_attendance = st.number_input("Target Attendance", 70.0, 100.0, 92.0, 1.0)
            target_release = st.number_input("Target Release Rate", 50.0, 100.0, 75.0, 1.0)
            target_transfer = st.number_input("Target Transfer Rate", 0.0, 13.0, 8.0, 0.5)
        
        with col2:
            st.subheader("📊 Actual Metrics")
            actual_quality = st.number_input("Actual Quality", 60.0, 100.0, 82.0, 1.0)
            actual_competency = st.number_input("Actual Competency", 60.0, 100.0, 78.0, 1.0)
            actual_attendance = st.number_input("Actual Attendance", 70.0, 100.0, 90.0, 1.0)
            actual_release = st.number_input("Actual Release Rate", 50.0, 100.0, 72.0, 1.0)
            actual_transfer = st.number_input("Actual Transfer Rate", 0.0, 13.0, 9.0, 0.5)
        
            st.subheader("🧠 Intelligence Factors")
            oif = st.slider("Operational Intelligence", 0.0, 1.0, 0.85, 0.05)
            bif = st.slider("Business Intelligence", 0.0, 1.0, 0.72, 0.05)
            mif = st.slider("Member Intelligence", 0.0, 1.0, 0.68, 0.05)
            total_calls = st.number_input("Total Calls Received", 0, 5000, 1850, 50)
        
        # Predict button
        if st.button("🚀 Predict Health", use_container_width=True, type="primary"):
            payload = {
                "target_quality": target_quality,
                "target_competency": target_competency,
                "target_attendance": target_attendance,
                "target_release_rate": target_release,
                "target_transfer_rate": target_transfer,
                "actual_quality": actual_quality,
                "actual_competency": actual_competency,
                "actual_attendance": actual_attendance,
                "actual_release_rate": actual_release,
                "actual_transfer_rate": actual_transfer,
                "total_calls_received": total_calls,
                "operational_intelligence_factor": oif,
                "business_intelligence_factor": bif,
                "member_intelligence_factor": mif
            }
            
            with st.spinner("Predicting..."):
                try:
                    response = requests.post(
                        f"{API_URL}/api/v1/health/predict",
                        json=payload,
                        timeout=10
                    )
                    if response.status_code == 200:
                        result = response.json()
                        st.session_state['health_result'] = result
                        st.success("✅ Prediction successful!")
                        st.balloons()
                    else:
                        st.error(f"❌ Error: {response.status_code}")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    
    with tab2:
        if 'health_result' in st.session_state:
            result = st.session_state['health_result']
            
            # Display results
            st.subheader("📊 Prediction Results")
            
            # Big metric
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "🏥 Operational Health",
                    f"{result.get('operational_health', 75.5):.1f}/100",
                    delta="Good"
                )
            with col2:
                st.metric(
                    "📊 Status",
                    "Stable" if result.get('operational_health', 75) > 70 else "At Risk",
                    delta="Normal" if result.get('operational_health', 75) > 70 else "Warning"
                )
                st.metric(
                    "⏰ Timestamp",
                    datetime.now().strftime("%H:%M:%S"),
                    delta=""
                )
            
            # Gauge chart
            st.subheader("📈 Health Gauge")
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = result.get('operational_health', 75.5),
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Operational Health Index"},
                delta = {'reference': 70},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 60], 'color': "red"},
                        {'range': [60, 70], 'color': "orange"},
                        {'range': [70, 85], 'color': "yellow"},
                        {'range': [85, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show raw response
            with st.expander("📋 Full Response"):
                st.json(result)
        else:
            st.info("👈 Enter values in the Input tab and click Predict.")
    
    with tab3:
        st.info("📈 Historical predictions will appear here after multiple predictions.")
        # Placeholder for history

# ============================================================================
# PAGE: NPS PREDICTOR
# ============================================================================

elif page == "⭐ NPS Predictor":
    st.markdown('<h2 class="section-header">⭐ NPS Predictor</h2>', unsafe_allow_html=True)
    
    if not api_connected:
        st.error("❌ API not connected. Please start the API first.")
        st.stop()
    
    st.info("Predict NPS distribution (0-10 scores) from operational health.")
    
    tab1, tab2 = st.tabs(["📝 Input", "📊 Results"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        st.subheader("🎯 Target KPI Inputs")
        target_quality_nps = st.number_input("Target Quality", 60.0, 100.0, 87.0, 1.0)
        target_competency_nps = st.number_input("Target Competency", 60.0, 100.0, 93.0, 1.0)
        target_attendance_nps = st.number_input("Target Attendance", 70.0, 100.0, 90.0, 1.0)
        target_release_nps = st.number_input("Target Release Rate", 50.0, 100.0, 60.0, 1.0)
        target_transfer_nps = st.number_input("Target Transfer Rate", 0.0, 20.0, 9.0, 0.5)

        st.subheader("📊 Actual KPI Inputs")
        actual_quality_nps = st.number_input("Actual Quality", 0.0, 100.0, 87.0, 1.0)
        actual_competency_nps = st.number_input("Actual Competency", 0.0, 100.0, 93.0, 1.0)
        actual_attendance_nps = st.number_input("Actual Attendance", 0.0, 100.0, 90.0, 1.0)
        actual_release_nps = st.number_input("Actual Release Rate", 0.0, 100.0, 60.0, 1.0)
        actual_transfer_nps = st.number_input("Actual Transfer Rate", 0.0, 20.0, 9.0, 0.5)

        st.subheader("⚙️ Operational Inputs")
        operational_health = st.number_input("Operational Health", 0.0, 120.0, 88.0, 0.5)
        bif_nps = st.slider("Business Intelligence", 0.0, 1.0, 0.72, 0.05)
        mif_nps = st.slider("Member Intelligence", 0.0, 1.0, 0.68, 0.05)
        total_calls_nps = st.number_input("Total Calls Received", 0, 20000, 2000, 50)
        
        if st.button("🚀 Predict NPS", use_container_width=True, type="primary"):
            payload = {
                "operational_health": operational_health,
                "business_intelligence_factor": bif_nps,
                "member_intelligence_factor": mif_nps,

                "target_quality": target_quality_nps,
                "target_competency": target_competency_nps,
                "target_attendance": target_attendance_nps,
                "target_release_rate": target_release_nps,
                "target_transfer_rate": target_transfer_nps,

                "actual_quality": actual_quality_nps,
                "actual_competency": actual_competency_nps,
                "actual_attendance": actual_attendance_nps,
                "actual_release_rate": actual_release_nps,
                "actual_transfer_rate": actual_transfer_nps,

                "release_gap": target_release_nps - actual_release_nps,
                "release_delta": 1.5,

                "quality_gap": target_quality_nps - actual_quality_nps,
                "competency_gap": target_competency_nps - actual_competency_nps,
                "attendance_gap": target_attendance_nps - actual_attendance_nps,
                "transfer_gap": actual_transfer_nps - target_transfer_nps,

                "total_calls_received": total_calls_nps
            }
            
            with st.spinner("Predicting NPS..."):
                try:
                    response = requests.post(
                        f"{API_URL}/api/v1/nps/predict",
                        json=payload,
                        timeout=10
                    )
                    if response.status_code == 200:
                        result = response.json()
                        st.session_state['nps_result'] = result
                        st.success("✅ NPS prediction successful!")
                        st.balloons()
                    else:
                        st.error(f"❌ Error: {response.status_code}")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    
    with tab2:
        if 'nps_result' in st.session_state:
            result = st.session_state['nps_result']
            
            st.subheader("📊 NPS Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "⭐ NPS Score",
                    f"{result.get('nps', 82.5):.1f}",
                    delta="Good"
                )
            with col2:
                st.metric(
                    "📊 Promoters",
                    f"{result.get('promoters', 623):.0f}",
                    delta="High"
                )
                st.metric(
                    "⚠️ Detractors",
                    f"{result.get('detractors', 32):.0f}",
                    delta="Low"
                )
            
            # NPS Distribution Chart
            st.subheader("📈 NPS Distribution")
            
            if 'distribution' in result:
                dist = result['distribution']
                scores = [f"Score {i}" for i in range(11)]
                values = [dist.get(f"score_{i}", 0) for i in range(11)]
                
                colors = ['red']*7 + ['orange']*2 + ['green']*2
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=scores,
                        y=values,
                        marker_color=colors,
                        text=values,
                        textposition='auto'
                    )
                ])
                fig.update_layout(
                    height=400,
                    title="NPS Score Distribution (0-10)",
                    xaxis_title="Scores",
                    yaxis_title="Number of Responses"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Show raw response
            with st.expander("📋 Full Response"):
                st.json(result)
        else:
            st.info("👈 Enter values in the Input tab and click Predict.")

# ============================================================================
# PAGE: ANALYTICS
# ============================================================================

elif page == "📈 Analytics":
    st.markdown('<h2 class="section-header">📈 Advanced Analytics</h2>', unsafe_allow_html=True)
    
    if not api_connected:
        st.error("❌ API not connected. Please start the API first.")
        st.stop()
    
    # Generate mock analytics data
    np.random.seed(42)
    days = 30
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    analytics_data = pd.DataFrame({
        'date': dates,
        'health': 75 + np.cumsum(np.random.normal(0, 0.5, days)).clip(-10, 10) + np.random.normal(0, 2, days),
        'nps': 82 + np.cumsum(np.random.normal(0, 0.3, days)).clip(-8, 8) + np.random.normal(0, 2, days),
        'calls': 2000 + np.cumsum(np.random.normal(0, 10, days)).clip(-300, 300),
        'quality': 82 + np.random.normal(0, 3, days).clip(65, 95),
        'release': 72 + np.random.normal(0, 3, days).clip(55, 90)
    })
    
    tab1, tab2, tab3 = st.tabs(["📊 Trends", "📈 Correlations", "📋 Data"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏥 Health Trend")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=analytics_data['date'],
                y=analytics_data['health'],
                mode='lines+markers',
                name='Health',
                line=dict(color='#1f77b4', width=3)
            ))
            fig.add_hline(y=70, line_dash="dash", line_color="red")
            fig.add_hline(y=80, line_dash="dash", line_color="green")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("⭐ NPS Trend")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=analytics_data['date'],
                y=analytics_data['nps'],
                mode='lines+markers',
                name='NPS',
                line=dict(color='#ff7f0e', width=3)
            ))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("📈 Correlation Matrix")
        corr = analytics_data[['health', 'nps', 'calls', 'quality', 'release']].corr()
        fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("📋 Historical Data")
        st.dataframe(analytics_data.tail(20))
        csv = analytics_data.to_csv(index=False)
        st.download_button("📥 Download CSV", csv, "analytics_data.csv")

# ============================================================================
# PAGE: SYSTEM
# ============================================================================

elif page == "⚙️ System":
    st.markdown('<h2 class="section-header">⚙️ System Management</h2>', unsafe_allow_html=True)
    
    if not api_connected:
        st.error("❌ API not connected.")
        st.stop()
    
    # System status
    try:
        status = requests.get(f"{API_URL}/api/v1/system/status").json()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📡 System Status", status.get('status', 'Unknown').upper())
        with col2:
            st.metric("🏥 Engine 1", status.get('components', {}).get('engine_1', {}).get('status', 'Unknown'))
            st.metric("⭐ Engine 2", status.get('components', {}).get('engine_2', {}).get('status', 'Unknown'))
        
        st.subheader("📋 System Details")
        st.json(status)
        
    except Exception as e:
        st.error(f"Could not fetch system status: {e}")
    
    st.markdown("---")
    
    # Quick commands
    st.subheader("🔧 Quick Commands")
    col1, col2 = st.columns(2)
    
    with col1:
        st.code("""
# Start API
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Check API
curl http://localhost:8000/health

# Test Health Predictor
curl -X POST http://localhost:8000/api/v1/health/predict \\
  -H "Content-Type: application/json" \\
  -d '{"target_quality":85,"target_competency":80,...}'
        """, language="bash")
    
    with col2:
        st.code("""
# Stop API
sudo fuser -k 8000/tcp

# View logs
tail -f logs/api.log

# Test NPS Predictor
curl -X POST http://localhost:8000/api/v1/nps/predict \\
  -H "Content-Type: application/json" \\
  -d '{"operational_health":76.4,...}'
        """, language="bash")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    f"<center>🎯 AxiPulseAI Unified Dashboard v1.0 | "
    f"API: {API_URL} | "
    f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
    f"Status: {'🟢 Online' if api_connected else '🔴 Offline'}</center>",
    unsafe_allow_html=True
)

# Auto-refresh
if auto_refresh and api_connected:
    time.sleep(refresh_interval)
    st.rerun()
