import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from datetime import datetime

st.set_page_config(page_title="AxiPulseAI Live Dashboard", layout="wide")
st.title("📊 AxiPulseAI Live Analytics")

API_URL = st.sidebar.text_input("API URL", "http://localhost:8000")
if st.sidebar.button("🔄 Fetch Live Data"):
    try:
        response = requests.get(f"{API_URL}/api/v1/system/status")
        st.success("✅ Connected to API")
    except:
        st.error("❌ Cannot connect to API")

st.sidebar.markdown("---")
st.sidebar.info("📊 Using mock data. Connect API for live data.")

# Generate mock data with trends
days = 30
dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
np.random.seed(42)
data = pd.DataFrame({
    'date': dates,
    'operational_health': 75 + np.cumsum(np.random.normal(0, 1, days)).clip(-15, 15) + np.random.normal(0, 3, days),
    'nps': 82 + np.cumsum(np.random.normal(0, 0.8, days)).clip(-10, 10) + np.random.normal(0, 2, days),
    'promoters': 550 + np.cumsum(np.random.normal(0, 5, days)).clip(-100, 100) + np.random.randint(-20, 20, days),
    'passives': 50 + np.random.randint(-15, 15, days),
    'detractors': 35 + np.random.randint(-10, 10, days),
    'calls': 2000 + np.cumsum(np.random.normal(0, 20, days)).clip(-400, 400) + np.random.randint(-100, 100, days)
})
numeric_cols = data.select_dtypes(include=[np.number]).columns
data[numeric_cols] = data[numeric_cols].clip(lower=0)

# Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("🏥 Health", f"{data['operational_health'].iloc[-1]:.1f}", f"{data['operational_health'].iloc[-1] - data['operational_health'].iloc[-2]:+.1f}")
col2.metric("⭐ NPS", f"{data['nps'].iloc[-1]:.1f}", f"{data['nps'].iloc[-1] - data['nps'].iloc[-2]:+.1f}")
col3.metric("📊 Promoters", f"{int(data['promoters'].iloc[-1]):,}", f"{int(data['promoters'].iloc[-1] - data['promoters'].iloc[-2]):+,d}")
col4.metric("📞 Calls", f"{int(data['calls'].iloc[-1]):,}", f"{int(data['calls'].iloc[-1] - data['calls'].iloc[-2]):+,d}")

# Charts
c1, c2 = st.columns(2)
with c1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['date'], y=data['operational_health'], mode='lines+markers', name='Health', line=dict(color='#1f77b4', width=3)))
    fig.add_hline(y=70, line_dash="dash", line_color="red")
    fig.add_hline(y=80, line_dash="dash", line_color="green")
    fig.update_layout(height=400, title="Operational Health Trend", xaxis_title="Date", yaxis_title="Health Index")
    st.plotly_chart(fig, use_container_width=True)

with c2:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['date'], y=data['nps'], mode='lines+markers', name='NPS', line=dict(color='#ff7f0e', width=3)))
    fig.update_layout(height=400, title="NPS Performance", xaxis_title="Date", yaxis_title="NPS Score")
    st.plotly_chart(fig, use_container_width=True)

# NPS Distribution
c1, c2 = st.columns(2)
with c1:
    last = data.iloc[-1]
    fig = go.Figure(data=[
        go.Bar(name='Promoters', x=['Distribution'], y=[last['promoters']], marker_color='green'),
        go.Bar(name='Passives', x=['Distribution'], y=[last['passives']], marker_color='orange'),
        go.Bar(name='Detractors', x=['Distribution'], y=[last['detractors']], marker_color='red')
    ])
    fig.update_layout(height=350, barmode='stack', title="NPS Distribution")
    st.plotly_chart(fig, use_container_width=True)

with c2:
    fig = go.Figure(data=[go.Bar(x=['Promoters', 'Passives', 'Detractors'], 
                                  y=[last['promoters'], last['passives'], last['detractors']],
                                  marker_color=['green', 'orange', 'red'])])
    fig.update_layout(height=350, title="NPS Breakdown", yaxis_title="Count")
    st.plotly_chart(fig, use_container_width=True)

# Data table
st.markdown("---")
st.subheader("📋 Historical Data")
st.dataframe(data.tail(10).style.background_gradient(cmap='Blues', subset=['operational_health', 'nps']), use_container_width=True)

# Download
csv = data.to_csv(index=False)
st.download_button("📥 Download Full Data", csv, f"axipulse_data_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")

st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
