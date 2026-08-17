#!/bin/bash

echo "📊 Starting AxiPulseAI Dashboard..."

# Install dependencies
pip install -q streamlit plotly pandas numpy 2>/dev/null

# Create simple dashboard if not exists
if [ ! -f "dashboard/app.py" ]; then
    cat > dashboard/app.py << 'DASH'
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import requests

st.set_page_config(page_title="AxiPulseAI Dashboard", layout="wide")
st.title("📊 AxiPulseAI Analytics Dashboard")

# Sidebar
st.sidebar.title("⚙️ Controls")
API_URL = st.sidebar.text_input("API URL", "http://localhost:8000")
days = st.sidebar.slider("Days", 7, 90, 30)
refresh = st.sidebar.button("🔄 Refresh Data")

# Generate or fetch data
@st.cache_data
def generate_data(days):
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    data = pd.DataFrame({
        'date': dates,
        'health': 75 + np.cumsum(np.random.normal(0, 0.5, days)).clip(-15, 15) + np.random.normal(0, 3, days),
        'nps': 82 + np.cumsum(np.random.normal(0, 0.3, days)).clip(-10, 10) + np.random.normal(0, 2, days),
        'promoters': 550 + np.cumsum(np.random.normal(0, 3, days)).clip(-100, 100),
        'passives': 50 + np.random.randint(-15, 15, days),
        'detractors': 35 + np.random.randint(-10, 10, days),
        'calls': 2000 + np.cumsum(np.random.normal(0, 10, days)).clip(-400, 400)
    })
    return data.clip(lower=0)

data = generate_data(days)

# Metrics
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("🏥 Health", f"{data['health'].iloc[-1]:.1f}", f"{data['health'].iloc[-1]-data['health'].iloc[-2]:+.1f}")
with c2:
    st.metric("⭐ NPS", f"{data['nps'].iloc[-1]:.1f}", f"{data['nps'].iloc[-1]-data['nps'].iloc[-2]:+.1f}")
with c3:
    st.metric("📊 Promoters", f"{data['promoters'].iloc[-1]:,}")
with c4:
    st.metric("📞 Calls", f"{data['calls'].iloc[-1]:,}")

# Charts
c1, c2 = st.columns(2)
with c1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['date'], y=data['health'], mode='lines+markers', name='Health', line=dict(color='#1f77b4', width=3)))
    fig.add_hline(y=70, line_dash="dash", line_color="red")
    fig.add_hline(y=80, line_dash="dash", line_color="green")
    fig.update_layout(height=400, title="Health Trend")
    st.plotly_chart(fig, use_container_width=True)

with c2:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['date'], y=data['nps'], mode='lines+markers', name='NPS', line=dict(color='#ff7f0e', width=3)))
    fig.update_layout(height=400, title="NPS Trend")
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
    fig = go.Figure(data=[
        go.Bar(x=['Promoters', 'Passives', 'Detractors'], 
               y=[last['promoters'], last['passives'], last['detractors']],
               marker_color=['green', 'orange', 'red'])
    ])
    fig.update_layout(height=350, title="NPS Breakdown")
    st.plotly_chart(fig, use_container_width=True)

# Correlations
st.markdown("---")
st.subheader("📊 Correlations")
corr = data[['health', 'nps', 'calls']].corr()
fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
fig.update_layout(height=300)
st.plotly_chart(fig, use_container_width=True)

# Data
st.markdown("---")
st.subheader("📋 Recent Data")
st.dataframe(data.tail(10))
st.download_button("📥 Download", data.to_csv(index=False), "axipulse_data.csv")

st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
DASH
fi

# Start dashboard
streamlit run dashboard/app.py --server.port 8501
