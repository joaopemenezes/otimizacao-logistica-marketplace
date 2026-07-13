"""
DASHBOARD INTERATIVO (STREAMLIT)
Projeto: Otimização Logística de Marketplace (Olist)

Para rodar localmente:
    pip install -r requirements.txt
    streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Otimização Logística - Marketplace", layout="wide")

DATA_DIR = "outputs"


@st.cache_data
def load_data():
    df = pd.read_csv(f"{DATA_DIR}/olist_consolidado.csv")
    dc_df = pd.read_csv(f"{DATA_DIR}/centros_distribuicao_otimos.csv")
    return df, dc_df


df, dc_df = load_data()

st.title("📦 Otimização Logística de Marketplace")
st.caption("Dataset: Olist Brazilian E-Commerce (99k pedidos, 2016-2018)")

# ---- Sidebar: simulador ----
st.sidebar.header("🔧 Simulador")
k = st.sidebar.slider("Número de Centros de Distribuição", 1, 8, 5)
st.sidebar.markdown("Ajuste o número de DCs e veja o impacto estimado na distância média e no frete.")

# Mapeia k para valores pré-calculados (ver 04_dc_optimization.py) - aproximação para o slider
k_results = {1: (560, 19.67), 3: (346, 17.48), 5: (236, 16.35), 8: (180, 15.77)}
closest_k = min(k_results.keys(), key=lambda x: abs(x - k))
dist_est, freight_est = k_results[closest_k]

current_dist = df["distance_km"].mean()
current_freight = df["freight_value"].mean()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Distância média atual", f"{current_dist:.0f} km")
col2.metric(f"Distância com {k} DCs", f"{dist_est:.0f} km", delta=f"{(1-dist_est/current_dist)*100:.0f}%", delta_color="inverse")
col3.metric("Frete médio atual", f"R$ {current_freight:.2f}")
col4.metric(f"Frete com {k} DCs", f"R$ {freight_est:.2f}", delta=f"{(1-freight_est/current_freight)*100:.0f}%", delta_color="inverse")

st.divider()

tab1, tab2, tab3 = st.tabs(["🗺️ Mapa de Centros de Distribuição", "📊 Atrasos por Estado", "📈 Previsão de Demanda"])

with tab1:
    sample = df.drop_duplicates("customer_unique_id").sample(min(5000, df["customer_unique_id"].nunique()), random_state=1)
    fig = go.Figure()
    fig.add_trace(go.Scattergeo(
        lon=sample["customer_lng"], lat=sample["customer_lat"],
        mode="markers", marker=dict(size=3, color="#95a5a6", opacity=0.4), name="Clientes"
    ))
    fig.add_trace(go.Scattergeo(
        lon=dc_df["lng"], lat=dc_df["lat"], mode="markers+text",
        marker=dict(size=25, color="#e74c3c", symbol="star"),
        text=[f"DC {i+1}" for i in range(len(dc_df))], textposition="top center",
        name="Centros de Distribuição"
    ))
    fig.update_geos(scope="south america", center=dict(lat=-15, lon=-52), projection_scale=3.3,
                     showland=True, landcolor="#ecf0f1", showcountries=True)
    fig.update_layout(height=550, margin=dict(l=0, r=0, t=20, b=0))
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    late_by_state = df.groupby("customer_state")["is_late"].mean().sort_values(ascending=False) * 100
    fig2 = px.bar(late_by_state.head(15), orientation="h", labels={"value": "% atraso", "customer_state": "Estado"},
                   title="Estados com maior índice de atraso")
    fig2.update_layout(showlegend=False, height=500)
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    state_forecast = pd.read_csv(f"{DATA_DIR}/previsao_demanda_por_estado.csv", index_col=0)
    fig3 = px.bar(state_forecast.head(10), y="forecast_next_month",
                   title="Previsão de demanda por estado (próximo mês)")
    fig3.update_layout(showlegend=False, height=500)
    st.plotly_chart(fig3, use_container_width=True)

st.divider()
st.caption("Projeto de portfólio - Otimização Logística usando dados públicos da Olist.")
