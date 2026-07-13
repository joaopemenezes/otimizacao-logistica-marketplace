"""
ETAPA 4 - DASHBOARD INTERATIVO
Projeto: Otimização Logística de Marketplace (Olist)

Gera um dashboard HTML standalone (Plotly) com mapa dos centros de
distribuição, KPIs antes/depois e principais gráficos do projeto.
Não requer servidor - basta abrir o .html no navegador.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

OUT_DIR = "../outputs"

df = pd.read_csv(f"{OUT_DIR}/olist_consolidado.csv")
dc_df = pd.read_csv(f"{OUT_DIR}/centros_distribuicao_otimos.csv")
state_forecast = pd.read_csv(f"{OUT_DIR}/previsao_demanda_por_estado.csv", index_col=0)

current_avg_distance = df["distance_km"].mean()
current_avg_freight = df["freight_value"].mean()
optimized_avg_distance = 236  # k=5, calculado na etapa 3b
optimized_avg_freight = 16.35
late_pct = df["is_late"].mean() * 100

customers_sample = df.drop_duplicates("customer_unique_id").sample(6000, random_state=1)

fig = make_subplots(
    rows=2, cols=2,
    specs=[[{"type": "scattergeo", "rowspan": 2}, {"type": "indicator"}],
           [None, {"type": "bar"}]],
    column_widths=[0.62, 0.38],
    row_heights=[0.5, 0.5],
    subplot_titles=("Centros de Distribuição Otimizados vs. Clientes", "", "Impacto Estimado da Otimização"),
    horizontal_spacing=0.08,
)

# Mapa: clientes
fig.add_trace(go.Scattergeo(
    lon=customers_sample["customer_lng"], lat=customers_sample["customer_lat"],
    mode="markers", marker=dict(size=3, color="#95a5a6", opacity=0.4),
    name="Clientes", showlegend=True
), row=1, col=1)

# Mapa: DCs otimizados
fig.add_trace(go.Scattergeo(
    lon=dc_df["lng"], lat=dc_df["lat"],
    mode="markers+text",
    marker=dict(size=dc_df["n_pedidos_atendidos"] / dc_df["n_pedidos_atendidos"].max() * 40 + 15,
                color="#e74c3c", symbol="star", line=dict(width=1, color="white")),
    text=[f"DC {i+1}" for i in range(len(dc_df))],
    textposition="top center",
    name="Centros de Distribuição (novos)",
    showlegend=True
), row=1, col=1)

fig.update_geos(
    scope="south america", center=dict(lat=-15, lon=-52),
    projection_scale=3.3, showland=True, landcolor="#ecf0f1",
    showcountries=True, countrycolor="#bdc3c7", row=1, col=1
)

# Indicador: redução de distância
fig.add_trace(go.Indicator(
    mode="number+delta",
    value=optimized_avg_distance,
    number={"suffix": " km", "font": {"size": 40}},
    delta={"reference": current_avg_distance, "relative": True, "valueformat": ".1%",
           "increasing": {"color": "#c0392b"}, "decreasing": {"color": "#27ae60"}},
    title={"text": "Distância média cliente-DC<br><span style='font-size:12px'>(atual: %.0f km)</span>" % current_avg_distance},
), row=1, col=2)

# Barra: comparação frete atual vs otimizado
fig.add_trace(go.Bar(
    x=["Cenário Atual", "Cenário Otimizado (5 DCs)"],
    y=[current_avg_freight, optimized_avg_freight],
    marker_color=["#c0392b", "#27ae60"],
    text=[f"R$ {current_avg_freight:.2f}", f"R$ {optimized_avg_freight:.2f}"],
    textposition="outside",
    showlegend=False
), row=2, col=2)
fig.update_yaxes(title_text="Frete médio (R$)", row=2, col=2)

fig.update_layout(
    height=650, width=1150,
    title=dict(
        text="<b>Dashboard - Otimização Logística de Marketplace (Olist)</b><br>"
             "<span style='font-size:13px'>Simulação: rede de vendedor único (SP) → 5 centros de distribuição regionais</span>",
        x=0.02
    ),
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=-0.05, xanchor="left", x=0)
)

fig.write_html(f"{OUT_DIR}/dashboard.html", include_plotlyjs="cdn")
print("Dashboard salvo em outputs/dashboard.html")

# KPIs resumidos em texto (para o README)
kpis = f"""
RESUMO DE KPIs - OTIMIZACAO LOGISTICA
======================================
Distancia media cliente-DC:  {current_avg_distance:.0f} km -> {optimized_avg_distance:.0f} km   (-{(1-optimized_avg_distance/current_avg_distance)*100:.0f}%)
Frete medio estimado:        R$ {current_avg_freight:.2f} -> R$ {optimized_avg_freight:.2f}   (-{(1-optimized_avg_freight/current_avg_freight)*100:.0f}%)
% pedidos atrasados (atual):  {late_pct:.1f}%
Reducao em roteirizacao (VRP, simulacao 18 entregas/3 veiculos): 37.7%
Previsao de demanda (proximo mes, nacional): {state_forecast['forecast_next_month'].sum():.0f} pedidos (MAPE validacao: 3.3%)
"""
with open(f"{OUT_DIR}/kpis_resumo.txt", "w") as f:
    f.write(kpis)
print(kpis)
