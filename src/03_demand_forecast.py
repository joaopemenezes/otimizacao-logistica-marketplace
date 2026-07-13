"""
ETAPA 3A - PREVISAO DE DEMANDA
Projeto: Otimização Logística de Marketplace (Olist)

Objetivo: prever o volume de pedidos dos próximos meses (nacional e por estado),
para dimensionar centros de distribuição e capacidade logística.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

OUT_DIR = "../outputs"

df = pd.read_csv(f"{OUT_DIR}/olist_consolidado.csv", parse_dates=["order_purchase_timestamp"])
df["year_month"] = df["order_purchase_timestamp"].dt.to_period("M").astype(str)

# Remove os 2 últimos meses (dados incompletos/residuais no dataset original)
valid_months = sorted(df["year_month"].unique())[:-2]
df = df[df["year_month"].isin(valid_months)]

# ---------- 1) Série nacional mensal ----------
monthly = df.groupby("year_month").size().reset_index(name="n_orders")
monthly["month_idx"] = range(len(monthly))
monthly["month_of_year"] = pd.to_datetime(monthly["year_month"]).dt.month

print("=" * 60)
print("SERIE MENSAL NACIONAL DE PEDIDOS")
print("=" * 60)
print(monthly[["year_month", "n_orders"]].to_string(index=False))

# Treino/teste: últimos 3 meses como teste
train = monthly.iloc[:-3]
test = monthly.iloc[-3:]

model = RandomForestRegressor(n_estimators=300, max_depth=4, random_state=42)
model.fit(train[["month_idx", "month_of_year"]], train["n_orders"])
pred_test = model.predict(test[["month_idx", "month_of_year"]])

mae = mean_absolute_error(test["n_orders"], pred_test)
mape = mean_absolute_percentage_error(test["n_orders"], pred_test) * 100
print(f"\nValidação (últimos 3 meses): MAE={mae:.0f} pedidos | MAPE={mape:.1f}%")

# Refit com todos os dados e prever os próximos 3 meses
model_full = RandomForestRegressor(n_estimators=300, max_depth=4, random_state=42)
model_full.fit(monthly[["month_idx", "month_of_year"]], monthly["n_orders"])

future_idx = range(monthly["month_idx"].max() + 1, monthly["month_idx"].max() + 4)
future_month_of_year = [(monthly["month_of_year"].iloc[-1] + i - 1) % 12 + 1 for i in range(1, 4)]
future = pd.DataFrame({"month_idx": future_idx, "month_of_year": future_month_of_year})
future["forecast"] = model_full.predict(future[["month_idx", "month_of_year"]])

print("\nPrevisão para os próximos 3 meses (nacional):")
print(future[["month_idx", "forecast"]].round(0).to_string(index=False))

fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(monthly["year_month"], monthly["n_orders"], marker="o", label="Histórico", color="#2c3e50")
future_labels = [f"F+{i+1}" for i in range(3)]
ax.plot(future_labels, future["forecast"], marker="o", linestyle="--", color="#e74c3c", label="Previsão")
ax.set_xticklabels(list(monthly["year_month"]) + future_labels, rotation=60, ha="right")
ax.set_ylabel("Nº de pedidos")
ax.set_title(f"Previsão de Demanda Nacional (MAPE de validação: {mape:.1f}%)", fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/05_previsao_demanda_nacional.png", bbox_inches="tight")
plt.close()

# ---------- 2) Participação por estado (para alocar a previsão nacional regionalmente) ----------
state_share = df.groupby("customer_state").size().sort_values(ascending=False)
state_share_pct = (state_share / state_share.sum() * 100).round(2)

print("\n" + "=" * 60)
print("PARTICIPAÇÃO DE CADA ESTADO NA DEMANDA (top 10)")
print("=" * 60)
print(state_share_pct.head(10))

next_month_forecast = future["forecast"].iloc[0]
state_forecast = (state_share_pct / 100 * next_month_forecast).round(0).sort_values(ascending=False)
state_forecast.to_csv(f"{OUT_DIR}/previsao_demanda_por_estado.csv", header=["forecast_next_month"])

print(f"\nPrevisão do próximo mês distribuída por estado (salvo em previsao_demanda_por_estado.csv)")
print(state_forecast.head(10))

print("\nGráfico salvo: 05_previsao_demanda_nacional.png")
