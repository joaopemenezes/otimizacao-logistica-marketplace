"""
ETAPA 2 - ANÁLISE EXPLORATÓRIA (EDA)
Projeto: Otimização Logística de Marketplace (Olist)

Objetivo: identificar os principais gargalos logísticos do marketplace:
- Onde estão concentrados os atrasos (por estado)
- Relação entre distância e tempo/atraso de entrega
- Efeito do tempo de repasse à transportadora (handoff)
- Concentração geográfica de vendedores vs. clientes (o problema real)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

OUT_DIR = "../outputs"
plt.rcParams["figure.dpi"] = 120
plt.rcParams["font.size"] = 10

df = pd.read_csv(f"{OUT_DIR}/olist_consolidado.csv")

print("=" * 60)
print("RESUMO GERAL")
print("=" * 60)
print(f"Total de itens de pedido analisados: {len(df):,}")
print(f"Tempo médio de entrega: {df['delivery_time_days'].mean():.1f} dias")
print(f"Tempo médio estimado: {df['estimated_time_days'].mean():.1f} dias")
print(f"% de pedidos atrasados: {df['is_late'].mean()*100:.1f}%")
print(f"Distância média vendedor-cliente: {df['distance_km'].mean():.0f} km")
print(f"Frete médio: R$ {df['freight_value'].mean():.2f}")

# 1) CONCENTRAÇÃO GEOGRÁFICA: vendedores estão concentrados, clientes espalhados
print("\n" + "=" * 60)
print("CONCENTRAÇÃO DE VENDEDORES (top 5 estados)")
print("=" * 60)
seller_conc = df.drop_duplicates("seller_id")["seller_state"].value_counts(normalize=True).head(5) * 100
print(seller_conc.round(1))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
seller_share = df.drop_duplicates("seller_id")["seller_state"].value_counts(normalize=True).head(10) * 100
customer_share = df.drop_duplicates("customer_unique_id")["customer_state"].value_counts(normalize=True).head(10) * 100

axes[0].bar(seller_share.index, seller_share.values, color="#c0392b")
axes[0].set_title("% de Vendedores por Estado (top 10)")
axes[0].set_ylabel("% do total de vendedores")
axes[0].tick_params(axis="x", rotation=45)

axes[1].bar(customer_share.index, customer_share.values, color="#2980b9")
axes[1].set_title("% de Clientes por Estado (top 10)")
axes[1].set_ylabel("% do total de clientes")
axes[1].tick_params(axis="x", rotation=45)

plt.suptitle("O Desequilíbrio Geográfico: Oferta concentrada, Demanda espalhada", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/01_concentracao_geografica.png", bbox_inches="tight")
plt.close()

# 2) ATRASO POR ESTADO DO CLIENTE
print("\n" + "=" * 60)
print("% DE ATRASO POR ESTADO DO CLIENTE (top 10 piores)")
print("=" * 60)
late_by_state = df.groupby("customer_state")["is_late"].mean().sort_values(ascending=False) * 100
n_orders_by_state = df.groupby("customer_state").size()
late_by_state_filtered = late_by_state[n_orders_by_state >= 100]  # remove estados com poucos dados
print(late_by_state_filtered.head(10).round(1))

fig, ax = plt.subplots(figsize=(10, 6))
top_late = late_by_state_filtered.head(15)
colors = ["#c0392b" if v > df["is_late"].mean() * 100 else "#7f8c8d" for v in top_late.values]
ax.barh(top_late.index[::-1], top_late.values[::-1], color=colors[::-1])
ax.axvline(df["is_late"].mean() * 100, color="black", linestyle="--", linewidth=1, label="Média nacional")
ax.set_xlabel("% de pedidos entregues com atraso")
ax.set_title("Estados com Maior Índice de Atraso na Entrega", fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/02_atraso_por_estado.png", bbox_inches="tight")
plt.close()

# 3) DISTÂNCIA vs TEMPO DE ENTREGA (correlação)
print("\n" + "=" * 60)
print("RELAÇÃO DISTÂNCIA x TEMPO DE ENTREGA")
print("=" * 60)
corr = df[["distance_km", "delivery_time_days", "delay_days", "carrier_handoff_days"]].corr()
print(corr.round(2))

fig, ax = plt.subplots(figsize=(8, 6))
sample = df.sample(min(8000, len(df)), random_state=42)
sc = ax.scatter(sample["distance_km"], sample["delivery_time_days"], alpha=0.15, s=8, c=sample["is_late"], cmap="coolwarm")
ax.set_xlabel("Distância vendedor-cliente (km)")
ax.set_ylabel("Tempo de entrega (dias)")
ax.set_title("Distância vs. Tempo de Entrega\n(vermelho = entrega atrasada)", fontweight="bold")
ax.set_xlim(0, 3000)
ax.set_ylim(0, 60)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/03_distancia_vs_tempo.png", bbox_inches="tight")
plt.close()

# 4) DECOMPOSIÇÃO DO TEMPO: quanto tempo é "handoff" (loja->transportadora) vs transporte real
print("\n" + "=" * 60)
print("DECOMPOSIÇÃO DO TEMPO DE ENTREGA")
print("=" * 60)
print(f"Tempo médio até despachar p/ transportadora: {df['carrier_handoff_days'].mean():.1f} dias")
print(f"Tempo médio de transporte (transportadora->cliente): {df['carrier_to_customer_days'].mean():.1f} dias")

fig, ax = plt.subplots(figsize=(8, 5))
labels = ["Loja -> Transportadora\n(handoff)", "Transportadora -> Cliente\n(transporte)"]
values = [df["carrier_handoff_days"].mean(), df["carrier_to_customer_days"].mean()]
bars = ax.bar(labels, values, color=["#e67e22", "#16a085"])
for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.1, f"{v:.1f} dias", ha="center", fontweight="bold")
ax.set_ylabel("Dias (média)")
ax.set_title("Onde o Tempo é Gasto na Entrega", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/04_decomposicao_tempo.png", bbox_inches="tight")
plt.close()

# 5) TOP CATEGORIAS COM MAIS ATRASO
print("\n" + "=" * 60)
print("CATEGORIAS COM MAIOR % DE ATRASO (min. 200 pedidos)")
print("=" * 60)
cat_stats = df.groupby("product_category_name_english").agg(
    n=("order_id", "count"), late_pct=("is_late", "mean")
)
cat_stats = cat_stats[cat_stats["n"] >= 200].sort_values("late_pct", ascending=False)
print((cat_stats.head(10) * [1, 100]).round(1))

print("\n" + "=" * 60)
print("Gráficos salvos em outputs/: 01 a 04 .png")
print("=" * 60)
