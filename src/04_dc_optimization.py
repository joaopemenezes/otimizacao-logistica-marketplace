"""
ETAPA 3B - OTIMIZACAO DE CENTROS DE DISTRIBUICAO (DC)
Projeto: Otimização Logística de Marketplace (Olist)

Objetivo: hoje ~60% dos vendedores estão em SP, mas a demanda é nacional.
Vamos simular onde posicionar K centros de distribuição regionais para
minimizar a distância média até os clientes (e, por consequência, o frete
e o tempo de entrega), comparando com o cenário atual (vendedor único
concentrado em SP).
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from math import radians

OUT_DIR = "../outputs"

df = pd.read_csv(f"{OUT_DIR}/olist_consolidado.csv")


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


# ---------- 0) Modelo simples: custo de frete em função da distância ----------
freight_model = LinearRegression()
sample_fit = df.dropna(subset=["distance_km", "freight_value"]).sample(min(30000, len(df)), random_state=42)
freight_model.fit(sample_fit[["distance_km"]], sample_fit["freight_value"])
print("Modelo de frete: freight_value = %.4f * distance_km + %.2f (R2 informativo)" % (
    freight_model.coef_[0], freight_model.intercept_))

# Um ponto por cliente único (evita duplicar peso de clientes com vários itens)
customers = df.drop_duplicates("customer_unique_id")[
    ["customer_unique_id", "customer_lat", "customer_lng", "customer_state"]
].dropna()

# Volume de pedidos por cliente (peso)
weights = df.groupby("customer_unique_id").size().reindex(customers["customer_unique_id"]).values

coords = customers[["customer_lat", "customer_lng"]].values

# ---------- 1) CENÁRIO ATUAL: distância real vendedor -> cliente ----------
current_avg_distance = df["distance_km"].mean()
current_avg_freight = df["freight_value"].mean()

print("\n" + "=" * 60)
print("CENARIO ATUAL (vendedores concentrados, majoritariamente SP)")
print("=" * 60)
print(f"Distância média vendedor->cliente: {current_avg_distance:.0f} km")
print(f"Frete médio real: R$ {current_avg_freight:.2f}")

# ---------- 2) CENÁRIO OTIMIZADO: K centros de distribuição via k-means ponderado ----------
results = []
for k in [1, 3, 5, 8]:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(coords, sample_weight=weights)
    centers = km.cluster_centers_
    labels = km.labels_

    dists = haversine(
        coords[:, 0], coords[:, 1],
        centers[labels][:, 0], centers[labels][:, 1]
    )
    avg_dist_weighted = np.average(dists, weights=weights)
    avg_freight_est = freight_model.predict([[avg_dist_weighted]])[0]
    results.append({
        "k_centros": k,
        "distancia_media_km": avg_dist_weighted,
        "frete_medio_estimado": avg_freight_est,
        "centers": centers,
        "labels": labels,
    })
    print(f"\nK={k} centro(s) de distribuição:")
    print(f"  Distância média cliente->DC mais próximo: {avg_dist_weighted:.0f} km "
          f"({(1 - avg_dist_weighted/current_avg_distance)*100:.1f}% de redução vs. atual)")
    print(f"  Frete médio estimado: R$ {avg_freight_est:.2f} "
          f"({(1 - avg_freight_est/current_avg_freight)*100:.1f}% de redução vs. atual)")

best = next(r for r in results if r["k_centros"] == 5)

# Localização sugerida dos 5 DCs (cidade mais próxima do centróide, aproximado)
print("\n" + "=" * 60)
print("COORDENADAS SUGERIDAS PARA OS 5 CENTROS DE DISTRIBUICAO")
print("=" * 60)
for i, c in enumerate(best["centers"]):
    mask = best["labels"] == i
    n_customers_served = int(weights[mask].sum())
    print(f"  DC {i+1}: lat={c[0]:.3f}, lng={c[1]:.3f}  |  atende ~{n_customers_served:,} pedidos "
          f"({n_customers_served/weights.sum()*100:.1f}% da demanda)")

# ---------- 3) Gráfico: trade-off nº de DCs x distância/frete ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ks = [r["k_centros"] for r in results]
dists_ = [r["distancia_media_km"] for r in results]
freights_ = [r["frete_medio_estimado"] for r in results]

axes[0].plot(ks, dists_, marker="o", color="#2980b9")
axes[0].axhline(current_avg_distance, color="#c0392b", linestyle="--", label="Cenário atual")
axes[0].set_xlabel("Nº de Centros de Distribuição")
axes[0].set_ylabel("Distância média cliente-DC (km)")
axes[0].set_title("Redução de Distância por Nº de DCs")
axes[0].legend()

axes[1].plot(ks, freights_, marker="o", color="#27ae60")
axes[1].axhline(current_avg_freight, color="#c0392b", linestyle="--", label="Cenário atual")
axes[1].set_xlabel("Nº de Centros de Distribuição")
axes[1].set_ylabel("Frete médio estimado (R$)")
axes[1].set_title("Redução de Frete por Nº de DCs")
axes[1].legend()

plt.suptitle("Trade-off: Nº de Centros de Distribuição vs. Ganho Logístico", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/06_otimizacao_dc_trade_off.png", bbox_inches="tight")
plt.close()

# ---------- 4) Mapa: clientes coloridos por cluster + DCs ----------
fig, ax = plt.subplots(figsize=(9, 9))
sc = ax.scatter(coords[:, 1], coords[:, 0], c=best["labels"], cmap="tab10", s=3, alpha=0.3)
ax.scatter(best["centers"][:, 1], best["centers"][:, 0], c="black", s=250, marker="*",
           edgecolor="white", linewidth=1.5, label="Centros de Distribuição", zorder=5)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Alocação Ótima de 5 Centros de Distribuição\n(clientes coloridos por DC mais próximo)", fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/07_mapa_centros_distribuicao.png", bbox_inches="tight")
plt.close()

# Salva resultado para uso no dashboard
dc_df = pd.DataFrame(best["centers"], columns=["lat", "lng"])
dc_df["n_pedidos_atendidos"] = [int(weights[best["labels"] == i].sum()) for i in range(len(best["centers"]))]
dc_df.to_csv(f"{OUT_DIR}/centros_distribuicao_otimos.csv", index=False)

print("\nGráficos salvos: 06_otimizacao_dc_trade_off.png, 07_mapa_centros_distribuicao.png")
print("Centros ótimos salvos em: centros_distribuicao_otimos.csv")
