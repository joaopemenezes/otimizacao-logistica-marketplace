"""
ETAPA 3C - ROTEIRIZACAO (VRP)
Projeto: Otimização Logística de Marketplace (Olist)

Objetivo: dado um Centro de Distribuição e um conjunto de clientes a
atender em uma janela de tempo, encontrar rotas otimizadas de entrega
(Vehicle Routing Problem) usando OR-Tools, comparando com uma rota
"ingênua" (ordem sequencial / nearest neighbor simples).
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ortools.constraint_solver import routing_enums_pb2, pywrapcp

OUT_DIR = "../outputs"

df = pd.read_csv(f"{OUT_DIR}/olist_consolidado.csv")
dc_df = pd.read_csv(f"{OUT_DIR}/centros_distribuicao_otimos.csv")


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


# Escolhe o DC de maior volume (DC 5, região de SP/interior) para simular a rota do dia
dc = dc_df.sort_values("n_pedidos_atendidos", ascending=False).iloc[0]
depot = (dc["lat"], dc["lng"])

# Seleciona uma amostra de clientes próximos a esse DC para simular "as entregas do dia"
customers = df.drop_duplicates("customer_unique_id")[["customer_unique_id", "customer_lat", "customer_lng"]].dropna()
customers["dist_to_dc"] = haversine(customers["customer_lat"], customers["customer_lng"], depot[0], depot[1])
daily_stops = customers.nsmallest(500, "dist_to_dc").sample(18, random_state=7).reset_index(drop=True)

points = [depot] + list(zip(daily_stops["customer_lat"], daily_stops["customer_lng"]))
n = len(points)

# Matriz de distâncias (km) -> convertida para inteiros (metros) exigido pelo OR-Tools
dist_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if i != j:
            dist_matrix[i, j] = haversine(points[i][0], points[i][1], points[j][0], points[j][1])
dist_matrix_int = (dist_matrix * 1000).astype(int)  # metros

NUM_VEHICLES = 3
VEHICLE_CAPACITY_STOPS = 8  # paradas máximas por veículo


def solve_vrp():
    manager = pywrapcp.RoutingIndexManager(n, NUM_VEHICLES, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return dist_matrix_int[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Restrição de capacidade (nº de paradas por veículo)
    def demand_callback(from_index):
        node = manager.IndexToNode(from_index)
        return 0 if node == 0 else 1

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index, 0, [VEHICLE_CAPACITY_STOPS] * NUM_VEHICLES, True, "Capacity"
    )

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.seconds = 5

    solution = routing.SolveWithParameters(search_parameters)
    routes = []
    total_distance_m = 0
    if solution:
        for vehicle_id in range(NUM_VEHICLES):
            index = routing.Start(vehicle_id)
            route = []
            route_distance = 0
            while not routing.IsEnd(index):
                node = manager.IndexToNode(index)
                route.append(node)
                previous_index = index
                index = solution.Value(routing.NextVar(index))
                route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
            route.append(manager.IndexToNode(index))
            routes.append(route)
            total_distance_m += route_distance
    return routes, total_distance_m


routes, optimized_distance_m = solve_vrp()
optimized_distance_km = optimized_distance_m / 1000

# --- Rota "ingênua": visita todos os pontos em ordem sequencial, 1 único veículo ---
naive_order = list(range(n)) + [0]
naive_distance_km = sum(
    dist_matrix[naive_order[i], naive_order[i + 1]] for i in range(len(naive_order) - 1)
)

print("=" * 60)
print("SIMULACAO DE ROTEIRIZACAO (18 entregas, 1 DC, 3 veiculos)")
print("=" * 60)
print(f"Rota ingênua (sequencial, 1 veículo): {naive_distance_km:.1f} km")
print(f"Rota otimizada (OR-Tools, {NUM_VEHICLES} veículos): {optimized_distance_km:.1f} km")
print(f"Redução de distância percorrida: {(1 - optimized_distance_km/naive_distance_km)*100:.1f}%")

for i, route in enumerate(routes):
    stops = len(route) - 2
    print(f"  Veículo {i+1}: {stops} paradas | sequência de nós: {route}")

# --- Gráfico comparando as rotas ---
fig, axes = plt.subplots(1, 2, figsize=(13, 6))

lats = [p[0] for p in points]
lngs = [p[1] for p in points]

# Rota ingênua
axes[0].plot([points[i][1] for i in naive_order], [points[i][0] for i in naive_order],
             "-o", color="#c0392b", markersize=5, linewidth=1)
axes[0].scatter([depot[1]], [depot[0]], c="black", s=200, marker="s", zorder=5, label="Centro de Distribuição")
axes[0].set_title(f"Rota Ingênua (sequencial)\n{naive_distance_km:.0f} km percorridos", fontweight="bold")
axes[0].legend()

# Rotas otimizadas
colors = ["#2980b9", "#27ae60", "#8e44ad", "#e67e22"]
for i, route in enumerate(routes):
    route_lats = [points[node][0] for node in route]
    route_lngs = [points[node][1] for node in route]
    axes[1].plot(route_lngs, route_lats, "-o", color=colors[i % len(colors)], markersize=5, linewidth=1.5,
                 label=f"Veículo {i+1}")
axes[1].scatter([depot[1]], [depot[0]], c="black", s=200, marker="s", zorder=5, label="Centro de Distribuição")
axes[1].set_title(f"Rotas Otimizadas (OR-Tools, {NUM_VEHICLES} veículos)\n{optimized_distance_km:.0f} km percorridos", fontweight="bold")
axes[1].legend(fontsize=8)

plt.suptitle("Roteirização: Rota Ingênua vs. Otimizada (VRP)", fontweight="bold", fontsize=13)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/08_roteirizacao_vrp.png", bbox_inches="tight")
plt.close()

print("\nGráfico salvo: 08_roteirizacao_vrp.png")
