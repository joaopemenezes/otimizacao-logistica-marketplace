"""
ETAPA 1 - PIPELINE DE DADOS
Projeto: Otimização Logística de Marketplace (Olist)

Objetivo: consolidar as 8 tabelas do Olist em um dataset único por item de pedido,
calcular a distância real vendedor->cliente (Haversine) e as métricas de
performance de entrega (tempo real, atraso vs. estimativa, etc.)
"""

import pandas as pd
import numpy as np
from math import radians, cos, sin, asin, sqrt

DATA_DIR = "../data"
OUT_DIR = "../outputs"

pd.set_option("display.max_columns", None)


def haversine(lat1, lon1, lat2, lon2):
    """Distância em km entre dois pontos geográficos (fórmula de Haversine)."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c  # raio da Terra em km


def load_data():
    orders = pd.read_csv(f"{DATA_DIR}/olist_orders_dataset.csv", parse_dates=[
        "order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date",
        "order_delivered_customer_date", "order_estimated_delivery_date"
    ])
    items = pd.read_csv(f"{DATA_DIR}/olist_order_items_dataset.csv", parse_dates=["shipping_limit_date"])
    customers = pd.read_csv(f"{DATA_DIR}/olist_customers_dataset.csv")
    sellers = pd.read_csv(f"{DATA_DIR}/olist_sellers_dataset.csv")
    products = pd.read_csv(f"{DATA_DIR}/olist_products_dataset.csv")
    category_translation = pd.read_csv(f"{DATA_DIR}/product_category_name_translation.csv")
    payments = pd.read_csv(f"{DATA_DIR}/olist_order_payments_dataset.csv")
    geo = pd.read_csv(f"{DATA_DIR}/olist_geolocation_dataset.csv")
    return orders, items, customers, sellers, products, category_translation, payments, geo


def build_geo_lookup(geo):
    """Reduz a geolocalização a 1 lat/lng médio por prefixo de CEP (existem duplicatas)."""
    geo_agg = (
        geo.groupby("geolocation_zip_code_prefix")
        .agg(lat=("geolocation_lat", "median"), lng=("geolocation_lng", "median"))
        .reset_index()
    )
    return geo_agg


def main():
    print("Carregando dados brutos...")
    orders, items, customers, sellers, products, cat_trans, payments, geo = load_data()

    print("Construindo lookup de geolocalização (por CEP)...")
    geo_agg = build_geo_lookup(geo)

    print("Unindo pedidos + itens...")
    df = items.merge(orders, on="order_id", how="left")

    print("Unindo clientes...")
    df = df.merge(customers, on="customer_id", how="left")

    print("Unindo vendedores...")
    df = df.merge(sellers, on="seller_id", how="left")

    print("Unindo produtos + tradução de categoria...")
    df = df.merge(products, on="product_id", how="left")
    df = df.merge(cat_trans, on="product_category_name", how="left")

    print("Agregando pagamentos por pedido (soma do valor pago)...")
    pay_agg = payments.groupby("order_id").agg(
        payment_value=("payment_value", "sum"),
        payment_installments=("payment_installments", "max"),
    ).reset_index()
    df = df.merge(pay_agg, on="order_id", how="left")

    print("Anexando coordenadas do cliente e do vendedor...")
    df = df.merge(
        geo_agg.rename(columns={"geolocation_zip_code_prefix": "customer_zip_code_prefix",
                                 "lat": "customer_lat", "lng": "customer_lng"}),
        on="customer_zip_code_prefix", how="left"
    )
    df = df.merge(
        geo_agg.rename(columns={"geolocation_zip_code_prefix": "seller_zip_code_prefix",
                                 "lat": "seller_lat", "lng": "seller_lng"}),
        on="seller_zip_code_prefix", how="left"
    )

    print("Calculando distância vendedor -> cliente (km)...")
    df["distance_km"] = haversine(df["seller_lat"], df["seller_lng"], df["customer_lat"], df["customer_lng"])

    print("Calculando métricas de tempo de entrega...")
    df["delivery_time_days"] = (df["order_delivered_customer_date"] - df["order_purchase_timestamp"]).dt.days
    df["estimated_time_days"] = (df["order_estimated_delivery_date"] - df["order_purchase_timestamp"]).dt.days
    df["delay_days"] = (df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]).dt.days
    df["is_late"] = df["delay_days"] > 0
    df["carrier_handoff_days"] = (df["order_delivered_carrier_date"] - df["order_purchase_timestamp"]).dt.days
    df["carrier_to_customer_days"] = (df["order_delivered_customer_date"] - df["order_delivered_carrier_date"]).dt.days

    print("Filtrando apenas pedidos com status 'delivered' e dados geográficos válidos...")
    df_clean = df[
        (df["order_status"] == "delivered")
        & df["distance_km"].notna()
        & df["delivery_time_days"].notna()
        & (df["delivery_time_days"] >= 0)
    ].copy()

    print(f"\nRegistros brutos (item x pedido): {len(df):,}")
    print(f"Registros limpos (delivered + geo válida): {len(df_clean):,}")

    out_path = f"{OUT_DIR}/olist_consolidado.csv"
    df_clean.to_csv(out_path, index=False)
    print(f"\nDataset consolidado salvo em: {out_path}")
    print(f"Colunas finais: {list(df_clean.columns)}")

    return df_clean


if __name__ == "__main__":
    main()
