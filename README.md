# 📦 Otimização Logística de Marketplace

Projeto de portfólio de ponta a ponta: dados brutos → EDA → modelos de otimização → dashboard interativo.

**Dataset:** [Olist Brazilian E-Commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) — 99.441 pedidos reais e anonimizados de um marketplace brasileiro (2016-2018), incluindo geolocalização de clientes e vendedores.

## 🎯 Problema de negócio

O Olist opera como marketplace: ~60% dos vendedores estão concentrados em São Paulo, mas os clientes estão espalhados por todo o Brasil. Isso gera distâncias longas de entrega, fretes altos e atrasos concentrados nas regiões Norte/Nordeste (até 3x acima da média nacional). Este projeto simula uma rede logística com múltiplos centros de distribuição regionais para reduzir distância, custo de frete e atraso.

## 🗂️ Estrutura do projeto

```
marketplace_logistics/
├── data/                          # CSVs brutos da Olist
├── src/
│   ├── 01_etl.py                  # Consolidação das 8 tabelas + cálculo de distância (Haversine)
│   ├── 02_eda.py                  # Análise exploratória: gargalos logísticos
│   ├── 03_demand_forecast.py      # Previsão de demanda (RandomForest)
│   ├── 04_dc_optimization.py      # Otimização de centros de distribuição (K-Means ponderado)
│   ├── 05_routing_optimization.py # Roteirização / VRP (OR-Tools)
│   └── 06_dashboard.py            # Geração do dashboard HTML (Plotly)
├── outputs/                        # Dataset consolidado, gráficos (.png) e dashboard.html
├── notebooks/
│   └── portfolio_notebook.ipynb   # Notebook único com toda a narrativa do projeto
├── streamlit_app.py               # Dashboard interativo (rodar localmente)
└── requirements.txt
```

## 🚀 Como rodar

```bash
pip install -r requirements.txt

# Pipeline completo (na ordem)
cd src
python 01_etl.py
python 02_eda.py
python 03_demand_forecast.py
python 04_dc_optimization.py
python 05_routing_optimization.py
python 06_dashboard.py

# Dashboard interativo (opcional)
cd ..
streamlit run streamlit_app.py
```

O `outputs/dashboard.html` pode ser aberto diretamente no navegador, sem precisar rodar nada (standalone).

## 📊 Principais resultados

| Métrica | Cenário Atual | Cenário Otimizado (5 DCs) | Ganho |
|---|---|---|---|
| Distância média cliente–origem | 596 km | 236 km | **-60%** |
| Frete médio estimado | R$ 19,94 | R$ 16,35 | **-18%** |
| Distância percorrida em rota (simulação VRP) | 128,8 km | 80,2 km | **-38%** |
| Previsão de demanda (validação do modelo) | — | MAPE 3,3% | alta precisão |

**Principais insights:**
- 59,6% dos vendedores estão em SP, mas apenas 41,5% dos clientes estão lá — desequilíbrio estrutural que causa entregas longas para o resto do país.
- Estados como Alagoas (20,9%), Maranhão (17,8%) e Sergipe (16,3%) têm os maiores índices de atraso — bem acima da média nacional de 6,6%.
- Do tempo total de entrega, 8,7 dias (73%) são gastos no transporte físico, contra 2,8 dias de processamento interno — mostrando que o ganho está mais em rede logística do que em processos internos.
- Simulação de 5 centros de distribuição regionais (via K-Means ponderado por volume de pedidos) reduziria a distância média percorrida em 60%.

## 🛠️ Stack técnica

- **Dados:** Pandas, NumPy
- **Geolocalização:** Haversine (distância real entre coordenadas)
- **Machine Learning:** Scikit-learn (RandomForest para previsão de demanda, regressão linear para custo de frete)
- **Otimização:** K-Means ponderado (alocação de centros de distribuição), Google OR-Tools (roteirização / VRP)
- **Visualização:** Matplotlib, Plotly, Streamlit

## 📌 Próximos passos (extensões possíveis)

- Incluir custo fixo de abertura de cada DC na função objetivo (hoje otimiza só distância)
- VRP com janelas de tempo (time windows) e múltiplos depósitos simultâneos
- Modelo de previsão de demanda com sazonalidade mais rica (ex.: Black Friday, feriados regionais)
- Deploy do dashboard Streamlit (Streamlit Community Cloud) com link público no portfólio
