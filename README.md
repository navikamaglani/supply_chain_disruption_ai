# AI-Powered Supply Chain Disruption Intelligence System

## Overview

The AI-Powered Supply Chain Disruption Intelligence System is an operational intelligence platform designed to predict and monitor global supply chain disruption risks using machine learning, weather intelligence, and international trade analytics.

The system combines historical trade data, real-time weather alerts, country-level dependency mapping, and predictive analytics to identify potential disruptions before significant downstream impacts occur.

By integrating external risk signals with trade flow analysis, the platform provides actionable insights for logistics planners, supply chain analysts, and decision-makers.

---

## Problem Statement

Global supply chains are increasingly vulnerable to disruptions caused by:

- Extreme weather events
- Trade route bottlenecks
- Port dependency risks
- Geopolitical instability
- Sudden shifts in trade activity

Traditional monitoring systems often react after disruptions occur. This project aims to provide an early-warning intelligence system that proactively identifies elevated disruption risk using data-driven models.

---

## Key Features

### Global Trade Analytics

- Analyze international trade flows across countries and regions
- Identify import/export trends
- Monitor trade volume anomalies
- Detect significant deviations from historical baselines

### Weather Intelligence Integration

- NOAA weather alert integration
- Real-time severe weather monitoring
- Weather severity scoring
- Trade-weather risk correlation analysis

### Supply Chain Dependency Modeling

- Country-to-port dependency mapping
- Trade route exposure assessment
- Dependency risk scoring
- Identification of high-risk logistics corridors

### Machine Learning Risk Prediction

- Logistic Regression baseline model
- Random Forest classification model
- Supply chain disruption risk prediction
- Probabilistic risk scoring

### Explainable AI

- SHAP-based feature importance analysis
- Model transparency
- Risk factor attribution
- Operational decision support

### Interactive Dashboard

- Real-time disruption monitoring
- Country-level risk visualization
- Trend analysis dashboards
- Scenario simulation tools

---

## System Architecture

```text
                NOAA Weather Alerts
                         |
                         v
+------------------------------------------------+
|          Data Collection Layer                 |
|------------------------------------------------|
| NOAA API | U.S. Census Trade API | Metadata    |
+------------------------------------------------+
                         |
                         v
+------------------------------------------------+
|          Data Processing Layer                 |
|------------------------------------------------|
| Cleaning | Aggregation | Feature Engineering   |
+------------------------------------------------+
                         |
                         v
+------------------------------------------------+
|        Machine Learning Pipeline               |
|------------------------------------------------|
| Logistic Regression | Random Forest            |
| Class Balancing     | Model Evaluation         |
+------------------------------------------------+
                         |
                         v
+------------------------------------------------+
|            Explainability Layer                |
|------------------------------------------------|
| SHAP Analysis | Feature Attribution            |
+------------------------------------------------+
                         |
                         v
+------------------------------------------------+
|             Streamlit Dashboard                |
|------------------------------------------------|
| Risk Scores | Trends | Alerts | Simulations    |
+------------------------------------------------+
```

---

## Dataset

### U.S. Census International Trade Data

**Source:** U.S. Census Bureau International Trade APIs

**Data Includes:**
- Trade value
- Country-level trade statistics
- Import and export metrics
- Historical trade activity

### NOAA Weather Alerts

**Source:** National Oceanic and Atmospheric Administration (NOAA)

**Data Includes:**
- Severe weather warnings
- Storm events
- Hazard classifications
- Geographic impact areas

---

## Feature Engineering

The following features were engineered to improve disruption prediction:

### Trade Activity Features

- Trade volume
- Rolling averages
- Moving trends
- Trade growth rates
- Trade volatility

### Weather Features

- Alert frequency
- Weather severity scores
- Hazard category encoding
- Weather impact indicators

### Dependency Features

- Country dependency score
- Port exposure score
- Supply chain concentration index
- Trade route risk metrics

---

## Machine Learning Models

### Logistic Regression

Used as a baseline classification model.

**Advantages**
- Interpretable
- Fast training
- Strong baseline performance

### Random Forest

Primary predictive model.

**Advantages**
- Handles nonlinear relationships
- Robust to noise
- Effective with mixed feature types

---

## Model Performance

| Metric | Random Forest |
|----------|----------|
| Accuracy | 81% |
| ROC-AUC | 0.88 |
| Precision | High |
| Recall | High |

The Random Forest model demonstrated superior performance in identifying disruption risks compared to baseline models.

---

## Explainability with SHAP

To improve transparency and trust in model predictions, SHAP (SHapley Additive exPlanations) was used to:

- Identify key disruption drivers
- Explain individual predictions
- Visualize feature importance
- Support operational decision-making

### Example Insights

- Severe weather alerts significantly increased disruption risk.
- High dependency trade routes showed elevated vulnerability.
- Sudden trade volume anomalies were strong disruption indicators.

---

## Dashboard Capabilities

The Streamlit dashboard provides:

- Global disruption risk monitoring
- Country-level risk assessment
- Trade trend visualization
- Weather impact analysis
- Model explainability views
- Interactive filtering and exploration

---

## Technology Stack

| Category | Technologies |
|-----------|-------------|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-Learn |
| Explainability | SHAP |
| Visualization | Plotly, Matplotlib |
| Dashboard | Streamlit |
| APIs | NOAA API, U.S. Census API |

---

## Project Structure

```text
supply-chain-disruption-intelligence/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── exploration.ipynb
│   └── modeling.ipynb
│
├── src/
│   ├── data_ingestion.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── explainability.py
│
├── app/
│   └── streamlit_app.py
│
├── models/
│   └── random_forest.pkl
│
├── requirements.txt
│
└── README.md
```

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/supply-chain-disruption-intelligence.git
cd supply-chain-disruption-intelligence
```

### Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Application

```bash
streamlit run app/streamlit_app.py
```

The application will launch in your browser and provide real-time supply chain risk monitoring and analytics.

---

## Future Enhancements

- Real-time streaming data pipelines
- Port congestion intelligence
- Geopolitical risk integration
- LLM-powered disruption summaries
- Multi-horizon forecasting
- AWS cloud deployment

---

## Business Impact

This platform demonstrates how AI and external intelligence sources can be combined to improve supply chain resilience by:

- Detecting disruption signals earlier
- Improving operational visibility
- Supporting proactive mitigation strategies
- Enhancing risk-aware decision making

---

## Author

**Navika Maglani**

MS Data Science | Machine Learning | AI Engineering | Supply Chain Analytics
