# Data Dictionary

This document outlines the machine learning features used within the **EV Charging Station AI Agent** pipeline. It covers both the raw input parameters collected from the user and the dynamically engineered features created during inference.

## 1. Input Features
These are the base features collected interactively by the LangGraph agent before invoking the prediction tool.

| Feature | Type | Description |
| :--- | :--- | :--- |
| `country_code` | Categorical (String) | ISO Country Code (e.g., 'US', 'IN'). It is automatically encoded to an integer using `label_encoder.pkl`. |
| `latitude` | Numerical (Float) | The geographical latitude of the station location. |
| `longitude` | Numerical (Float) | The geographical longitude of the station location. |
| `ports` | Numerical (Integer) | Total number of charging interfaces (ports) available at the station. |

## 2. Engineered Features
To capture complex, non-linear relationships for the Random Forest and XGBoost ensemble, the following features are mathematically generated inside `app.py` after standardization.

| Engineered Feature | Calculation | Purpose |
| :--- | :--- | :--- |
| `latitude_x_longitude` | `latitude * longitude` | Cross-interaction term reflecting continuous coordinate spaces and regional grids. |
| `ports_x_latitude` | `ports * latitude` | Interaction linking infrastructure size to Northern/Southern spatial distribution. |
| `ports_x_longitude` | `ports * longitude` | Interaction linking infrastructure size to Eastern/Western spatial distribution. |
| `latitude_squared` | `latitude ^ 2` | Captures non-linear impacts of latitude extremes. |
| `longitude_squared` | `longitude ^ 2` | Captures non-linear impacts of longitude extremes. |
| `ports_squared` | `ports ^ 2` | Captures the compounding effect and statistical weight of massive charging hubs. |

## 3. Output / Target
**Prediction Target:** The model outputs a probability score representing the likelihood of a location supporting **Fast DC Charging** vs Level 2. The probability is averaged between Random Forest and XGBoost, then evaluated against `optimal_threshold.pkl`.
