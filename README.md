# Gradient-Boosting-Algorithm

#  AQI Prediction using Gradient Boosting

Predicting **Air Quality Index (AQI)** from pollutant concentration data using the **Gradient Boosting Regressor**. This project leverages ensemble learning to model complex relationships between pollutants and air quality.

---

## 📌 Overview

This project:
- Uses real-world air quality data (New York)
- Applies Gradient Boosting for AQI regression
- Evaluates model performance using key metrics
- Compares against baseline models

---

## 📁 Dataset

- **File**: `New_York_Air_Quality.csv`
- **Features**:
  - `CO`, `NO2`, `SO2`, `O3`, `PM2.5`, `PM10` — pollutant levels
  - `AQI` — target variable

---

### ✅ Gradient Boosting Regressor

Gradient Boosting builds an ensemble of decision trees **sequentially**, where each tree tries to **correct errors** made by the previous one.

```python
model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=0
)


📊 Model Performance
Metric	Value
R² Score	0.553
MSE	105.06
RMSE	10.25

📈 RMSE indicates the model's predictions are off by ~10 AQI units on average — a clear improvement over Linear Regression.

