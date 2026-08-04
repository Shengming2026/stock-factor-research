import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def train_factor_model(df):
    feature_cols = ["ret_1d", "ma5", "ma20", "vol_20d", "momentum_20d"]
    data = df.dropna(subset=feature_cols + ["future_ret"])
    X = data[feature_cols]
    y = data["future_ret"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"测试集MSE: {mse:.6f}")
    return model

if __name__ == "__main__":
    from data_loader import load_raw_stock_data
    df = load_raw_stock_data("data/processed/stock_feature.csv")
    rf_model = train_factor_model(df)
    print("模型训练结束")
