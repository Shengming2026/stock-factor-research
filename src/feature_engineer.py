import pandas as pd
import numpy as np

def calc_technical_features(df):
    """
    基础技术因子计算
    """
    data = df.copy()
    # 收益率
    data['ret_1d'] = data['close'].pct_change()
    # 移动平均线
    data['ma5'] = data['close'].rolling(window=5).mean()
    data['ma20'] = data['close'].rolling(window=20).mean()
    # 波动率
    data['vol_20d'] = data['ret_1d'].rolling(window=20).std()
    # 价格动量
    data['momentum_20d'] = data['close'] / data['close'].shift(20) - 1
    return data

def generate_label(df, forward_period=5):
    """
    构造预测标签：未来N日收益率
    """
    data = df.copy()
    data['future_ret'] = data['close'].shift(-forward_period) / data['close'] - 1
    return data

if __name__ == "__main__":
    from data_loader import load_raw_stock_data, save_processed_data
    df = load_raw_stock_data("data/processed/stock_clean.csv")
    df_feature = calc_technical_features(df)
    df_feature = generate_label(df_feature, forward_period=5)
    save_processed_data(df_feature, "data/processed/stock_feature.csv")
    print("特征工程完成")
