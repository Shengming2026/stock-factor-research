import pandas as pd
import os

def load_raw_stock_data(file_path):
    """
    读取原始行情数据
    :param file_path: 原始csv文件路径
    :return: DataFrame
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在：{file_path}")
    df = pd.read_csv(file_path)
    return df

def save_processed_data(df, save_path):
    """
    保存清洗后的数据到processed文件夹
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    raw_path = "data/raw/stock.csv"
    out_path = "data/processed/stock_clean.csv"
    data = load_raw_stock_data(raw_path)
    print("原始数据形状：", data.shape)
    # 简单去空值示例
    data_clean = data.dropna()
    save_processed_data(data_clean, out_path)
    print("数据清洗完成并保存")
