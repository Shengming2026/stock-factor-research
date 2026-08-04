import pandas as pd
import numpy as np

def calculate_ic(df, factor_col, return_col="future_ret"):
    """
    计算月度IC序列
    :param df: 包含因子与未来收益的数据
    :param factor_col: 因子列名称
    :param return_col: 预测收益标签
    :return: ic均值, ic标准差, icir
    """
    df_valid = df.dropna(subset=[factor_col, return_col])
    # 皮尔逊相关系数作为IC
    ic_series = df_valid.groupby("date")[[factor_col, return_col]].corr().iloc[0::2, -1]
    ic_series = ic_series.reset_index()[return_col]

    ic_mean = ic_series.mean()
    ic_std = ic_series.std()
    icir = ic_mean / ic_std if ic_std != 0 else 0
    return ic_mean, ic_std, icir, ic_series

def batch_ic_test(df, factor_list):
    """批量对多个因子做IC检验"""
    result = []
    for fac in factor_list:
        mean_ic, std_ic, icir, _ = calculate_ic(df, fac)
        result.append({
            "factor": fac,
            "mean_ic": round(mean_ic, 4),
            "ic_std": round(std_ic, 4),
            "icir": round(icir, 4)
        })
    res_df = pd.DataFrame(result)
    return res_df

if __name__ == "__main__":
    from data_loader import load_raw_stock_data
    df = load_raw_stock_data("data/processed/stock_feature.csv")
    factor_names = ["ret_1d", "ma5", "ma20", "vol_20d", "momentum_20d"]
    ic_result = batch_ic_test(df, factor_names)
    print("因子IC检验结果：")
    print(ic_result)
