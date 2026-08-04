from data_loader import load_raw_stock_data, save_processed_data
from feature_engineer import calc_technical_features, generate_label
from model_train import train_factor_model

def main():
    # 1.加载原始数据
    raw_path = "data/raw/stock.csv"
    clean_path = "data/processed/stock_clean.csv"
    feature_path = "data/processed/stock_feature.csv"

    print("===== 1.读取原始数据 =====")
    df = load_raw_stock_data(raw_path)
    df_clean = df.dropna()
    save_processed_data(df_clean, clean_path)

    print("===== 2.特征工程 =====")
    df_feature = calc_technical_features(df_clean)
    df_feature = generate_label(df_feature, forward_period=5)
    save_processed_data(df_feature, feature_path)

    print("===== 3.训练因子预测模型 =====")
    model = train_factor_model(df_feature)
    print("项目全流程执行完毕！")

if __name__ == "__main__":
    main()
