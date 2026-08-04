stock-factor-research
A股股票多因子分析框架，实现行情数据获取、基本面&量价因子构造、IC有效性检验、分层回测与简易选股策略。
data
├── raw         # 原始行情数据
└── processed   # 清洗、加工后的特征数据
notebooks       # 探索性分析jupyter
output          # 模型输出、预测结果
src             # 核心代码
├── data_loader.py      # 数据读写
├── feature_engineer.py # 因子计算与标签构建
├── model_train.py      # 机器学习模型训练
└── main.py             # 项目入口
安装依赖
```bash
pip install -r requirements.txt
python src/main.py
