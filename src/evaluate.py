import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix

# ===== 1. 读取数据 =====
test = pd.read_csv("data/test.csv", encoding="gb2312")

# 删除无用列
test.drop(['Unnamed: 0', 'id'], axis=1, inplace=True)

# 删除缺失值
test = test.dropna()

# ===== 2. 编码函数（必须和 train 一致）=====
def encode(df):
    df = df.copy()

    for col in ['性别', '顾客类型', '旅行类型', '舱位', '满意度']:
        df[col] = df[col].astype(str).str.strip()

    df['性别'] = df['性别'].map({'男': 1, '女': 0})
    df['顾客类型'] = df['顾客类型'].map({'忠诚顾客': 1, '非忠诚顾客': 0})
    df['旅行类型'] = df['旅行类型'].map({'商务旅行': 1, '个人旅行': 0})
    df['舱位'] = df['舱位'].map({'商务舱': 1, '经济舱': 0, '经济舱 Plus': 0})
    df['满意度'] = df['满意度'].map({'满意': 1, '中立或不满意': 0})

    return df

test = encode(test)
test = test.dropna()

# ===== 3. 特征 & 标签 =====
X_test = test.drop('满意度', axis=1)
y_test = test['满意度'].astype(int).to_numpy()

# ===== 4. 加载训练集标准化参数 =====
train_mean = joblib.load("models/train_mean.pkl")
train_std = joblib.load("models/train_std.pkl")

X_test = (X_test - train_mean) / train_std

# 特征选择（和 train 一致）
if '门位置' in X_test.columns:
    X_test = X_test.drop(['门位置'], axis=1)

# ===== 5. 加载模型 =====
models = {
    "linear": joblib.load("models/svc_linear.pkl"),
    "rbf": joblib.load("models/svc_rbf.pkl"),
    "poly": joblib.load("models/svc_poly.pkl"),
}

# ===== 6. 评估 =====
for name, model in models.items():
    print(f"\n===== {name.upper()} MODEL =====")

    y_pred = model.predict(X_test)

    # 误差
    error = np.mean(y_test != y_pred)
    print("测试误差:", error)

    # 混淆矩阵（标准格式）
    conf = confusion_matrix(y_test, y_pred, labels=[0, 1])

    conf_df = pd.DataFrame(
        conf,
        index=['实际不满意(0)', '实际满意(1)'],
        columns=['预测不满意(0)', '预测满意(1)']
    )

    print("混淆矩阵:")
    print(conf_df)