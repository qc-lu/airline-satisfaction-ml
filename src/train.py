import numpy as np
import pandas as pd
import joblib
from sklearn.svm import SVC

# 读取数据
df = pd.read_csv("data/train.csv", encoding="gb2312")
test = pd.read_csv("data/test.csv", encoding="gb2312")

# 删除无用列
df.drop(['Unnamed: 0', 'id'], axis=1, inplace=True)
test.drop(['Unnamed: 0', 'id'], axis=1, inplace=True)

# 删除缺失值
df = df.dropna()
test = test.dropna()

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

df = encode(df)
test = encode(test)

print("训练集满意度原始唯一值映射后：", df['满意度'].unique())
print("测试集满意度原始唯一值映射后：", test['满意度'].unique())
print("训练集满意度 dtype:", df['满意度'].dtype)
print("测试集满意度 dtype:", test['满意度'].dtype)
print("训练集满意度缺失数:", df['满意度'].isna().sum())
print("测试集满意度缺失数:", test['满意度'].isna().sum())

df = df.dropna()
test = test.dropna()

# 拆特征和标签
X_train = df.drop('满意度', axis=1)
y_train = df['满意度'].astype(int).to_numpy()

X_test = test.drop('满意度', axis=1)
y_test = test['满意度'].astype(int).to_numpy()

# 标准化
train_mean = X_train.mean()
train_std = X_train.std()

X_train = (X_train - train_mean) / train_std

joblib.dump(train_mean, 'models/train_mean.pkl')
joblib.dump(train_std, 'models/train_std.pkl')

# 特征选择
if '门位置' in X_train.columns:
    X_train = X_train.drop(['门位置'], axis=1)
    X_test = X_test.drop(['门位置'], axis=1)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

# 训练模型
svc_rbf = SVC(kernel='rbf')
svc_linear = SVC(kernel='linear')
svc_poly = SVC(kernel='poly')

svc_rbf.fit(X_train, y_train)
svc_linear.fit(X_train, y_train)
svc_poly.fit(X_train, y_train)

# 保存模型
joblib.dump(svc_rbf, 'models/svc_rbf.pkl')
joblib.dump(svc_linear, 'models/svc_linear.pkl')
joblib.dump(svc_poly, 'models/svc_poly.pkl')

print("模型训练完成并已保存！")