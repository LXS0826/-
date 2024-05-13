# encoding:utf-8
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt

''' 
导入数据
'''
data = pd.read_excel(r"C:\Users\lixinshuai\Desktop\output.xlsx")

# 设定保存路径，以列名为目录名
v_names = data.columns
j = 2
column_name = v_names[j]

# 选择 B 到 S 列作为输入
X = data.loc[:, '高度/M':'SAVI1'].values
# 选择 T 列作为输出
y = data[('CC')].values

# 分割数据集
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

''' 
建模
'''
# 定义一个交叉评估函数 Validation function
n_folds = 5
def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(x_train)
    rmse= np.sqrt(-cross_val_score(model, x_train, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

# 岭回归（Kernel Ridge Regression）
KRR = make_pipeline(RobustScaler(), KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5))
score = rmsle_cv(KRR)
print("\nKernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


''' 
预测
'''
# 将 y_train 和 x_train 重新赋值为原来的变量名
y_train = data[('CC')].values
x_train = data.loc[:, '高度/M':'SAVI1'].values
KRR.fit(x_train, y_train)
y_pred = KRR.predict(x_test)
# 绘制实际值和预测值的对比图
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.xlabel('Sample')
plt.ylabel('CC')
plt.title('Actual vs Predicted')
plt.legend()
plt.show()
