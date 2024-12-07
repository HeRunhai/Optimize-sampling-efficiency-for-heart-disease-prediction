import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# 1. 加载数据集
data = pd.read_csv('Heart.csv')

# 2. 数据预处理：分离特征和目标变量，处理分类变量
X = data.drop(columns=['target'])
y = data['target']

# 二进制分类特征转换为虚拟变量
X = pd.get_dummies(X, columns=['sex', 'cp', 'restecg', 'exang', 'slope', 'thal'], drop_first=True)

# 3. 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=88)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 定义模型
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=88),
    "Bayesian Ridge": BayesianRidge(),
    "Gaussian Process": GaussianProcessRegressor(
        kernel=C(1.0, (1e-4, 1e1)) * RBF(1.0, (1e-4, 1e1)),
        n_restarts_optimizer=10, alpha=1e-2
    )
}

# 存储结果
results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    results[name] = {
        "y_pred": y_pred,
        "mse": mean_squared_error(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
    }

# 可视化各模型的预测结果与实际值的比较
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()
for i, (name, result) in enumerate(results.items()):
    axes[i].scatter(y_test, result["y_pred"], c='darkblue', label='Predictions', alpha=0.6)
    axes[i].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2, label='True Values')
    axes[i].set_xlabel('True Values', color='black', fontsize=12)
    axes[i].set_ylabel('Predicted Values', color='black', fontsize=12)
    axes[i].set_title(f'{name} Predicted vs True', color='black', fontsize=14)
    axes[i].legend()
plt.tight_layout()
plt.savefig('Model_Predicted_vs_True.jpg', format='jpg')
plt.show()

# 误差指标比较 - 使用柱状图展示 MSE, MAE, R²
metrics = ['mse', 'mae', 'r2']
metric_values = {metric: [results[model][metric] for model in models] for metric in metrics}

plt.figure(figsize=(10, 6))
x = np.arange(len(models))
width = 0.25
colors = ['#2F4F4F', '#556B2F', '#8B0000']  # 深灰绿，深橄榄绿，深红

for i, metric in enumerate(metrics):
    plt.bar(x + i * width, metric_values[metric], width, label=metric.upper(), color=colors[i])
plt.xticks(x + width, models.keys(), color='black', fontsize=12)
plt.ylabel('Metric Score', color='black', fontsize=12)
plt.title('Model Performance Comparison', color='black', fontsize=14)
plt.legend()
plt.savefig('Model_Performance_Comparison.jpg', format='jpg')
plt.show()

# 展示贝叶斯岭回归和高斯过程模型的预测不确定性
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
for i, name in enumerate(["Bayesian Ridge", "Gaussian Process"]):
    y_pred = results[name]["y_pred"]
    y_std = None
    if name == "Bayesian Ridge":
        _, y_std = models[name].predict(X_test_scaled, return_std=True)
    elif name == "Gaussian Process":
        _, y_std = models[name].predict(X_test_scaled, return_std=True)

    if y_std is not None:
        ax[i].errorbar(range(len(y_pred)), y_pred, yerr=y_std, fmt='o', label='Prediction Interval', alpha=0.5)
        ax[i].fill_between(range(len(y_pred)), y_pred - y_std, y_pred + y_std, color='gray', alpha=0.2)
    ax[i].scatter(range(len(y_test)), y_test, color='red', s=10, label='True Values')
    ax[i].set_title(f'{name} - Predicted Values with Uncertainty', color='black', fontsize=14)
    ax[i].legend()
plt.savefig('Prediction_Uncertainty.jpg', format='jpg')
plt.show()

# 打印模型性能总结
print("\nModel Performance Summary:")
for name, result in results.items():
    print(f"{name}:")
    print(f"  - MSE: {result['mse']:.4f}")
    print(f"  - MAE: {result['mae']:.4f}")
    print(f"  - R²: {result['r2']:.4f}\n")
