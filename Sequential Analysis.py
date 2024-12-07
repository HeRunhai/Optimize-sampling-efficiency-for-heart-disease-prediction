import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. 加载数据
data = pd.read_csv('Heart.csv')
X = data.drop(columns=['target'])
y = data['target']
X = pd.get_dummies(X, columns=['sex', 'cp', 'restecg', 'exang', 'slope', 'thal'], drop_first=True)

# 2. 数据分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=88)

# 3. 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. 序贯分析：初始化高斯过程回归模型
kernel = C(1.0, (1e-4, 1e1)) * RBF(1.0, (1e-4, 1e1))
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-2)

# 5. 分批训练
batch_size = int(len(X_train_scaled) / 10)  # 将训练集分为10个批次
mse_list, mae_list, r2_list, batch_sizes = [], [], [], []

# 设置停止条件
convergence_threshold = 0.01
max_batches = 10
previous_mse = float('inf')

# 序贯分析
for i in range(1, max_batches + 1):
    batch_X_train = X_train_scaled[:i * batch_size]
    batch_y_train = y_train[:i * batch_size]

    # 训练模型
    gpr.fit(batch_X_train, batch_y_train)
    y_pred, y_std = gpr.predict(X_test_scaled, return_std=True)

    # 评估性能
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # 存储结果
    mse_list.append(mse)
    mae_list.append(mae)
    r2_list.append(r2)
    batch_sizes.append(len(batch_y_train))

    print(f"Batch {i}: Samples = {len(batch_y_train)}, MSE = {mse:.4f}, MAE = {mae:.4f}, R² = {r2:.4f}")

    # 检查是否达到停止条件
    if abs(previous_mse - mse) < convergence_threshold:
        print(f"Convergence reached with batch {i} (Sample Size: {len(batch_y_train)}).")
        break

    previous_mse = mse

# 6. 可视化误差和收敛趋势
plt.figure(figsize=(12, 6))
plt.plot(batch_sizes, mse_list, 'o-', label="Mean Squared Error (MSE)")
plt.plot(batch_sizes, mae_list, 's-', label="Mean Absolute Error (MAE)")
plt.plot(batch_sizes, r2_list, 'd-', label="R-squared (R²)")
plt.axhline(y=mse_list[-1], color='gray', linestyle='--', alpha=0.7, label="Final MSE Threshold")
plt.xlabel("Sample Size", fontsize=14)
plt.ylabel("Error Metrics", fontsize=14)
plt.title("Sequential Analysis with Gaussian Process Regression", fontsize=16)
plt.legend()
plt.show()
