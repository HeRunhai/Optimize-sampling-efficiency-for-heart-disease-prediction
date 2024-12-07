import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# 1. 加载数据集
data = pd.read_csv('Heart.csv')

# 2. 数据预处理：分离特征和目标变量，处理分类变量
X = data.drop(columns=['target'])
y = data['target']

# 二进制分类特征转换为虚拟变量；将分类变量转换为一系列0和1的列，以便于机器学习算法处理
X = pd.get_dummies(X, columns=['sex', 'cp', 'restecg', 'exang', 'slope', 'thal'], drop_first=True)

# 3. 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=88)

# 4. 标准化数据
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. 线性回归模型
linear_reg = LinearRegression()
linear_reg.fit(X_train_scaled, y_train)
y_pred_linear = linear_reg.predict(X_test_scaled)
mse_linear = mean_squared_error(y_test, y_pred_linear)
mae_linear = mean_absolute_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)
print(f"\nLinear Regression - Mean Squared Error (MSE): {mse_linear}")
print(f"\nLinear Regression - Mean Absolute Error (MAE): {mae_linear}")
print(f"Linear Regression - R-squared (R²): {r2_linear}")

# 6. 随机森林回归模型
random_forest = RandomForestRegressor(n_estimators=100, random_state=88)
random_forest.fit(X_train_scaled, y_train)
y_pred_rf = random_forest.predict(X_test_scaled)
mse_rf = mean_squared_error(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print(f"\nRandom Forest Regression - Mean Squared Error (MSE): {mse_rf}")
print(f"\nGaussian Process Regression - Mean Absolute Error (MAE): {mae_rf}")
print(f"Random Forest Regression - R-squared (R²): {r2_rf}")

# 7. 贝叶斯线性回归模型
bayesian_ridge = BayesianRidge()
bayesian_ridge.fit(X_train_scaled, y_train)
y_pred_bayesian, y_std = bayesian_ridge.predict(X_test_scaled, return_std=True)
mse_bayesian = mean_squared_error(y_test, y_pred_bayesian)
mae_bayesian = mean_absolute_error(y_test, y_pred_bayesian)
r2_bayesian = r2_score(y_test, y_pred_bayesian)
print(f"\nBayesian Ridge Regression - Mean Squared Error (MSE): {mse_bayesian}")
print(f"\nBayesian Ridge Regression - Mean Absolute Error (MAE): {mae_bayesian}")
print(f"Bayesian Ridge Regression - R-squared (R²): {r2_bayesian}")

# 8. 可视化贝叶斯回归的预测和不确定性
plt.figure(figsize=(10, 6))
plt.errorbar(range(len(y_pred_bayesian)), y_pred_bayesian, yerr=y_std, fmt='o', label='Prediction Interval')
plt.fill_between(range(len(y_pred_bayesian)), y_pred_bayesian - y_std, y_pred_bayesian + y_std, color='gray', alpha=0.2)
plt.title('Predicted Values with Uncertainty (Bayesian Ridge)')
plt.xlabel('Sample Index')
plt.ylabel('Predicted Value')
plt.legend()
plt.show()

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox


# 真实值与预测值的对比图
def plot_pred_vs_true(y_true, y_pred, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, c='blue', label='Predictions')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', lw=2, label='True Values')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    plt.legend()
    plt.show()


# 残差图
def plot_acf_with_confidence(residuals, title):
    plt.figure(figsize=(10, 6))
    plot_acf(residuals, lags=30, alpha=0.05)  # alpha=0.05表示95%的置信区间
    plt.title(f'ACF with Confidence Interval - {title}')
    plt.xlabel('Lags')
    plt.ylabel('ACF')
    plt.show()


"""
Box-Ljung Test:
Null Hypothesis (零假设)：残差是白噪声，即滞后期的自相关为零。
p 值：如果 p 值较大（例如 > 0.05），我们不能拒绝零假设，意味着残差是白噪声
"""


def box_ljung_test(residuals, title):
    lb_test = acorr_ljungbox(residuals, lags=[30], return_df=True)  # lags可以调整为更高
    print(f'\nBox-Ljung Test - {title}')
    print(lb_test)


# 线性回归残差白噪声检验
box_ljung_test(y_test - y_pred_linear, 'Linear Regression')

# 随机森林回归残差白噪声检验
box_ljung_test(y_test - y_pred_rf, 'Random Forest Regression')

# 贝叶斯回归残差白噪声检验
box_ljung_test(y_test - y_pred_bayesian, 'Bayesian Ridge Regression')

"""
改进模型的合理性：如果通过残差检验发现模型存在未解释的模式，提出使用更复杂的贝叶斯模型或非线性模型（如高斯过程回归）是合理的改进策略
"""
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# 使用常数核和RBF核（径向基函数）来初始化高斯过程
kernel = C(1.0, (1e-4, 1e1)) * RBF(1.0, (1e-4, 1e1))
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-2)

# 训练模型
gpr.fit(X_train_scaled, y_train)

# 预测
y_pred_gpr, y_std_gpr = gpr.predict(X_test_scaled, return_std=True)

# 评估高斯过程回归的性能
mse_gpr = mean_squared_error(y_test, y_pred_gpr)
mae_gpr = mean_absolute_error(y_test, y_pred_gpr)
r2_gpr = r2_score(y_test, y_pred_gpr)
print(f"\nGaussian Process Regression - Mean Squared Error (MSE): {mse_gpr}")
print(f"\nGaussian Process Regression - Mean Absolute Error (MAE): {mae_gpr}")
print(f"Gaussian Process Regression - R-squared (R²): {r2_gpr}")

# 绘制高斯过程回归的预测值与置信区间
plt.figure(figsize=(10, 6))
plt.errorbar(range(len(y_pred_gpr)), y_pred_gpr, yerr=y_std_gpr, fmt='o', label='Prediction Interval')
plt.fill_between(range(len(y_pred_gpr)), y_pred_gpr - y_std_gpr, y_pred_gpr + y_std_gpr, color='gray', alpha=0.2)
plt.title('Predicted Values with Uncertainty (Gaussian Process Regression)')
plt.xlabel('Sample Index')
plt.ylabel('Predicted Value')
plt.legend()
plt.show()

box_ljung_test(y_test - y_pred_gpr, 'Gaussian Process Regression')

from sklearn.metrics import (roc_curve, precision_recall_curve, average_precision_score,
                             confusion_matrix, roc_auc_score)
import seaborn as sns


def plot_evaluation(y_true, y_pred_proba):
    # 1. 预测概率分布
    class0_probs = y_pred_proba[y_true == 0]
    class1_probs = y_pred_proba[y_true == 1]

    plt.figure(figsize=(10, 6))
    plt.hist(class0_probs, bins=20, alpha=0.5, label='Class 0', color='skyblue', density=True)
    plt.hist(class1_probs, bins=20, alpha=0.5, label='Class 1', color='salmon', density=True)
    plt.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.title('Prediction Probability Distribution')
    plt.legend()
    plt.show()

    # 2. 混淆矩阵
    plt.figure(figsize=(10, 6))
    y_pred_binary = (y_pred_proba >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred_binary)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['True 0', 'True 1'])
    plt.title('Confusion Matrix')
    plt.show()


plot_evaluation(y_test, y_pred_gpr)

"""
创新点与模型改进
几个潜在的创新点可以突出：
贝叶斯方法的引入：贝叶斯回归提供了不确定性量化，这是普通机器学习模型（如随机森林回归）无法提供的。在医疗数据集（如心脏病预测）的研究中，能够给出预测结果的不确定性（如置信区间）有很强的实际应用价值。
集成或混合模型：将贝叶斯回归与其他模型（如随机森林）进行集成或混合，可以结合各模型的优势。这种集成策略如果能够提高模型的精度并提供不确定性量化，是非常有意义的创新方向
"""
