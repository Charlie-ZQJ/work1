import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

# 参数设置
R0 = 8.3  # Surf-GC距离 (kpc)
A = 20.41
a = 9.03
b = 13.99
R_pdf = 3.76
r_max = 20  # 最大采样范围 (kpc)

# 定义密度函数
def rho(r):
    denominator = R0 + R_pdf
    term1 = ((r + R_pdf) / denominator) ** a
    term2 = np.exp(-b * (r - R0) / denominator)
    return A * term1 * term2

# 寻找密度函数的最大值
result = minimize_scalar(lambda r: -rho(r), bounds=(0, r_max), method='bounded')
max_r = result.x
max_f = rho(max_r)
print(f"密度函数最大值: {max_f:.2f}，位于 r = {max_r:.2f} kpc")

# 拒绝采样生成样本
np.random.seed(42)  # 固定随机种子以便复现
samples = []
N = 130000
total_trials = 0

while len(samples) < N:
    x = np.random.uniform(0, r_max)
    u = np.random.uniform()
    if u <= rho(x) / max_f:
        samples.append(x)
    total_trials += 1

samples = np.array(samples)
acceptance_rate = N / total_trials
print(f"接受率: {acceptance_rate:.4f}")

# 计算理论曲线并归一化
r_values = np.linspace(0, r_max, 1000)
rho_values = rho(r_values)
integral = np.trapz(rho_values, r_values)
rho_normalized = rho_values / integral

# 绘制结果
plt.figure(figsize=(10, 6))
plt.hist(samples, bins=50, density=True, alpha=0.6, color='blue', label='样本直方图')
plt.plot(r_values, rho_normalized, 'r-', lw=2, label='归一化理论曲线')
plt.xlabel('距离 r (kpc)')
plt.ylabel('概率密度')
plt.title('脉冲星密度分布的拒绝采样结果')
plt.legend()
plt.grid(True)
plt.show()