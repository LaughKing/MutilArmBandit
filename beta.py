import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# 定义不同参数的 Beta 分布
alpha_beta_pairs = [(1, 1), (5, 2), (2, 5), (10, 10)]

x = np.linspace(0, 1, 100)  # 定义 x 轴范围

# 绘制不同参数的 Beta 分布
plt.figure(figsize=(10, 6))
for (alpha, beta) in alpha_beta_pairs:
    y = stats.beta.pdf(x, alpha, beta)  # 计算 Beta 分布的概率密度函数
    plt.plot(x, y, label=f'Beta({alpha},{beta})')

plt.title('Beta Distributions for Different α and β')
plt.xlabel('Probability')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()