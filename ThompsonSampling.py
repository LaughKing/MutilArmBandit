import numpy as np
import matplotlib.pyplot as plt
from plot_results import plot_results

class BernoulliBandit:
    def __init__(self, K) -> None:
        self.probs = np.random.uniform(size=K)  # 随机生成一组概率
        self.best_idx = np.argmax(self.probs)  # 找到最大概率的拉杆
        self.best_prob = self.probs[self.best_idx]  # 获得最大的获奖概率
        self.K = K

    def step(self, k):
        if np.random.rand() < self.probs[k]:
            return 1
        else:
            return 0
        
np.random.seed(1)
K = 10
bandit_10_arm = BernoulliBandit(K)
print(f"生成了一个{K}臂老虎机")
print(f"获奖概率最佳的拉杆是{bandit_10_arm.best_idx},概率是{bandit_10_arm.best_prob}")


class Solver:
    def __init__(self, bandit):
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.K)  # 每个拉杆次数的初始值为0
        self.action = []
        self.regrets = []
        self.regret = 0

    # 根据策略选择动作
    def run_one_step(self):
        raise NotImplementedError

    def update_regret(self, k):
        self.regret += self.bandit.best_prob - self.bandit.probs[k]
        self.regrets.append(self.regret)

    def run(self, num_step):
        for _ in range(num_step):
            k = self.run_one_step() 
            self.counts[k] += 1 
            self.update_regret(k)  # 根据动作获得奖励
            self.action.append(k)  # 更新累计懊悔和计数

class ThompsonSampling(Solver):
    """ 汤普森采样算法,继承Solver类 """
    def __init__(self, bandit):
        super(ThompsonSampling, self).__init__(bandit)
        self._a = np.ones(self.bandit.K)  # 列表,表示每根拉杆奖励为1的次数
        self._b = np.ones(self.bandit.K)  # 列表,表示每根拉杆奖励为0的次数

    def run_one_step(self):
        samples = np.random.beta(self._a, self._b)  # 按照Beta分布采样一组奖励样本
        k = np.argmax(samples)  # 选出采样奖励最大的拉杆
        r = self.bandit.step(k)

        self._a[k] += r  # 更新Beta分布的第一个参数
        self._b[k] += (1 - r)  # 更新Beta分布的第二个参数
        return k


np.random.seed(1)
thompson_sampling_solver = ThompsonSampling(bandit_10_arm)
thompson_sampling_solver.run(5000)
print('汤普森采样算法的累积懊悔为：', thompson_sampling_solver.regret)
plot_results([thompson_sampling_solver], ["ThompsonSampling"])