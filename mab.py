import numpy as np
import matplotlib.pyplot as plt 

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


class EpsilonGreedy(Solver):
    """epsilon贪心算法，继承solver类"""

    def __init__(self, bandit, epsilon=0.01, init_prob=1.0):
        super(EpsilonGreedy, self).__init__(bandit)
        self.epsilon = epsilon
        self.estimates = np.array([init_prob] * self.bandit.K)

    def run_one_step(self):
        if np.random.random() < self.epsilon:
            k = np.random.randint(0, self.bandit.K)  # 随机选择一根拉杆
        else:
            k = np.argmax(self.estimates)
        r = self.bandit.step(k)  # 得到本次动作的奖励
        self.estimates[k] += 1. * (r - self.estimates[k]) / (self.counts[k] + 1)
        return k
    

def plot_results(solvers, solver_names):
    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative Regret')
    plt.title('%d-armed bandit' % solvers[0].bandit.K)
    plt.legend()
    plt.show()

np.random.seed(1)
epsilon_greedy_solver = EpsilonGreedy(bandit_10_arm, epsilon=1)
epsilon_greedy_solver.run(5000)
print('epsilon-贪婪算法的累计懊悔为：', epsilon_greedy_solver.regret)
plot_results([epsilon_greedy_solver], ["EpsilonGreedy"])


class UCB(Solver):

    def __init__(self, bandit,coef,init_prob=1.0):
        super(UCB,self).__init__(bandit)
        self.esitimates = np.array([init_prob]*self.bandit.K)
        self.total_count = 0
        self.coef = coef ## 确定不确定性比重的系数

    def run_one_step(self):
        self.total_count += 1
        ucb = self.esitimates + self.coef * np.sqrt(np.log(self.total_count) / (2*(self.counts + 1)))
        k = np.argmax(ucb)
        r = self.bandit.step
        self.esitimates[k] += 1 / (self.counts[k] + 1) * (r-self.esitimates[k])
        return k

