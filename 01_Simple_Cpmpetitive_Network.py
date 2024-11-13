import numpy as np
import matplotlib.pyplot as plt

# 初始参数
eta_initial = 0.6  # 初始学习率
N = 100  # 总训练次数
angles_x = [185, 175, 160, 270, 250, 240, 30, 60]  # 样本x向量的角度（度数）


# 将角度转换为单位圆上的向量
def angle_to_vector(angle):
    radians = np.radians(angle)
    return np.array([np.cos(radians), np.sin(radians)])


# 样本向量初始化
x_vectors = np.array([angle_to_vector(angle) for angle in angles_x])


# 胜者为王（WTA）更新函数
def WTA_update(W, x, eta):
    distances = np.linalg.norm(W - x, axis=1)
    winner_idx = np.argmin(distances)
    W[winner_idx] += eta * (x - W[winner_idx])
    W[winner_idx] /= np.linalg.norm(W[winner_idx])
    return W


# 测试函数
def run_training(W_initial, eta_decay, eta_initial=0.6, N=100):
    results = []
    W = W_initial.copy()
    eta = eta_initial

    for n in range(N):
        for x in x_vectors:
            W = WTA_update(W, x, eta)
        results.append(W.copy())

        # 学习率衰减策略
        if eta_decay == "geometric":
            eta *= 0.75
        elif eta_decay == "linear":
            eta = (N - n) / N * 0.8

    return W


# 不同初始化位置的权重向量配置
initial_positions = {
    "near_samples": [45, 155, 300],  # 初始权重接近样本的角度
    "far_from_samples": [10, 120, 210]  # 初始权重远离样本的角度
}

# 不同学习速率衰减策略
decay_strategies = ["geometric", "linear"]

# 结果存储
final_results = {}

for init_name, angles_W in initial_positions.items():
    W_initial = np.array([angle_to_vector(angle) for angle in angles_W])
    final_results[init_name] = {}
    for decay in decay_strategies:
        W_final = run_training(W_initial, decay)
        final_results[init_name][decay] = W_final

# 输出最终权重
for init_name, decay_results in final_results.items():
    print(f"\n初始位置: {init_name}")
    for decay, W_final in decay_results.items():
        print(f"学习速率衰减: {decay}")
        print(W_final)
