import numpy as np
import matplotlib.pyplot as plt

# 设置初始参数
eta_initial = 0.6  # 初始学习率
N = 100  # 总训练次数
angles_W = [45, 155, 300]  # W向量的初始角度（度数）
angles_x = [185, 175, 160, 270, 250, 240, 30, 60]  # 样本x向量的角度（度数）

# 将角度转换为单位圆上的向量
def angle_to_vector(angle):
    radians = np.radians(angle)
    return np.array([np.cos(radians), np.sin(radians)])

# 初始化权重和输入向量为单位圆上的向量
W_vectors = np.array([angle_to_vector(angle) for angle in angles_W])
x_vectors = np.array([angle_to_vector(angle) for angle in angles_x])

# 胜者为王（WTA）更新函数
def WTA_update(W, x, eta):
    # 找到与x最接近的胜者神经元索引
    distances = np.linalg.norm(W - x, axis=1)
    winner_idx = np.argmin(distances)
    # 更新胜者的权重向量
    W[winner_idx] += eta * (x - W[winner_idx])
    # 归一化更新后的胜者向量，使其保持在单位圆上
    W[winner_idx] /= np.linalg.norm(W[winner_idx])
    return W

# 绘制网络的函数
def shownet(s, w, title):
    plt.clf()
    plt.scatter(s[:,0], s[:,1], c='b', s=20.0, alpha=1)
    plt.scatter(w[:,0], w[:,1], c='r', s=30.0, alpha=1)
    a = np.linspace(0, 2 * np.pi, 100)
    plt.plot(np.cos(a), np.sin(a), 'g--', linewidth=1)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True)
    plt.title(title)
    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)

# 运行训练过程（等比下降和线性下降的学习率）
results_geometric = []
results_linear = []

# 等比下降
eta = eta_initial
W_geometric = W_vectors.copy()
for n in range(N):
    for x in x_vectors:
        W_geometric = WTA_update(W_geometric, x, eta)
    results_geometric.append(W_geometric.copy())
    shownet(x_vectors, W_geometric, f"Step: {n+1}, η={eta:.4f}")
    eta *= 0.75  # 学习率按等比下降

# 线性下降
W_linear = W_vectors.copy()
for n in range(N):
    eta = (N - n) / N * 0.8  # 学习率按线性下降
    for x in x_vectors:
        W_linear = WTA_update(W_linear, x, eta)
    results_linear.append(W_linear.copy())
    shownet(x_vectors, W_linear, f"Step: {n+1}, η={eta:.4f}")

# 最终结果输出
print("等比下降最终权重：", results_geometric[-1])
print("线性下降最终权重：", results_linear[-1])

plt.show()
