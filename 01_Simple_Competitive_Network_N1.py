import numpy as np

# 设置学习速率
eta = 0.6  # 给定学习速率

# 初始权重和样本角度
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

# 执行一次训练迭代，按指定顺序调整权重
for x in x_vectors:
    W_vectors = WTA_update(W_vectors, x, eta)

# 输出调整后的权重
print("经过一次训练后的权重调整结果：")
print(W_vectors)
