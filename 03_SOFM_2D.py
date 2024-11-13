import numpy as np
import matplotlib.pyplot as plt
import random
from math import sqrt

plt.rcParams['font.sans-serif'] = ['Songti SC']  # 使用黑体字体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
# 生成均匀分布在 (0, 1) x (0, 1) 区域内的数据点
def generate_data_square(num):
    return np.random.rand(num, 2)


# 可视化函数：显示数据点和神经元分布
def show_data(data, neurons, title=''):
    plt.scatter(data[:, 0], data[:, 1], s=10, c='blue', label='Data Points')
    plt.scatter(neurons[:, 0], neurons[:, 1], s=35, c='red', label='Neurons')
    plt.plot(neurons[:, 0], neurons[:, 1], 'y-', linewidth=1, label='Neuron Connections')
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# WTA算法：返回最接近的神经元ID
def WTA2(x, w):
    dist = np.array([(x - ww).dot(x - ww) for ww in w])
    return np.argmin(dist)


# 邻域竞争函数
def neighborid2d(id, grid_shape, radius):
    neighbors = []
    x_id, y_id = id // grid_shape[1], id % grid_shape[1]
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            if abs(i) + abs(j) <= radius:
                nx, ny = x_id + i, y_id + j
                if 0 <= nx < grid_shape[0] and 0 <= ny < grid_shape[1]:
                    neighbors.append(nx * grid_shape[1] + ny)
    return neighbors


def compete2d(x, w, eta, grid_shape, radius):
    for xx in x:
        id = WTA2(xx, w)
        iddim = neighborid2d(id, grid_shape, radius)
        for iidd in iddim:
            w[iidd] = w[iidd] + eta * (xx - w[iidd])
    return w


# 初始化方法
def initialize_neurons(num_neurons, method, data_shape=(1, 1)):
    if method == 'random':
        neurons = np.random.rand(num_neurons, 2) * data_shape
    elif method == 'grid':
        side = int(sqrt(num_neurons))
        neurons = np.array([[i / (side - 1), j / (side - 1)] for i in range(side) for j in range(side)]) * data_shape
    elif method == 'boundary':
        side = int(sqrt(num_neurons))
        neurons = []
        for i in range(side):
            neurons.append([i / (side - 1), 0])  # 左边界
        for j in range(1, side):
            neurons.append([1, j / (side - 1)])  # 上边界
        for i in range(side - 2, -1, -1):
            neurons.append([i / (side - 1), 1])  # 右边界
        for j in range(side - 2, 0, -1):
            neurons.append([0, j / (side - 1)])  # 下边界
        neurons = np.array(neurons) * data_shape
    return neurons


# 主函数：初始化参数并进行训练
def train_sofm(num_data=100, num_neurons=100, train_steps=100, eta_begin=0.3, eta_end=0.01, radius_begin=5,
               radius_end=0, init_method='random'):
    # 生成数据
    data = generate_data_square(num_data)

    # 初始化神经元
    grid_shape = (int(sqrt(num_neurons)), int(sqrt(num_neurons)))
    neurons = initialize_neurons(num_neurons, init_method)

    # 训练过程
    for i in range(train_steps):
        eta = (eta_begin - eta_end) * (train_steps - i) / (train_steps - 1) + eta_end
        radius = int((radius_begin - radius_end) * (train_steps - i) / (train_steps - 1) + radius_end)

        neurons = compete2d(data, neurons, eta, grid_shape, radius)

        if (i + 1) % (train_steps // 10) == 0:
            show_data(data, neurons,
                      title=f"Step: {i + 1}/{train_steps}, Radius: {radius}, Eta: {eta:.2f} ({init_method} init)")

    # 显示最终结果
    show_data(data, neurons, title=f"Final Neuron Distribution ({init_method} init)")


# 调用主函数
for method in ['random', 'grid', 'boundary']:
    train_sofm(num_data=100, num_neurons=100, train_steps=100, eta_begin=0.3, eta_end=0.01, radius_begin=5,
               radius_end=0, init_method=method)
