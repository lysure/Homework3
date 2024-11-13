import numpy as np
import matplotlib.pyplot as plt
import random
from math import sqrt


# 数据生成函数：生成三角形区域内的随机点
def generate_data_triangle(num):
    pointdim = []
    for _ in range(num):
        while True:
            x = random.uniform(0, 1)
            y = random.uniform(0, sqrt(3) / 2)
            y_limit = sqrt(3) * (0.5 - abs(x - 0.5))
            if y <= y_limit:
                pointdim.append([x, y])
                break
    return np.array(pointdim)


# 可视化函数：显示数据点和神经元分布
def show_data(data, neurons, lineflag=0, title=''):
    plt.scatter(data[:, 0], data[:, 1], s=10, c='blue', label='Data Points')
    if lineflag == 1:
        plt.scatter(neurons[:, 0], neurons[:, 1], s=35, c='red', label='Neurons')
        plt.plot(neurons[:, 0], neurons[:, 1], 'y-', linewidth=1, label='Neuron Connections')

    # 绘制三角形边界
    board_x = np.linspace(0, 1, 200)
    board_y = [sqrt(3) * (0.5 - abs(x - 0.5)) for x in board_x]
    plt.plot(board_x, board_y, 'c--', linewidth=1)
    plt.plot(board_x, np.zeros(len(board_x)), 'c--', linewidth=1)

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.axis([-0.05, 1.05, -0.05, 0.9])
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
def neighborid1(id, row, r):
    if r <= 0:
        return [id]
    return [i for i in range(max(0, id - r), min(row, id + r + 1))]


def compete1(x, w, eta, r):
    for xx in x:
        id = WTA2(xx, w)
        iddim = neighborid1(id, w.shape[0], r)
        for iidd in iddim:
            w[iidd] = w[iidd] + eta * (xx - w[iidd])
    return w


# 主函数：初始化参数并进行训练
def train_sofm(num_data=100, num_neurons=15, train_steps=100, eta_begin=0.3, eta_end=0.01, radius_begin=5,
               radius_end=0):
    # 生成三角形区域的随机数据
    data = generate_data_triangle(num_data)

    # 初始化神经元
    neurons = np.random.rand(num_neurons, data.shape[1])
    neurons[:, 1] *= sqrt(3) / 2  # 限制 y 坐标在三角形内

    # 训练过程
    for i in range(train_steps):
        eta = (eta_begin - eta_end) * (train_steps - i) / (train_steps - 1) + eta_end
        radius = int((radius_begin - radius_end) * (train_steps - i) / (train_steps - 1) + radius_end)

        neurons = compete1(data, neurons, eta, radius)

        if (i + 1) % (train_steps // 10) == 0:
            show_data(data, neurons, lineflag=1, title=f"Step: {i + 1}/{train_steps}, Radius: {radius}, Eta: {eta:.2f}")

    # 显示最终结果
    show_data(data, neurons, lineflag=1, title="Final Neuron Distribution")


# 调用主函数
train_sofm()
