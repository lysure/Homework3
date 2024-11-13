import numpy as np
import matplotlib.pyplot as plt
import os

# 定义三种位置分布的数据
distributions = [
    np.array([[349, 198], [268, 510], [736, 381], [1048, 187], [924, 480],
              [969, 682], [1034, 793], [597, 754], [631, 556], [173, 304]]) / [1200, 800],

    np.array([[297, 338], [403, 604], [736, 381], [1039, 286], [668, 553],
              [929, 598], [900, 137], [606, 761], [304, 448], [521, 430]]) / [1200, 800],

    np.array([[369, 170], [713, 415], [742, 600], [828, 325], [876, 675],
              [106, 340], [1038, 803], [845, 823], [1165, 151], [546, 814]]) / [1200, 800]
]


# 显示数据函数
def show_data(data, lineflag=0, title=''):
    plt.scatter(data[:, 0], data[:, 1], s=10 if lineflag == 0 else 35,
                c='blue' if lineflag == 0 else 'red', label='View Site' if lineflag == 0 else 'Neural Position')
    if lineflag == 1:
        plt.plot(data[:, 0], data[:, 1], 'y--', linewidth=1)

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.axis([-0.05, 1.05, -0.05, 1.05])
    if title:
        plt.title(title)

    plt.grid(True)
    plt.tight_layout()
    plt.legend(loc='upper right')


# 胜者为王算法（WTA）
def WTA2(x, w):
    dist = np.array([(x - ww).dot(x - ww) for ww in w])
    return np.argmin(dist)


# 邻域函数
def neighborid10(id, total_neurons, r):
    return [(id + i) % total_neurons for i in range(-r, r + 1)]


# 竞争更新权重
def compete1(x, w, eta, r):
    for xx in x:
        id = WTA2(xx, w)
        neighbors = neighborid10(id, w.shape[0], r)
        for neighbor_id in neighbors:
            w[neighbor_id] += eta * (xx - w[neighbor_id])
    return w


# 训练参数设置
TRAIN_NUM = 200
ETA_BEGIN = 0.3
ETA_END = 0.01
RATIO_BEGIN = 2
RATIO_END = 0

# 针对每个位置分布进行优化
for idx, x_data in enumerate(distributions):
    print(f"优化第 {idx + 1} 种位置分布")

    SAMPLE_NUM = x_data.shape[0]
    NEURAL_NUM = SAMPLE_NUM
    W = np.random.rand(NEURAL_NUM, x_data.shape[1])

    plt.draw()
    plt.pause(0.2)

    for i in range(TRAIN_NUM):
        eta = ETA_END + (ETA_BEGIN - ETA_END) * np.exp(-5 * i / TRAIN_NUM)
        ratio = max(int(RATIO_END + (RATIO_BEGIN - RATIO_END) * np.exp(-5 * i / TRAIN_NUM)), RATIO_END)

        W = compete1(x_data, W, eta, ratio)

    # 绘制并展示最终结果
    plt.clf()
    show_data(x_data, lineflag=0)
    show_data(W, lineflag=1, title=f"Final result for distribution {idx + 1}")
    plt.show()
