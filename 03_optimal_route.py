#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import imageio
import os

# 自定义 PlotGIF 类
class PlotGIF:
    def __init__(self, save_path='temp_gif'):
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.frames = []

    def append(self, plt_figure):
        # 保存当前图像帧
        frame_path = os.path.join(self.save_path, f'frame_{len(self.frames)}.png')
        plt_figure.savefig(frame_path)
        self.frames.append(frame_path)

    def save(self, output_path):
        # 将所有保存的帧生成GIF
        with imageio.get_writer(output_path, mode='I', duration=0.1) as writer:
            for frame_path in self.frames:
                image = imageio.imread(frame_path)
                writer.append_data(image)
        # 清理临时文件
        for frame_path in self.frames:
            os.remove(frame_path)
        os.rmdir(self.save_path)
        print(f"GIF saved at {output_path}")

# 景点坐标数据
x_data = [[236,53], [408,79], [909,89], [115,264], [396,335],
          [185,456], [699,252], [963,317], [922,389], [649,515]]
x_data = np.array([[xy[0] / 1000, xy[1] / 600] for xy in x_data])  # 归一化坐标

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


# 参数初始化
SAMPLE_NUM = x_data.shape[0]
NEURAL_NUM = SAMPLE_NUM

# 初始化权重
W = np.random.rand(NEURAL_NUM, x_data.shape[1])

# 胜者为王算法（WTA）
def WTA2(x, w):
    """找到与样本x最近的权重节点索引"""
    dist = np.array([(x - ww).dot(x - ww) for ww in w])
    return np.argmin(dist)

#------------------------------------------------------------
# 邻域函数
def neighborid10(id, total_neurons, r):
    """获取指定半径r内的邻域索引"""
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

plt.draw()
plt.pause(0.2)

# GIF动画保存类实例化
pltgif = PlotGIF()

# 训练过程
for i in range(TRAIN_NUM):
    # 使用指数衰减更新学习率和邻域范围
    eta = ETA_END + (ETA_BEGIN - ETA_END) * np.exp(-5 * i / TRAIN_NUM)
    ratio = max(int(RATIO_END + (RATIO_BEGIN - RATIO_END) * np.exp(-5 * i / TRAIN_NUM)), RATIO_END)

    # 更新权重
    W = compete1(x_data, W, eta, ratio)

    # 可视化训练过程
    plt.clf()
    show_data(x_data, lineflag=0)
    show_data(W, lineflag=1, title=f"Step:{i + 1}/{TRAIN_NUM}, R:{ratio}, eta:{eta:.4f}")
    plt.draw()
    plt.pause(0.001)
    pltgif.append(plt)

# 保存GIF
pltgif.save(r'd:\temp\1.gif')
plt.show()
#!/usr/local/bin/python
# -*- coding: gbk -*-
