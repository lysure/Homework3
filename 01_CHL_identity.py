import numpy as np
import matplotlib.pyplot as plt
import random

# 样本数据定义（字母 C, H, L 的 5x5 点阵）
c_data = [0,1,1,1,0,1,0,0,0,1,1,0,0,0,0,1,0,0,0,1,0,1,1,1,0]
h_data = [1,0,0,0,1,1,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1,0,0,0,1]
l_data = [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1,1,1,1,0,0,0,0,0]
x_data = np.array([c_data, h_data, l_data]).astype('float32')

# 初始化神经网络权重
Wcit = np.random.rand(3, x_data.shape[1])  # 3个神经元，每个神经元25个权重

# 添加噪声的函数
def add_noise(x, nn=1):
    """在样本数据中随机选择若干个点添加噪声"""
    for sample in x:
        for _ in range(nn):
            idx = random.randint(0, x.shape[1] - 1)
            sample[idx] = 1.0 if sample[idx] == 0 else 0.0
    return x

# 赢家为王算法
def WTA(x, w):
    """计算输入样本与每个神经元的距离，返回距离最小的神经元索引"""
    dist = np.array([(x - neuron).dot(x - neuron) for neuron in w])
    return np.argmin(dist)

# 竞争学习函数
def compete(x, w, eta):
    """对输入数据进行竞争学习，更新权重"""
    for sample in x:
        winner = WTA(sample, w)
        w[winner] += eta * (sample - w[winner])  # 更新获胜神经元的权重
    return w

# 可视化函数
def plot_weights(w, title):
    plt.clf()
    for i, neuron_weights in enumerate(w):
        x_offset = i * 6  # 每个字母在 x 轴方向的偏移
        for j, weight in enumerate(neuron_weights):
            row, col = divmod(j, 5)
            y, x = 5 - row, x_offset + col  # 显示权重的网格位置
            plt.scatter(x, y, s=weight * 500, c='b', alpha=0.5)
    plt.title(title)
    plt.xlim(-1, 18)
    plt.ylim(-1, 7)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.pause(0.2)

# 训练过程
for epoch in range(100):
    eta = 0.5 * (1 - epoch / 100)  # 学习率逐渐减小
    noisy_data = add_noise(x_data.copy(), nn=1)  # 每轮生成带有噪声的样本
    Wcit = compete(noisy_data, Wcit, eta)  # 竞争学习更新权重
    plot_weights(Wcit, f'Step {epoch + 1}, eta={eta:.2f}')  # 可视化当前权重

plt.show()

# 输出最终的权重矩阵
print("最终训练后的权重矩阵：")
print(Wcit)

# 生成剩余的 20 个噪声样本
def generate_noise_samples(original_sample):
    noise_samples = []
    for i in range(len(original_sample)):
        noisy_sample = original_sample.copy()
        noisy_sample[i] = 1.0 if noisy_sample[i] == 0 else 0.0  # 翻转一个位置的值
        noise_samples.append(noisy_sample)
    return np.array(noise_samples)

# 使用训练后的 Wcit 权重对噪声样本进行分类
def classify_samples(noise_samples, w, original_sample_class):
    correct_predictions = 0
    for sample in noise_samples:
        predicted_class = WTA(sample, w)
        if predicted_class == original_sample_class:
            correct_predictions += 1
    accuracy = correct_predictions / len(noise_samples)
    return accuracy

# 对三个字母生成噪声样本并进行测试
c_noise_samples = generate_noise_samples(c_data)
h_noise_samples = generate_noise_samples(h_data)
l_noise_samples = generate_noise_samples(l_data)

c_accuracy = classify_samples(c_noise_samples, Wcit, 0)
h_accuracy = classify_samples(h_noise_samples, Wcit, 1)
l_accuracy = classify_samples(l_noise_samples, Wcit, 2)

print("字母 C 噪声样本识别准确率：", c_accuracy * 100, "%")
print("字母 H 噪声样本识别准确率：", h_accuracy * 100, "%")
print("字母 L 噪声样本识别准确率：", l_accuracy * 100, "%")
