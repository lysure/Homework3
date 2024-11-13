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

# 训练过程
for epoch in range(100):
    eta = 0.5 * (1 - epoch / 100)  # 学习率逐渐减小
    noisy_data = add_noise(x_data.copy(), nn=1)  # 每轮生成带有噪声的样本
    Wcit = compete(noisy_data, Wcit, eta)  # 竞争学习更新权重

# 输出最终的权重矩阵
print("最终训练后的权重矩阵：")
print(Wcit)

# 生成随机的海明距离为 2 的噪声样本
def generate_hamming2_samples(original_sample, num_samples=20):
    noise_samples = []
    for _ in range(num_samples):
        noisy_sample = original_sample.copy()
        idx1, idx2 = random.sample(range(len(original_sample)), 2)  # 随机选择两个不同的位置
        noisy_sample[idx1] = 1.0 if noisy_sample[idx1] == 0 else 0.0
        noisy_sample[idx2] = 1.0 if noisy_sample[idx2] == 0 else 0.0
        noise_samples.append(noisy_sample)
    return np.array(noise_samples)

# 使用训练后的 Wcit 权重对海明距离为 2 的噪声样本进行分类
def classify_samples(noise_samples, w, original_sample_class):
    correct_predictions = 0
    for sample in noise_samples:
        predicted_class = WTA(sample, w)
        if predicted_class == original_sample_class:
            correct_predictions += 1
    accuracy = correct_predictions / len(noise_samples)
    return accuracy

# 对三个字母生成海明距离为 2 的噪声样本并进行测试
c_noise_samples_h2 = generate_hamming2_samples(c_data)
h_noise_samples_h2 = generate_hamming2_samples(h_data)
l_noise_samples_h2 = generate_hamming2_samples(l_data)

c_accuracy_h2 = classify_samples(c_noise_samples_h2, Wcit, 0)
h_accuracy_h2 = classify_samples(h_noise_samples_h2, Wcit, 1)
l_accuracy_h2 = classify_samples(l_noise_samples_h2, Wcit, 2)

print("字母 C 海明距离为 2 的噪声样本识别准确率：", c_accuracy_h2 * 100, "%")
print("字母 H 海明距离为 2 的噪声样本识别准确率：", h_accuracy_h2 * 100, "%")
print("字母 L 海明距离为 2 的噪声样本识别准确率：", l_accuracy_h2 * 100, "%")
