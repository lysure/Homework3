import numpy as np
import matplotlib.pyplot as plt
from numpy import exp, dot, zeros, array, linspace, eye, linalg, where, amin, random, squeeze, vstack

# 定义 Hermite 多项式函数
def f_hermit(x):
    return 1.1 * (1 - x + 2 * x**2) * exp(-x**2 / 2)

# 生成训练数据
TRAIN_DATA_NUM = 500
x_train = random.uniform(-4, 4, TRAIN_DATA_NUM)
y_train = f_hermit(x_train)
xy_train = vstack((x_train, y_train)).T

# Winner-Take-All 函数
def WTA_nearest(x, v):
    err = [x - vv for vv in v]
    dist = [dot(e, e) for e in err]
    id = where(dist == amin(dist))[0][0]
    return id

# 邻近更新函数
def K_neighbor(v, xy, eta):
    for xy_point in xy:
        id = WTA_nearest(xy_point[0], v[:, 0])
        v[id] = v[id] + eta * (xy_point - v[id])
    return v

# 输出隐藏节点
def CPN_v_out(x, v):
    H = []
    for xx in x:
        h = zeros(v.shape[0])
        id = WTA_nearest(xx, v)
        h[id] = 1
        H.append(list(h))
    return array(H).T

# 计算权重矩阵
def CPN_W(h, y):
    vv = h.T.dot(h) + 0.000001 * eye(h.shape[1])
    vvv = linalg.inv(vv).dot(h.T)
    return y.reshape(1, -1).dot(vvv)

# 初始化节点数和种子节点
NODE_NUM = 50
v_seed_x = x_train[:NODE_NUM]
v_seed_y = f_hermit(v_seed_x)
v_data = vstack((v_seed_x, v_seed_y)).T

# 设置学习率和训练步数
ETA_BEGIN = 0.1
ETA_END = 0.0
TRAIN_STEP = 100

# 训练双向 CPN
for eta in linspace(ETA_BEGIN, ETA_END, TRAIN_STEP):
    v_data = K_neighbor(v_data, xy_train, eta)

# 最终计算网络输出
H = CPN_v_out(x_train, squeeze(v_data[:, 0]))
W = CPN_W(H, y_train)
yy = W.dot(H)

# 生成最后的拟合曲线
x_line = linspace(-4, 4, 500)
Hx = CPN_v_out(x_line, squeeze(v_data[:, 0]))
y_line = W.dot(Hx)

# 显示最终结果
plt.figure()
plt.plot(x_line, f_hermit(x_line), '--', color='blue', linewidth=1, label='Hermite Function')
plt.plot(x_line, y_line[0], color='orange', linewidth=1, label='CPN Approximation')
plt.scatter(x_train, y_train, s=10, color='green', label='Training Data')
plt.scatter(squeeze(v_data[:, 0]), squeeze(v_data[:, 1]), s=30, color='red', label='Hidden Nodes')
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.legend(loc='upper right')
plt.title('Final Result of Hermite Polynomial Approximation using Bidirectional CPN')
plt.tight_layout()
plt.show()

# 计算均方误差
mse = np.mean((y_train - yy[0])**2)
print(f"Mean Squared Error: {mse}")