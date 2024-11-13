import numpy as np
import matplotlib.pyplot as plt
from numpy import exp, dot, zeros, array, linspace, eye, linalg, where, amin, random

# 定义 Hermite 多项式函数
def f_hermit(x):
    return 1.1 * (1 - x + 2 * x**2) * exp(-x**2 / 2)

# 生成训练数据
TRAIN_DATA_NUM = 500
x_train = random.uniform(-4, 4, TRAIN_DATA_NUM)
y_train = f_hermit(x_train)

# 定义辅助函数
def WTA_nearest(x, v):
    err = [x - vv for vv in v]
    dist = [dot(e, e) for e in err]
    id = where(dist == amin(dist))[0][0]
    return id

def K_neighbor(v, x, eta):
    for xx in x:
        id = WTA_nearest(xx, v)
        v[id] = v[id] + eta * (xx - v[id])
    return v

def CPN_v_out(x, v):
    H = []
    for xx in x:
        h = zeros(v.shape[0])
        id = WTA_nearest(xx, v)
        h[id] = 1
        H.append(list(h))
    return array(H).T

def CPN_W(h, y):
    vv = h.T.dot(h) + 0.000001 * eye(h.shape[1])
    vvv = linalg.inv(vv).dot(h.T)
    return y.reshape(1, -1).dot(vvv)

# 设置节点数和初始节点
NODE_NUM = 50
v_data = x_train[0:NODE_NUM]

# 设置学习率和训练步数
ETA_BEGIN = 0.1
ETA_END = 0.0
TRAIN_STEP = 100

# 训练单向 CPN
for eta in linspace(ETA_BEGIN, ETA_END, TRAIN_STEP):
    # 更新节点
    v_data = K_neighbor(v_data, x_train, eta)

# 计算网络输出
H = CPN_v_out(x_train, v_data)
W = CPN_W(H, y_train)
yy = W.dot(H)

# 显示最终结果
x_line = linspace(-4, 4, 500)
y_line = W.dot(CPN_v_out(x_line, v_data))

plt.figure()
plt.plot(x_line, f_hermit(x_line), '--', c='grey', linewidth=1, label='Hermite Func')
plt.plot(x_line, y_line[0], c='red', linewidth=1, label='CPN Approximation')
plt.scatter(x_train, y_train, s=10, c='darkviolet', label='Train Data')
plt.scatter(v_data, W.dot(CPN_v_out(v_data, v_data))[0], s=30, c='darkcyan', label='V value')
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True)
plt.legend(loc='upper right')
plt.title('Hermite Polynomial Approximation using Unidirectional CPN')
plt.tight_layout()
plt.show()

# 计算均方误差
mse = np.mean((y_train - yy[0])**2)
print(f"Mean Squared Error: {mse}")

