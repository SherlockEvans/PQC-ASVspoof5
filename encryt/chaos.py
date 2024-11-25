import numpy as np
import time
import matplotlib.pyplot as plt
def hxf3dchaos(u0 = 0.3, v0 = 0.4, w0 = 0.5, iterationsvalue = 10, round=1):

    u_0 = u0  # 初始值，可以根据需要调整
    v_0 = v0  # 初始值，可以根据需要调整
    w_0 = w0  # 初始值，可以根据需要调整

    # 定义参数
    p = 10  # 参数，可以根据需要调整
    q = 3  # 参数，可以根据需要调整

    # 定义迭代次数
    iterations = iterationsvalue  # 可以根据需要调整
    i = round

    # 初始化变量
    u = np.empty(i*iterations + 1, dtype=np.float64)
    # print(u)
    v = np.empty([i*iterations + 1], dtype=np.float64)
    w = np.empty([i * iterations + 1], dtype=np.float64)
    # w = np.zeros(i*iterations + 1, dtype=np.float64)

    # 设置初始值
    u[0] = u_0
    v[0] = v_0
    w[0] = w_0

    # 迭代计算
    for k in range(i*iterations):
        #  arccos 的输入范围是 [-1, 1]

        u[k+1] = v[k] - w[k]
        v[k + 1] = np.sin(np.pi * u[k] - p * v[k])
        w[k + 1] = np.cos(q * np.arccos(w[k])+v[k])
        # print(w[k + 1])


    # print(w)
    # print(w.shape)
    u_part = u[(i-1)*iterations+1:i*iterations + 1]
    v_part = v[(i - 1) * iterations + 1:i * iterations + 1]
    w_part = w[(i-1)*iterations+1:i*iterations + 1]
    # print(w_part)
    return u_part,v_part,w_part
    # 打印结果
    # for i in range(iterations + 1):
    #     print(f"u({i}) = {u[i]}, v({i}) = {v[i]}, w({i}) = {w[i]}")


#切比雪夫映射
def Chebyshev(num=10, a=4, x0=0.3, size=1):
    x = []
    if x0 is None:
        x0 = np.random.uniform(-1, 1, size=size)
    for i in range(num):
        x0 = np.cos(a * (np.cos(x0) ** (-1)))
        x.append(x0.copy())
    return x




if __name__ == "__main__":
    iterations = 44600
    T1 = time.time()
    u,v,w=hxf3dchaos(u0=0.245893961547891, v0=0.474895152158748, w0=0.854123695478961, iterationsvalue=44600, round=1)
    u_2,v_2,w_2=hxf3dchaos(u0=0.245893961547897, v0=0.474895152158742, w0=0.854123695478969, iterationsvalue=500,round=1)
    # # print(type(en))
    # # print(en.shape)

    # np.set_printoptions(precision=15)
    # # en=np.around(en,16)
    # print(w)
    # # a=Chebyshev()
    # # print(a)
    # T2 = time.time()
    # print('程序运行时间:%s秒' % ((T2 - T1)))
    # 12,9
    # plt.figure(figsize=(10, 8))
    #
    # plt.subplot(3, 1, 1)
    # plt.plot(range(iterations), u, label='X-1', color='orange')
    # plt.plot(range(iterations), u_2, label='X-2', color='green')
    # plt.title('X')
    # plt.xlabel('Iteration', fontdict={'family': 'Times New Roman', 'size': 12})
    # plt.ylabel('X')
    # plt.legend()
    # plt.xticks([])
    #
    # plt.subplot(3, 1, 2)
    # plt.plot(range(iterations), v, label='Y-1', color='orange')
    # plt.plot(range(iterations), v_2, label='Y-2', color='green')
    # plt.title('Y')
    # plt.xlabel('Iteration',fontdict={'family': 'Times New Roman', 'size': 12})
    # plt.ylabel('Y')
    # plt.legend()
    # plt.xticks([])
    #
    # plt.subplot(3, 1, 3)
    # plt.plot(range(iterations), w, label='Z-1', color='orange')
    # plt.plot(range(iterations), w_2, label='Z-2', color='green')
    # plt.title('Z')
    # plt.xlabel('Iteration',fontdict={'family': 'Times New Roman', 'size': 12})
    # plt.ylabel('Z')
    # plt.legend()
    # plt.xticks([])
    #
    # plt.tight_layout()
    # plt.show()

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')

    # 使用scatter3D绘制点
    # ax.scatter(u, v, w, c=range(iterations), cmap='viridis', marker='.', s=1)

    # 设置标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Scatter Plot of Chaotic Map',fontdict={'family': 'Times New Roman', 'size': 12})


    # 绘制不带颜色的点
    ax.scatter(u, v, w, color='blue', marker='o', s=2)  # 使用不带颜色的点绘制
    # ax.scatter(u_2, v_2, w_2, color='blue', marker='o', s=2)
    # ax.plot(u, v, w, label='参数曲线')
    plt.show()
