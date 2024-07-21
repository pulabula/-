import cv2
from PIL import Image, ImageFilter
from scipy import sparse
import numpy as np
import math
import time
import matplotlib.pyplot as plt


# 问题1：采样
def sampling(src="source/img/bear.png", dst="source/img/bear_blurred.png"):
    # 读取png文件
    img = cv2.imread(src)
    # 应用高斯模糊 要调
    # 若先100*100 高斯选（11，11），11
    # img_blurred = cv2.GaussianBlur(img, (11, 11), 11)  # 效果有时候好？
    # img_sampled = cv2.resize(img_blurred, (100, 100))
    original_width = img.shape[0]
    original_height = img.shape[1]
    img_blurred = cv2.resize(img, (int(100 / original_width * original_height), 100))
    # img_blurred = cv2.resize(img, (100, 100))
    img_sampled = cv2.GaussianBlur(img_blurred, (3, 3), 1)
    # 创建100x100图像

    # 显示图像 这一步至少4s
    img = Image.fromarray(img_sampled)
    # img.show(img)  # 100*100图
    img.save(dst)
    return img_sampled


def similarity_graph(img_sampled, w, h):
    # 颜色值归一化
    img_sampled = img_sampled / 255
    # # 初始化相似图，（i，j）和（k，l）的相似度
    # similarity_graph = np.zeros(( w,  h,  w,  h))
    # 记录coo_matrix数据
    S, D, row, col = [], [], [], []
    count = 0
    for i in range(w):
        for j in range(h):
            d = 0  # 记录度
            pos_ij = np.asarray([i / w, j / h])
            # pos_ij = np.asarray([i, j])
            # 选取范围问题很大！！！！！
            for k in range(w - 4, w + 4):
                for l in range(h - 4, h + 4):
                    k = (k + w) % w
                    l = (l + h) % h
                    # 去除自圈
                    if i == k and j == l:
                        continue
                    pos_kl = np.asarray([k / w, l / h])
                    # pos_kl = np.asarray([k, l])
                    # 将位置信息和颜色信息一起计算距离
                    # distances = np.insert(img_sampled[i, j], 0, pos_ij) - np.insert(
                    #     img_sampled[k, l], 0, pos_kl
                    # )
                    distances = img_sampled[i, j] - img_sampled[k, l]
                    s = math.exp(
                        -4 * (np.dot(distances, distances) + ((i - k) / w) ** 2 + ((j - l) / h) ** 2)
                    )
                    # 设置阈值t=0.5
                    # if s > 0.1:
                    # 记录coo_matrix数据
                    S.append(s)
                    row.append(i * h + j)  # i=row//h, j=row%h
                    col.append(k * h + l)
                    count += 1
                    # d += 1  # 这个sheep效果好，bear杂点比较多，shuttle比较好，但是这个要求高斯模糊开的高
                    d += s  # 这个bear比+1效果好
            if d != 0:
                D.append(math.pow(d, -0.5))  # 便于后续归一化
            else:
                D.append(d)
    # 计算平均度和边数
    # count = np.count_nonzero(similarity_graph)
    degree = count / (w * h)
    print("边数：", count)
    print("平均度：", degree)
    similarity_graph = sparse.coo_matrix((S, (row, col)), shape=(h * w, h * w))
    degree_matrix = sparse.coo_matrix(
        (D, ([i for i in range(h * w)], [i for i in range(h * w)])),
        shape=(h * w, h * w),
    )
    # print(similarity_graph)
    # print(degree_matrix)
    return similarity_graph, degree_matrix, D


def Laplacian_matrix(
    similarity_graph, degree_matrix, x_d, img_sampled, w, h, dst="source/img/bear_projection.png"
):
    # 计算Laplacian_matrix
    I = sparse.eye(w * h)
    L = I - degree_matrix * similarity_graph * degree_matrix  # degree_matrix在上一问已处理，否则报NAN
    L_R = I + degree_matrix * similarity_graph * degree_matrix  # 将转化为半负定，最大的就是L的第二小的
    # L = degree_matrix - similarity_graph
    # 幂次法
    k = 500  # 迭代k次
    print("迭代次数k=", k)
    x_min = np.random.choice([1, -1], size=w * h)
    # x_max = np.random.choice([1, -1], size=w * h)
    for i in range(k):
        x_min = L_R * x_min
        # x_max = L * x_max
    x_min = x_min / math.sqrt(np.dot(x_min, x_min))
    # 尝试施密特正交化求两个
    # x_max = np.random.choice([1, -1], size=w * h) - np.dot(x_min, x_min) * x_min
    # for i in range(k):
    #     x_max = L_R * x_max
    # x_max = x_max / math.sqrt(np.dot(x_max, x_max))
    # 显示投影强度
    # x_min = similarity_graph @ x_min  # 做投影？？？会形成一个轮廓，不对，只是放大或缩小了x_min
    # x_max = similarity_graph @ x_max  # 做投影？？？会形成一个轮廓
    img_projection = np.zeros((w, h))
    for i in range(w):
        for j in range(h):
            img_projection[i, j] = x_min[i * h + j] ** 2
            # img_projection[i, j] = x_d[i * h + j] ** 2
            # img_projection[i, j] = 10000000 * (x_max[i * h + j] ** 2 + x_min[i * h + j] ** 2)

    # print(img_projection)
    # img = Image.fromarray(img_projection) # 这种方式需要 x_min[i * h + j] ** 2 * 10000000
    # img.show(img)
    # cv2.imwrite(dst, img_projection)
    # 另一种显示图像的方案
    plt.figure(1)
    plt.imshow(img_projection, cmap="viridis")
    plt.colorbar()
    plt.savefig(dst)
    return x_min


def kmeans(x, x_d, w, h, dst="source/img/bear_seg.png"):
    # x = x
    cluster = np.zeros((w, h))
    # print(x)
    k = 100
    m1 = x[0]
    m2 = x[w // 2 * h + h // 2]
    # m3 = x[w // 4 * h + h // 4]
    # m4 = x[w // 3 * h + h // 3]

    for t in range(k):
        m1_next = 0
        count1 = 0
        m2_next = 0
        count2 = 0
        # m3_next = 0
        # count3 = 0
        # m4_next = 0
        # count4 = 0
        for i in range(w):
            for j in range(h):
                if (x[i * h + j] - m1) ** 2 < (x[i * h + j] - m2) ** 2:
                    cluster[i, j] = 1
                    m1_next += x[i * h + j]
                    count1 += 1
                else:
                    cluster[i, j] = 0
                    m2_next += x[i * h + j]
                    count2 += 1
                # elif (x[i * h + j] - m3) ** 2 < (x[i * h + j] - m4) ** 2:
                #     cluster[i, j] = 3
                #     m3_next += x[i * h + j]
                #     count3 += 1
                # else:
                #     cluster[i, j] = 4
                #     m4_next += x[i * h + j]
                #     count4 += 1

        m1 = m1_next / count1
        m2 = m2_next / count2
        # m3 = m3_next / count3
        # m4 = m4_next / count4
    # img = Image.fromarray(cluster)
    # img.show(img)
    # cv2.imwrite(dst, cluster)
    # 另一种显示图像的方案
    plt.figure(2)
    plt.imshow(cluster)
    plt.colorbar()
    plt.savefig(dst)
    plt.show()


if __name__ == "__main__":
    src = "img/bear.png"
    dst1 = "img/bear.png_blurred.png"
    dst2 = "img/bear.png_projection.png"
    dst3 = "img/bear.png_seg.png"

    # 开始时间
    start_time = time.perf_counter()

    # 问题1：采样
    img_sampled = sampling(src, dst1)
    # 结束时间
    end_time = time.perf_counter()
    print("至问题1程序运行时间：", end_time - start_time, "秒")

    # 计算长、高
    w = img_sampled.shape[0]
    h = img_sampled.shape[1]

    # 问题2：构造相似图，返回值为一个相似矩阵（稀疏）和度**-0.5矩阵
    S, D, x_d = similarity_graph(img_sampled, w, h)
    # 结束时间
    end_time = time.perf_counter()
    print("至问题2程序运行时间：", end_time - start_time, "秒")

    # 问题3：计算L和其第二小特征值对应的向量，返回x
    x = Laplacian_matrix(S, D, x_d, img_sampled, w, h, dst2)
    # 结束时间
    end_time = time.perf_counter()
    print("至问题3程序运行时间：", end_time - start_time, "秒")

    # 问题3：k-means
    kmeans(x, x_d, w, h, dst3)
    # 结束时间
    end_time = time.perf_counter()
    print("至问题4程序运行时间：", end_time - start_time, "秒")
