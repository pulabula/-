# 应用谱聚类算法对图像进行分割

## 目的

本实验的目的是应用谱聚类算法对图像进行分割。通过实现幂方法和谱聚类算法，我们将学习如何将图像建模为数学对象，并使用这些方法来识别和分割图像中的对象。

## 步骤

### 导入所需的库

```python
import cv2
from PIL import Image
from scipy import sparse
import numpy as np
import math
import time
```

cv2和Image用于处理和显示图像，sparse、np和math、用于科学计算，time用于计算程序运行时间。

### 问题1：读取PNG图像文件并进行下采样

首先，我们需要编写一个函数来读取PNG图像文件，并将其应用高斯模糊平滑再下采样为100×100的图像。

```python
def sampling(src, dst):
    # 读取png文件
    img = cv2.imread(src)
    # 应用高斯模糊 要调
    img_blurred = cv2.GaussianBlur(img, (11, 11), 11)  # 消除环境光?！！！！！！不然熊右上角是黑的呜呜呜呜-_-
    # 创建100x100图像
    img_sampled = cv2.resize(img_blurred, (100, 100))
    # 显示图像 这一步至少4s
    img = Image.fromarray(img_sampled)
    img.show(img)  # 100*100图
    return img_sampled
```

### 问题2：构造相似图

接下来，我们需要编写一个函数来构造表示给定图像的相似图。该函数应该打印所构造的图的边数和平均度，返回值为邻接矩阵和做$-\frac{1}{2}$次幂处理的度矩阵。

```python
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
            # 选取范围问题很大！！！！！
            for k in range(w - 2, w + 2):
                for l in range(h - 2, h + 2):
                    k = (k + w) % w
                    l = (l + h) % h
                    # 去除自圈
                    if i == k and j == l:
                        continue
                    pos_kl = np.asarray([k / w, l / h])
                    distances = img_sampled[i, j] - img_sampled[k, l]
                    s = math.exp(
                        -4 * (np.dot(distances, distances) + ((i - k) / w) ** 2 + ((j - l) / h) ** 2)
                    )
                    # 记录coo_matrix数据
                    S.append(s)
                    row.append(i * h + j)  # i=row//h, j=row%h
                    col.append(k * h + l)
                    count += 1
                    d += 1
            if d != 0:
                D.append(math.pow(d, -0.5))  # 便于后续归一化
            else:
                D.append(d)
    # 计算平均度和边数
    degree = count / (w * h)
    print("边数：", count)
    print("平均度：", degree)
    similarity_graph = sparse.coo_matrix((S, (row, col)), shape=(h * w, h * w))
    degree_matrix = sparse.coo_matrix(
        (D, ([i for i in range(h * w)], [i for i in range(h * w)])),
        shape=(h * w, h * w),
    )
    return similarity_graph, degree_matrix
```

### 问题3：计算特征向量

然后，我们需要编写一个函数来计算归一化的拉普拉斯矩阵的第二小特征值（可以剔除噪声？）对应的特征向量。我们将使用幂法近似来实现这一点。而幂法近似计算的是最大特征值对应的特征向量，我们可以通过将$L_G$转化为$2I-L_G=I+D^{-\frac{1}{2}}AD^{-\frac{1}{2}}$，$2I-L_G$是半负定的，从而将求第二小特征值对应的特征向量转化为求最大特征值对应的特征向量。

```python
def Laplacian_matrix(similarity_graph, degree_matrix, img_sampled, w, h):
    # 计算Laplacian_matrix
    I = sparse.eye(w * h)
    L_R = I + degree_matrix * similarity_graph * degree_matrix  # 将转化为半负定，最大的就是L的第二小的
    # 幂次法
    k = 500  # 迭代k次
    print("迭代次数k=", k)
    x_min = np.random.choice([1, -1], size=w * h)
    for i in range(k):
        x_min = L_R * x_min
    x_min = x_min / math.sqrt(np.dot(x_min, x_min))
    # 显示投影强度
    img_projection = np.zeros((w, h))
    for i in range(w):
        for j in range(h):
            img_projection[i, j] = 10000000 * (x_min[i * h + j] ** 2) # 放大值更好显示
    img = Image.fromarray(img_projection)
    img.show(img)
    return x_min
```

### 问题4：谱聚类算法

最后，我们需要编写一个函数来实现谱聚类算法。我们将使用问题3得到的特征向量进行投影，将相似图的拉普拉斯矩阵投影至特征向量子空间，再利用k均值算法进行聚类。

```python
def kmeans(x, w, h):
    # x = 10000 * x
    cluster = np.zeros((w, h))
    # print(x)
    k = 30
    m1 = x[0]
    m2 = x[w // 2 * h + h // 2]
    for t in range(k):
        m1_next = 0
        count1 = 0
        m2_next = 0
        count2 = 0
        for i in range(w):
            for j in range(h):
                if (x[i * h + j] - m1) ** 2 <= (x[i * h + j] - m2) ** 2:
                    cluster[i, j] = 1000000
                    m1_next += x[i * h + j]
                    count1 += 1
                else:
                    cluster[i, j] = 0
                    m2_next += x[i * h + j]
                    count2 += 1
        m1 = m1_next / count1
        m2 = m2_next / count2
    img = Image.fromarray(cluster)
    img.show(img)
```

## 实验结果与分析

在实验过程中，我们使用了三个测试图像来评估我们的实现。通过观察实验结果，我们发现谱聚类算法能够有效地将图像分割成不同的区域。然而，由于时间限制，我们无法进一步优化参数以提高输出质量。在未来的工作中，我们可以尝试调整幂法中的迭代次数、谱聚类的簇数等参数，以获得更好的分割效果。

- bear：

  ![bear原图](/img/bear.png)
  
  ![bear高斯模糊](/img/bear_blurred.png)![bear向量投影](/img/bear_projection.png)![bear](/img/bear_seg.png)

- sheep：

  ![sheep原图](/img/sheep.png)
  
  ![sheep高斯模糊](/img/sheep_blurred.png)![sheep向量投影](/img/sheep_projection.png)![sheep](/img/sheep_seg.png)

- shuttle：

  ![shuttle原图](/img/shuttle.png)

  ![shuttle高斯模糊](/img/shuttle_blurred.png)![shuttle向量投影](/img/shuttle_projection.png)![shuttle](/img/shuttle_seg.png)
