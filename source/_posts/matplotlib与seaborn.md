---
title: matplotlib&seaborn 包的学习
date: 2023-04-16 10:56:43
categories: [DataScience, Visualization]
tags: [python, matplotlib, seaborn]
---

> [matplotlib 官方文档](https://matplotlib.org/stable/index.html)

# matplotlib 基础概念

## 图的组成

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202304161118323.png)

上图给出了 matplotlib 图像中各种组成部分，其中比较重要的有：

- `Axis` - 单个轴，比如 x 轴或是 y 轴。
- `Axes` - 坐标系，一个坐标系可以包含很多个轴。
- `Figure` - 图，一个图可以包含很多个坐标系，可以理解每个坐标系是一份子图。
- `Artist` - 图中可以看到的所有东西。

## 绘图函数的输入类型

官方文档的意思是希望是一个 `numpy.array` 对象。

## 代码风格

假设我需要画出 y = -x+4 的折线图：

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202304161151618.png)

我的代码风格大概分为两种：

### 面向对象风格

显式创建 `Figure` 与 `Axes` ，将它们视为对象，调用它们的方法。代码如下：

```python
x = np.linspace(-50, 50, 100)
fig, ax = plt.subplots()
ax.plot(x, -x+4)
ax.set_xlabel('x axis')
ax.set_ylabel('y axis')
ax.set_title('y=-x+4')
plt.show()
```

### pyplot/matlab 风格

依赖 `pyplot` 画图。代码如下：

```python
x = np.linspace(-50, 50, 100)
plt.plot(x, -x+4)
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('y=-x+4')
plt.show()
```

