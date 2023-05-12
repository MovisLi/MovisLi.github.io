---
title: matplotlib&seaborn 包的学习
date: 2023-04-16 10:56:43
categories: [DataScience, Visualization]
tags: [python, matplotlib, seaborn]
---

> [matplotlib 官方文档](https://matplotlib.org/stable/index.html)

# matplotlib 基础概念

## 1 - 快速开始

### 图的组成

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202304161118323.png)

上图给出了 matplotlib 图像中各种组成部分，其中比较重要的有：

- `Axis` - 单个轴，比如 x 轴或是 y 轴。
- `Axes` - 坐标系，一个坐标系可以包含很多个轴。
- `Figure` - 图，一个图可以包含很多个坐标系，可以理解每个坐标系是一份子图。
- `Artist` - 图中可以看到的所有东西。

### 输入类型

官方文档的意思是希望是一个 `numpy.array` 对象。

### 代码风格

假设我需要画出 y = -x+4 的折线图：

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202304161151618.png)

我的代码风格大概分为两种：

#### 面向对象风格

显式创建 `Figure` 与 `Axes` ，将它们视为对象，调用它们的方法。**这种手法比较灵活。**

代码如下：

```python
x = np.linspace(-50, 50, 100)
fig, ax = plt.subplots()
ax.plot(x, -x+4)
ax.set_xlabel('x axis')
ax.set_ylabel('y axis')
ax.set_title('y=-x+4')
plt.show()
```

#### pyplot/matlab 风格

依赖 `pyplot` 画图。**这种手法比较简洁。**

代码如下：

```python
x = np.linspace(-50, 50, 100)
plt.plot(x, -x+4)
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('y=-x+4')
plt.show()
```

#### 函数包装

官方的示例是将 `Axes` 作为一个参数传入函数，在此坐标系上画图。

```python
def write_func(ax, x, y, *args, **kwargs):
    out = ax.plot(x, y, *args, **kwargs)
    return out


x = np.linspace(-50, 50, 100)
fig, ax = plt.subplots()
write_func(ax, x, -x + 4)
```

### 元素样式

包括填充颜色，轮廓颜色，线的样式，线的宽度等。

比如：

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202304171141651.png)

代码如下：

```python
x = np.linspace(-50, 50, 100)
fig, ax = plt.subplots()
# 设置颜色为红色，线宽度为 5，线样式为虚线
ax.plot(x, -x+4, color='r', linewidth=5, linestyle='--')
ax.set_xlabel('x axis')
ax.set_ylabel('y axis')
ax.set_title('y=-x+4')
plt.show()
```

### 标签画图

标签的显示方式和副轴标签的显示方式如下。

假设图为：

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202304171415979.png)

代码如下：

```python
data = pd.DataFrame({'type': ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'B'], 'y': [1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0]})
data2 = data.groupby(['type'])['y'].agg(['count', 'sum']).reset_index()
data2['ratio'] = (data2['sum']/data2['count']).round(4)
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
l1 = ax1.bar(data2['type'].values, data2['count'].values)
l2, = ax2.plot(data2['type'].values, data2['ratio'].values, color='r')
plt.legend([l1, l2], ['count', '1_probability'])
plt.show()
```

### 轴的刻度

每个 `Axes` 都有两个（或三个） `Axis` 对象，分别代表 X 轴和 Y 轴。这些对象控制轴的比例、刻度定位器和刻度格式。可以附加其他轴来显示更多的轴对象。

比方说我需要画两个图（一个 X 轴正常，另一个 X 轴为对数坐标轴）：

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202304171423188.png)

代码如下：

```python
x = np.linspace(-50, 50, 100)
fig, axs = plt.subplots(1, 2, layout='constrained')
axs[0].plot(x, -x+4)
axs[1].set_xscale('log')
axs[1].plot(x, -x+4)
plt.show()
```

### 主题颜色

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202304180846385.png)

代码如下：

```python
x, y = np.meshgrid([-1, 0, 1], [-1, 0, 1])
z = x+y
fig, ax = plt.subplots()
pc = ax.pcolormesh(x, y, z, vmin=-1, vmax=2)
fig.colorbar(pc, ax=ax)
plt.show()
```

### 复合图与坐标系

官网给的例子是：

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202304180849327.png)

代码如下：

```python
fig, ax = plt.subplot_mosaic([['upleft', 'right'], ['lowleft', 'right']], layout='constrained')
ax['upleft'].set_title('upleft')
ax['lowleft'].set_title('lowleft')
ax['right'].set_title('right')
plt.show()
```

## 2 - Pyplot 教程

### Pyplot 介绍

`matplotlib.pyplot` 是一个函数集合，使 `matplotlib` 像 `MATLAB` 一样工作。每个 `pyplot` 函数都会对一个图形进行一些改变：例如，创建一个图形，在图形中创建一个绘图区，在绘图区绘制一些线条，用标签来装饰图形等等。

