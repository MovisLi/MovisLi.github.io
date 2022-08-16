---
title: 《Excel+Python：飞速搞定数据分析与处理》 - 读书笔记
date: 2022-08-16 23:30:47
categories: [DataScience, Excel]
tags: [python, excel]
---

# Python 入门

作者浅讲了一下 Python 在数据分析方面与 Excel 交汇的历史。

提到了应该坚守的原则：

- **模块化** - 一个应用程序通常被分为 3 层。
  - 数据层（ data layer ） - 也叫输入层，取数。
  - 业务层（ business layer ） - 也叫计算层，数据的处理与转换、分析。
  - 表示层（ presentation layer ） - 也叫输出层，数据分析结果的呈现。
- **DRY 原则** - Don't repeat yourself ，不要写重复的代码，就是函数式编程。
- **测试**
  - 作者提供了一篇他的文章 [Unit Tests for Microsoft Excel](https://www.xlwings.org/blog/unittests-for-microsoft-excel) 供参考。
- **版本控制**
  - 第一个选项，使用 xltrail ，这是个基于 Git 对 Excel 友好的版本控制系统，但是我选择用第二个选项。
  - 第二个选项，把业务逻辑迁移到 Python 中，然后用 Git 。

在面向对象概念中，

- variables 被称为 attribute 。
- function 被称为 method 。

Excel 单元格保存数值型数字时永远是浮点数。

在 Python 中，每个对象都可以被视作 True 或 False 。大部分对象会被视作 True ，但 None 、 False 、 0 或空数据类型（比如空字符串、空列表等）会被视作 False 。

Python 中切片的语法：

```python
sequence[start:stop:step]
```

切片允许越界，这个太好用了。

作者提倡用 [PEP 8 – Style Guide for Python Code | peps.python.org](https://peps.python.org/pep-0008/) 规范代码，当然这个应该是写 Python 人的共识。

# pandas 入门

NumPy 中的矩阵乘法：

- 矩阵的点乘用 `*` ，必须是大小一样的矩阵， NumPy 中也支持不一样的相乘，不过结果会将矩阵1 `[M1, N1]` 与矩阵2 `[M2, N2]` 扩大为 `[max(M1,M2), max(N1,N2)]` ，这在 NumPy 中叫**广播（ broadcasting ）**。
- 矩阵的叉乘用 `@` ，需要矩阵1 `[M1, N1]` 与矩阵2 `[M2, N2]` 满足 `N1=M2` 的条件。

# 在 Excel 之外读写 Excel 文件

# 使用 xlwings 对 Excel 应用程序进行编程
