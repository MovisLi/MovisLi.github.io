---
title: openpyxl 包的学习
date: 2022-09-03 17:47:23
categories: [DataScience, Excel]
tags: [python, excel]
---

# 学习资源

官方文档 [openpyxl - A Python library to read/write Excel 2010 xlsx/xlsm files](https://openpyxl.readthedocs.io/en/stable/)

文档给出了一个简单的示例。

```python
from openpyxl import Workbook

# 创建一个工作簿对象
wb = Workbook()

# 获取默认工作表
ws = wb.active

# 修改单元格的值
ws['A1'] = 42

# 添加行
ws.append([1, 2, 3])

# Python 类型会被自动转换
import datetime
ws['A2'] = datetime.datetime.now()

# 保存文件
wb.save("sample.xlsx")
```

最终得到一个这样的表格：

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202209052217498.png)

接下来我跟着 OpenPyXL 的文档学习。

# 教程

## 创建工作簿

```python
from openpyxl import Workbook

wb = Workbook()
```

这里的 `wb` 是一个工作簿对象而不是工作表对象，一个工作簿里可能有很多工作表。所以接下来就是获取工作表对象。

## 打开工作簿

```python
from openpyxl import Workbook

wb = loan_workbook(r'sample.xlsx')
```

打开一个已存在的工作簿用这种方式。至于用哪种方法获取工作簿对象就看需求了。

## 获取工作表

```python
ws = wb.active
```

默认获取最左边（下标为 0 ）的工作表。

当然，如果是读取一个 Excel ，它可能本身就有工作表，也可以这样获取工作表对象。

```python
ws = wb['Sheet1']
```

如果要获取工作簿里所有工作表的名称，则可以用工作簿对象的 `sheetnames` 这个属性。

```python
wb.sheetnames
```

## 创建工作表

```python
>>> ws1 = wb.create_sheet("Mysheet") # 在最后的位置插入工作表，类似 list 的 append 方法
# or
>>> ws2 = wb.create_sheet("Mysheet", 0) # 在第一个位置插入工作表
# or
>>> ws3 = wb.create_sheet("Mysheet", -1) # 在倒数第二个位置插入工作表，因为不写才是在最后插入
```

## 复制工作表

```python
copy_sheet = wb.copy_worksheet(ws)
```

得到这样的新工作表：

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202209052242561.png)

> 仅复制单元格（包括值、样式、超链接和注释）和某些工作表属性（包括尺寸、格式和属性）。不复制所有其他工作簿/工作表属性 - 例如图像，图表。
>
> 也不能在工作簿之间复制工作表。如果工作簿以只读或只写模式打开，则无法复制工作表。

## 工作表的属性

### 工作表的名称

```python
ws.title
```

### 工作表标签的颜色

```python
ws.sheet_properties.tabColor = "FF0000" # 比如我用红色
```

结果就是：

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202209052237731.png)

好像也没有那么红，不过确实跟白色不同。

## 获取单元格

### 单个单元格

```python
ws['A1'] # 获取 A1 这个单元格
# or
ws.cell(row=1, column=1) # 获取第一行第一列这个单元格

ws['A1'] = 1 # 更改 A1 这个表格的值
# or
ws.cell(row=1, column=1, value=1)
```

其实就是一个是以单元格名获取单元格，另一个以行名和列名获取单元格。

> 值得注意的是，单元格是在首次访问它时创建的，只要访问过了，不赋值也会创建。
>
> 这个跟表格的最大行数最大列数会有关系。

### 多个单元格

```python
ws['A1':'C2'] # 获取 A1 到 C2 这个范围的单元格
ws['A':'C'] # 获取 A 列到 C 列的单元格
ws[1:2] # 获取 1,2 行单元格
```

当然也可以用迭代器获取：

```python
# 行优先获取
ws.iter_rows(min_row, min_col, max_row, max_col)

# 列优先获取
ws.iter_rows(min_row, min_col, max_row, max_col)
```

一般使用 `for cells in iterator` ，返回的是行或列的元组而并非每一个格子。

比如：

```python
for cells in ws.iter_rows(min_row=1, min_col=1, max_row=3, max_col=3):
	print(cells)
```

结果是这样的：

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202209052313081.png)

如果想要直接看值，可以为生成器传入 `values_only=True` 生成全是值的迭代器。

```python
ws.iter_rows(min_row=1, max_col=3, max_row=2, values_only=True)
```

当然如果是想获取全部的行列也可以用：

```python
ws.rows
ws.columns
```

## 保存文件

```python
wb.save(r'/openpyxl_sample/sample.xlsx') # 保存到路径
```

这样保存路径，我个人习惯加 `r` 以免遇到反斜杠转义错误。

> 此操作将覆盖现有文件而不发出警告。所以说要小心一点不要随便用，不过这个应该都会的吧， pandas 的 to_csv 这类方法也是一样的。

# 简单使用

## 日期格式

```python
import datetime
from openpyxl import Workbook

wb = Workbook()
ws = wb.active
ws['A1'] = datetime.datetime(2022, 9, 5)
print(ws['A1'].value)
print(ws['A1'].number_format)
```

结果如下：

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202209052342203.png)

## 公式

```python
from openpyxl import Workbook
wb = Workbook()
ws = wb.active

ws['A1'] = 1
ws['B1'] = 1
ws['C1'] = '=SUM(A1:B1)'
print(ws['C1'].value)
```

输出如下：

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202209062142285.png)

不过在表格中是有效的，如下：

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202209052348329.png)

```python
from openpyxl.utils import FORMULAE

print('SUM' in FORMULAE)
```

可以用 `FORMULAE` 检验是否公式可识别。

## 合并/取消合并单元格

合并单元格时，除了左上角的所有单元格都将从工作表中删除。

```python
ws.merge_cells(start_row, start_column, end_row, end_column)
ws.unmerge_cells(start_row, start_column, end_row, end_column)
# or
ws.merge_cells('A1:C3')
ws.unmerge_cells('A1:C3')
```

取消合并的范围必须是已经合并的范围，否则会报错。

## 插入图像

```python
from openpyxl import Workbook
from openpyxl.drawing.image import Image

wb = Workbook()
ws = wb.active
img = Image('logo.jpg') # 加载图像
ws.add_image(img, 'A1') # 插入图像至工作表
wb.save('simple_usage.xlsx')
```

我也随便插入了张图片，效果如下：

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202209060222333.png)

看了下源码， openpyxl 这个库本身没有提供处理图片的方法（包括调整大小这种也没有）。不过它是用的 Pillow 导入的图片，应该也能用 Pillow 操作，这里暂时不研究了。

## 折叠表格

```python
# 按列折叠
ws.column_dimensions.group('A','D', hidden=True)
# 按行折叠
ws.row_dimensions.group(1,10, hidden=True)
```

这个折叠好像跟我想的有点不太一样，我以为是隐藏那个功能但不是，结果长这样：

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202209060234278.png)

可以通过图上的 `+` 与 `-` 折叠与展开。

我查了一下它是用的这个功能：

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202209060242781.png)

# 优化模式

## 只读模式

```python
from openpyxl import load_workbook

wb = load_workbook(filename='sample.xlsx', read_only=True)

# 关闭工作簿
wb.close()
```

> 只读模式的工作簿必须使用 `close()` 方法显式关闭工作簿。

## 只写模式

```python
from openpyxl import Workbook

wb = Workbook(write_only=True)
```

> - 只写模式的工作簿必须使用 `wb.create_sheet()` 方法创建新的工作表，而不能用 `wb.active`  。
> - 只写模式工作簿的工作表中，只能使用 `append()` 方法添加行，不能使用 `cell()` 或 `iter_rows()` 操作单元格。
> - 只写工作簿只能保存一次。之后，每次将工作簿或 `append()` 保存到现有工作表的尝试都会抛出异常。

# 插入删除行列与移动表格

## 插入行列

```python
# 插入行
ws.insert_rows()
# 插入列
ws.insert_cols()
```

像插入删除移动这种操作需要记住的是下标这个问题。比如：

```python
from openpyxl import Workbook

wb = Workbook()
ws = wb.active

ws['A1'] = 1

# ws.insert_rows(0)
# ws.insert_cols(0)
# or
# ws.insert_rows(1)
# ws.insert_cols(1)

wb.save('insert_delete_move.xlsx')
```

以上代表不管是以 0 这个位置还是 1 这个位置结果都长这样：

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202209070009244.png)

所以说就是在你的第 n 行/列前插入行列。 Excel 的插入行或者插入列也是在选择的行列之前。

## 删除行列

```python
# 删除行
ws.delete_rows()
# 删除列
ws.delete_cols()
```

当然这个方法可以删多行或者多列，比如：

```python
from openpyxl import Workbook

wb = Workbook()
ws = wb.active

# 每个格子填自己的单元格坐标
for row in range(1, 11):
    for col in range(1, 11):
        ws.cell(row=row, column=col, value=f'{chr(64+col)}{row}')

# 从第 3 列开始，删除 4 列（即 3(C)、4(D)、5(E)、6(F) 列）
ws.delete_cols(3,4)
print(ws.max_column)

wb.save('insert_delete_move.xlsx')
```

像这样，打印结果是 6 ，表格结果就是：

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202209070023828.png)

可以看到 C、D、E、F 本来的数据被删了导致后面的列放到前面来了。

## 单元格组的移动

```python
ws.move_range('B4:C5', rows=-1, cols=1)
```

- 在行这个维度上，负数代表向上移动，正数代表向下移动。
- 在列这个维度上，负数代表向左移动，正数代表向右移动。

比如：

```python
from openpyxl import Workbook

wb = Workbook()
ws = wb.active

# 每个格子填自己的单元格坐标
for row in range(1, 11):
    for col in range(1, 11):
        ws.cell(row=row, column=col, value=f'{chr(64+col)}{row}')

# B4:C5 向右下移动一格
ws.move_range('B4:C5', rows=1, cols=1)

wb.save('insert_delete_move.xlsx')
```

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202209070030284.png)

看，`B4:C5` 向右下移动了一位吧。

如果你不希望你的公式乱掉，你可以：

```python
ws.move_range('B4:C5', rows=1, cols=1, translate=True)
```

# NumPy 和 Pandas 相关

**这里是非常重要的一部分**。因为用 openpyxl 的目的就是为了展示数据统计或分析之后的成果，在 Python 里处理数据一般都是用 Pandas 或者 NumPy 。

## Worksheet 添加 DataFrame

```python
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
import pandas as pd

wb = Workbook()
ws = wb.active

df = pd.DataFrame({'number': [1, 2], 'str': ['1', '2']})

for r in dataframe_to_rows(df, header=True, index=False):
    ws.append(r)

wb.save('numpy_pandas.xlsx')
```

结果如下：

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202209070111115.png)

其实就是利用 `dataframe_to_rows()` 这个方法逐行添加。

## Worksheet 转换为 DataFrame

```python
import pandas as pd

df = pd.DataFrame(ws.values)
```

从这里我们可以看出来，取值的操作是加了 `values` 的，所以如果是用 `cell` 单元格，也得是 `cell.value` 才行。

# 图表

**这里我认为是最重要的一部分。**这里可以说是用 Excel 的原因所在。

## 作图流程示例

```python
'''
	数据层
'''
from openpyxl import Workbook

wb = Workbook()
ws = wb.active

for i in range(10):
    ws.append([i])

'''
	展示层
'''
from openpyxl.chart import BarChart, Reference, Series

values = Reference(ws, min_col=1, min_row=1, max_col=1, max_row=10) # 选择作图数据
chart = BarChart() # 选择图表类型
chart.add_data(values) # 关联数据与图表
ws.add_chart(chart, 'B1') # 将图表添加到工作表并指定位置

wb.save('charts.xlsx')
```

完成后的效果是这样的：

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202209070227159.png)

那么图表的大小也是可以修改的，比如上面所用的 `BarChart` ，它实际上继承了 `ChartBase` ，这里展示一些 `ChartBase` 的源码。

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202209070157686.png)

那很显然，这个 `ChartBase` 里是有宽与高的属性的，要更深层的自定义图表的效果势必要研究这一块的源码。

## 图表类型

### 面积图 - Area Charts

#### 二维面积图 - 2D Area Charts

```python
from openpyxl import Workbook

from openpyxl.chart import (
    AreaChart,
    Reference,
    Series,
)

'''
	数据层
'''
wb = Workbook()
ws = wb.active

rows = [
    ['Number', '分支1', '分支2'],
    [2, 40, 30],
    [3, 40, 25],
    [4, 50, 30],
    [5, 30, 10],
    [6, 25, 5],
    [7, 50, 10],
]

for row in rows:
    ws.append(row)

'''
	展示层
'''
# 定义图表类型
chart = AreaChart()
# 图表标题
chart.title = '面积图'
# 图表风格
chart.style = 10
# X 轴标题
chart.x_axis.title = '数据'
# Y 轴标题
chart.y_axis.title = '占比'

# 选择标签数据
cats = Reference(ws, min_col=1, min_row=1, max_row=7)
# 选择数据
data = Reference(ws, min_col=2, min_row=1, max_col=3, max_row=7)
# 将数据与图表关联
chart.add_data(data, titles_from_data=True)
# 将 X 轴与标签关联
chart.set_categories(cats)

# 将图表添加进工作表
ws.add_chart(chart, "A10")
wb.save('charts.xlsx')
```

效果如下：

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202209070216454.png)

#### 三维面积图 - 3D Area Charts

```python
from openpyxl import Workbook

from openpyxl.chart import (
    AreaChart3D,
    Reference,
    Series,
)

'''
	数据层
'''
wb = Workbook()
ws = wb.active

rows = [
    ['Number', '分支1', '分支2'],
    [2, 40, 30],
    [3, 40, 25],
    [4, 50, 30],
    [5, 30, 10],
    [6, 25, 5],
    [7, 50, 10],
]

for row in rows:
    ws.append(row)

'''
	展示层
'''
# 定义图表类型
chart = AreaChart3D()
# 图表标题
chart.title = '面积图'
# 图表风格
chart.style = 10
# X 轴标题
chart.x_axis.title = '数据'
# Y 轴标题
chart.y_axis.title = '占比'

# 选择标签数据
cats = Reference(ws, min_col=1, min_row=1, max_row=7)
# 选择数据
data = Reference(ws, min_col=2, min_row=1, max_col=3, max_row=7)
# 将数据与图表关联
chart.add_data(data, titles_from_data=True)
# 将 X 轴与标签关联
chart.set_categories(cats)

# 将图表添加进工作表
ws.add_chart(chart, "A10")
wb.save('charts.xlsx')
```

效果如下：

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202209070223200.png)

### 柱形图与条形图 - Bar and Column Charts

#### 二维柱状图、二维条形图与堆积图

```python
from openpyxl import Workbook

from openpyxl.chart import (
    BarChart,
    Reference,
    Series,
)
'''
	数据层
'''
wb = Workbook()
ws = wb.active

rows = [
    ['Number', '分支1', '分支2'],
    [2, 40, 30],
    [3, 40, 25],
    [4, 50, 30],
    [5, 30, 10],
    [6, 25, 5],
    [7, 50, 10],
]

for row in rows:
    ws.append(row)
'''
	展示层
'''
# 定义图表类型
chart1 = BarChart()
# 图表标题
chart1.title = '柱状图'
# 图表风格
chart1.style = 10
# X 轴标题
chart1.x_axis.title = '数据'
# Y 轴标题
chart1.y_axis.title = '占比'
# 选择柱状图的类型
chart1.type = 'col'

# 选择标签数据
cats = Reference(ws, min_col=1, min_row=1, max_row=7)
# 选择数据
data = Reference(ws, min_col=2, min_row=1, max_col=3, max_row=7)
# 将数据与图表关联
chart1.add_data(data, titles_from_data=True)
# 将 X 轴与标签关联
chart1.set_categories(cats)

# 将图表添加进工作表
ws.add_chart(chart1, 'A10')

from copy import deepcopy

chart2 = deepcopy(chart1)
chart2.type = 'bar'
chart2.title = '条形图'

ws.add_chart(chart2, 'J10')

chart3 = deepcopy(chart1)
chart3.grouping = 'stacked'
chart3.overlap = 100
chart3.title = '柱状堆积图'

ws.add_chart(chart3, 'A27')

chart4 = deepcopy(chart2)
chart4.grouping = 'percentStacked'
chart4.overlap = 100
chart4.title = '百分比堆积图'

ws.add_chart(chart4, 'J27')

wb.save('charts.xlsx')
```

效果如下：

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202209072331682.png)

#### 三维柱状图

```python
from openpyxl import Workbook

from openpyxl.chart import (
    BarChart3D,
    Reference,
    Series,
)
'''
	数据层
'''
wb = Workbook()
ws = wb.active

rows = [
    ['Number', '分支1', '分支2'],
    [2, 40, 30],
    [3, 40, 25],
    [4, 50, 30],
    [5, 30, 10],
    [6, 25, 5],
    [7, 50, 10],
]

for row in rows:
    ws.append(row)
'''
	展示层
'''
# 定义图表类型
chart1 = BarChart3D()
# 图表标题
chart1.title = '三维柱状图'
# 图表风格
chart1.style = 10
# X 轴标题
chart1.x_axis.title = '数据'
# Y 轴标题
chart1.y_axis.title = '占比'
# 选择柱状图的类型
chart1.type = 'col'

# 选择标签数据
cats = Reference(ws, min_col=1, min_row=1, max_row=7)
# 选择数据
data = Reference(ws, min_col=2, min_row=1, max_col=3, max_row=7)
# 将数据与图表关联
chart1.add_data(data, titles_from_data=True)
# 将 X 轴与标签关联
chart1.set_categories(cats)

# 将图表添加进工作表
ws.add_chart(chart1, 'A10')

wb.save('charts.xlsx')
```

效果如下：

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202209072339092.png)

### 气泡图 - Bubble Charts

> 气泡图类似于散点图，但使用另一维数据来确定气泡的大小。图表可以包括多个系列。

```python
from openpyxl import Workbook

from openpyxl.chart import (
    BubbleChart,
    Reference,
    Series,
)
'''
	数据层
'''
wb = Workbook()
ws = wb.active

rows = [
    ['Number', '分支1', '分支2'],
    [2, 40, 30],
    [3, 40, 25],
    [4, 50, 30],
    [5, 30, 10],
    [6, 25, 5],
    [7, 50, 10],
]

for row in rows:
    ws.append(row)
'''
	展示层
'''
# 定义图表类型
chart = BubbleChart()
chart.style = 10
chart.title = '气泡图'

# 定义数据的坐标
xvalues = Reference(ws, min_col=2, min_row=2, max_row=7)
yvalues = Reference(ws, min_col=3, min_row=2, max_row=7)
# 定义泡泡大小
size = Reference(ws, min_col=1, min_row=2, max_row=7)
# 关联图表与数据
series = Series(values=yvalues, xvalues=xvalues, zvalues=size, title='Number')
chart.series.append(series)

# 将图表添加进工作表
ws.add_chart(chart, 'A10')

wb.save('charts.xlsx')
```

效果如下：

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202209080031958.png)

### 折线图 - Line Charts

```python
from openpyxl import Workbook

from openpyxl.chart import (
    LineChart,
    Reference,
)
'''
	数据层
'''
wb = Workbook()
ws = wb.active

rows = [
    ['Number', '分支1', '分支2'],
    [2, 40, 30],
    [3, 40, 25],
    [4, 50, 30],
    [5, 30, 10],
    [6, 25, 5],
    [7, 50, 10],
]

for row in rows:
    ws.append(row)
'''
	展示层
'''
# 定义图表类型
chart = LineChart()
chart.title = '折线图'
chart.style = 10
# X 轴标题
chart.x_axis.title = '数据'
# Y 轴标题
chart.y_axis.title = '占比'

# 选择标签数据
cats = Reference(ws, min_col=2, min_row=1, max_row=7)
# 选择数据
data = Reference(ws, min_col=2, min_row=1, max_col=3, max_row=7)
# 将数据与图表关联
chart.add_data(data, titles_from_data=True)
# 将 X 轴与标签关联
chart.set_categories(cats)

# 将图表添加进工作表
ws.add_chart(chart, 'A10')

wb.save('charts.xlsx')
```

效果如下：

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202209080056319.png)

### 散点图 - Scatter Charts



### 饼图 - Pie Charts



### 圆环图 - Doughnut Charts



### 雷达图 - Radar Charts



### 股票图 - Stock Charts



### 曲面图 - Surface charts



## 轴的使用



## 图表布局



## 图表样式



## 高级图表



## 图表页工作表

