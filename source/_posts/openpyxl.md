---
title: openpyxl 包的学习
date: 2022-09-03 17:47:23
categories: [DataScience, Excel]
tags: [python, excel]
---

# 打开保存

## 保存

比如保存一个叫 `test.xlsx` 的文件在根目录。

```python
import openpyxl

# 实例工作簿对象
wb = openpyxl.Workbook()

# 返回激活的工作表
ws = wb.active

# 保存
wb.save('test.xlsx')
```

## 打开

比如打开刚刚保存的 `test.xlsx` 文件。

```python
import openpyxl

# 加载工作簿对象
wb = openpyxl.load_workbook(r'test.xlsx')

# 返回激活的工作表
ws = wb.active
```

**这里要注意的是要加载的工作簿文件必须是没有用 Excel 打开的否则会报错。**

# 操作工作表

## 查看工作表名

```python
import openpyxl

wb = openpyxl.Workbook()
ws1 = wb.active

# 查看工作表名
print(ws1.title)

# 查看工作簿所有工作表名
print(wb.sheetnames)
```

## 创建工作表

位置传参，第一个参数为工作表名，第二个参数为存放的 index 。

```python
wb.create_sheet('Sheet2', 1)
wb.create_sheet('Sheet3', 2)
```

## 取工作表

```python
ws2 = wb['Sheet2']
```

## 移动工作表

位置传参，第一个参数为工作表对象，第二个参数为移动的位数（正数向后移动，负数向前移动）

```python
wb.move_sheet(ws2, -2)
```

## 删除工作表

```python
del wb['Sheet']
```

# 访问单元格



# 操作单元格



# 使用公式



# 设置样式



# 过滤和排序



# 插入图表



# 只读只写

