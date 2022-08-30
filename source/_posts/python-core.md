---
title: Python Core
date: 2022-08-04 23:34:22
categories: [ComputerScience, Python]
tags: [python]
---

# 模块

## 模块编写规范

```python
#!/usr/bin/python # 通常只在 unix 环境有效，作用是指定解释器路径，然后可以直接使用脚本名来执行，不需要在前面调用解释器
# coding: utf-8

"""
模块文档描述
"""

import module # 导入模块

global_var = object() # 定义全局变量；如果不是必须，最好使用局部变量，这样可以提高代码的维护性，同时节省内存提高性能

class Name: # 定义类
    """
    类的注释
    """
    pass


def func(): # 定义函数
    """
    函数注释
    """
    pass


if __name__ == '__main__': # 主程序，在被当作脚本执行时，执行下面的代码
    func()
```

