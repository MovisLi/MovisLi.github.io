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

## 软件目录规范

```
│  README.md
│  requirements.txt
│
├─api
├─bin
│      run.py
│      setup.py
│
├─conf
│      settings.py
│
├─core
├─db
│      db_handle.py
│
├─lib
└─log
```

- bin - 放程序可执行文件夹，也可把执行文件放根目录下。
  - run.py - 启动文件。
  - setup.py - 安装、部署、打包的脚本。
- conf - 放用户自定义配置的文件夹。
  - settings.py - 用户自定义配置。
- lib - 程序常用的模块集合文件夹，包括模块、包。
- core - 核心代码逻辑文件夹。
- log - 存放项目日志。
- db - 数据库相关文件夹。
- api - 用户接口文件夹。
- requirement.txt - 第三方模块依赖文件。
- README.md - 项目简介。
