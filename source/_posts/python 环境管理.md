---
title: Python 环境管理
date: 2024-09-03 23:34:22
categories: [ComputerScience, Python]
tags: [python]
---

# pip

## 镜像源

### 从镜像源安装包

```shell
pip install <包名> -i https://pypi.tuna.tsinghua.edu.cn/simple
```

# conda

## 版本管理

### 查看环境

```shell
conda env list
```

标了 `*` 号的代表当前环境。

### 创建环境

```shell
conda create -n <环境名>
```

指定 python 版本：

```shell
conda create -n <环境名> python=<版本号>
```

### 删除环境

```shell
conda remove -n <环境名> --all
```

### 激活环境

```shell
conda activate <环境名>
```

### 退出激活环境

```shell
conda deactivate
```

## 环境变量管理

### 查看环境变量

```shell
conda env config vars list
```

### 设置环境变量

```shell
conda env config vars set <变量名>=<变量值>
```

### 取消设置环境变量

```shell
conda env config vars unset <变量名>
```

# poetry

### 初始化项目

```shell
poetry init
```

## 依赖管理

### 安装包

### 安装依赖

```shell
poetry add <包名>
```

为 `dev` 环境安装包，这个依赖仅用于开发环境中，比如测试工具和代码质量检查工具。

```shell
poetry add <包名> --group dev
```

### 更新包

更新项目中的所有依赖到最新的兼容版本，并更新 `poetry.lock` 文件。

```shell
poetry update
```

### 删除包和依赖

```shell
poetry remove <包名>
```

### 查看包

```shell
poetry show
```

## 打包与发布

```shell
# 构建项目，生成 .tar.gz 和 .whl 格式的分发包。
poetry build

# 将构建的包发布到 PyPI 或其他配置的仓库。
poetry publish
```

## 版本管理

```shell
# 显示或更新项目的版本号。例如 poetry version patch 自动递增补丁版本号
poetry version
```

