---
title: FastAPI
date: 2023-05-12 05:57:22
categories: [ComputerScience, Python, Backend]
tags: [python, fastapi]
---

# 基础

## WEB 基础

| 请求方法 | 说明                                                  |
| -------- | ----------------------------------------------------- |
| GET      | 请求 URL 的网页信息，并返回实体数据。一般用于查询数据 |
| POST     | 向 URL 提交数据进行处理请求。比如提交表单或是上传文件 |
| PUT      | 向 URL 上传数据内容                                   |
| DELETE   | 向 URL 发送删除资源请求                               |
| HEAD     | 与 GET 请求类似，只返回响应头                         |
| CONNECT  | HTTP/1.1 预留                                         |
| OPTIONS  | 获取服务器特定资源的 HTML 请求方法                    |
| TRACE    | 回复并显示服务器收到的请求                            |
| PATCH    | 对 PUT 方法的补充，用来对已知资源进行局部更新         |

URL （ Uniform Resource Locator，资源定位符）代表一个网站上资源的详细地址。完整的 URL 由以下部分依序组成：

1. 协议，如 HTTPS 协议。
2. 主机名，可能是域名，如 `www.google.com` ，也可能是 IP 地址加端口号，如 `192.168.1.1:80` 。
3. 资源相对路径，指网站上资源相对的地址，可能还带有参数。

HTTP 请求与响应详细实现步骤如下：

1. 客户端 TCP 连接到 Web 服务器。
2. 客户端发送 HTTP 请求。
3. Web 服务器接受请求并返回 HTTP 响应。
4. 释放 TCP 连接。
5. 客户端解析响应数据。

HTTP 请求大致组成如下：

- 请求行（请求方法，URL，协议版本）
- 请求头
- 空行
- 请求数据

HTTP 响应大致组成如下：

- 状态行（协议版本，状态码，状态码描述）
- 响应头
- 空行
- 响应收据

| 状态码 | 说明                                                 |
| ------ | ---------------------------------------------------- |
| 100    | 不带有响应对象的业务数据，需要请求后执行后续相关操作 |
| 200    | 成功                                                 |
| 201    | 也是成功，一般用于表示在数据库中创建了一条新的记录   |
| 204    | 提示客户端服务端成功处理，但没有返回内容             |
| 300    | 重定向                                               |
| 400    | 客户端异常                                           |
| 500    | 服务器异常                                           |

## FastAPI 框架组成

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202305120725715.png)

FastAPI 功能建立在 Python 类型提示（3.7, 3.9, 3.10 可能写法不一样）、Pydantic 框架、 Starlette 框架基础上。

### Python 类型提示

#### 使用方式

```python
def func(data: int = 0) -> str:
    res: str = str(data)
    return res
```

变量或参数后面使用冒号指定类型，不过运行时没用，也是防君子不防小人的一种体现。

#### 基础数据类型

- int - 整型
- float - 浮点型
- str - 字符串型
- bool - 逻辑型
- bytes - 字节型

#### 泛型

- list - 列表泛型

- tuple - 元组泛型
- set - 集合泛型
- dict - 字典泛型

```python
from typing import List, Optional
from datetime import datetime

# 3.7
my_list: List[str] = []
timestamp: Optional[datetime] = None

# 3.9
my_list: list[str] = []
timestamp: Optional[datetime] = None

# 3.10
my_list: list[str] = []
timestamp: datetime | None = None
```

#### 自定义类

```python
class Person:
    def __init__(self, name: str):
        self.name = name


# output: movis
print(Person('movis').name)
```

### Pydantic 框架

[Pydantic](https://docs.pydantic.dev/latest/) 是一套基于 Python 类型提示的数据模型定义及验证的框架。

这个框架在**运行时**强制执行类型提示。

### Starlette 框架

Starlette 是一个轻量级高性能的异步服务网关接口框架（ASGI）。

# 请求

# 响应

# 异常处理

# 中间件技术

# 依赖注入

# 数据库操作

# 安全机制

# 异步

# 架构

# 测试

# 部署

