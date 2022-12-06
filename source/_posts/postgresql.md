---
title: PostgreSQL 15.1
date: 2022-12-04 10:24:46
categories: [DataScience, Database]
tags: [sql, postgresql]
---

> [PostgreSQL: Documentation: 15: PostgreSQL 15.1 Documentation](https://www.postgresql.org/docs/current/)
>
> [PostgreSQL 14.1 手册](http://www.postgres.cn/docs/14/index.html)
>
> 以翻译总结的形式过一遍官方文档，学习 PostgreSQL 15.1 版本。上面两个链接是参考。

# 前言

PostgreSQL 是一个基于 [POSTGRES, Version 4.2](https://dsf.berkeley.edu/postgres.html) 的对象关系数据库管理系统 （ORDBMS） ，由加州大学伯克利分校计算机科学系开发。支持大部分 SQL 标准并提供许多现代特性，比如：

- 复杂查询
- 外键
- 触发器
- 可更新视图
- 事务完整性
- 多版本并发控制

然后它可以用许多方法扩展，比如增加新的：

- 数据类型
- 函数
- 操作符
- 聚集函数
- 索引方法
- 过程语言

# 教程

## 架构基础

PostgreSQL 使用**客户端/服务器**的模型也就是 **C/S** 模型。这里主要讲服务器可以处理来自客户端的多个并发请求的逻辑，就是服务器有一个主进程称为 `postgres` ，它负责处理来自客户端的连接请求，为每个连接启动新的进程，本身不负责和客户传输数据。

这里的客户端指的是相对 PostgreSQL 数据库的客户端，对于一个以 PostgreSQL 开发的 Web 应用程序来讲，Web 后端就是这里的客户端， Web 前端不会与数据库通信。

## 创建数据库

```postgresql
createdb db_name
```

这里有个小坑，我用的 Windows11，如果不指定用户的话是不行的会报错，所以也许应该用：

```postgresql
.\createdb.exe -U username db_name
```

这里我没有将 `bin` 文件夹添加进环境变量，所以就在这里试一试。

能够看到创建数据库成功。

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202212041123625.png)

创建成功了在 `pgAdmin4` 和 `Navicat` 这样的工具里都可以看到。

# SQL 语言

# 服务端管理

# 客户端接口

# 服务端编程

# 参考

# 内部

# 附录

