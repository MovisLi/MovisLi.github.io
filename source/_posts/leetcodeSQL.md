---
title: 「SQL」 - 学习计划 
date: 2022-12-13 08:18:58
categories: [ComputerScience, Algorithm, LeetCode]
tags: [SQL]
---

# SQL 入门

## 选择

### 595. 大的国家

简单的条件筛选。

```mysql
SELECT
    name,
    population,
    area
FROM
    World
WHERE
    area >= 3000000
    OR population >= 25000000;
```

### 1757. 可回收且低脂的产品

也是简单的条件筛选。

```mysql
SELECT
    product_id
FROM
    Products
WHERE
    low_fats = 'Y'
    AND recyclable = 'Y';
```

### 584. 寻找用户推荐人

这道题注意 `!=` , `=` , `>` ,  `<` 这些会忽略 `NULL` ，让结果中没有 `NULL` 。

```mysql
SELECT
    name
FROM
    customer
WHERE
    referee_id != 2
    OR referee_id IS NULL;
```

### 183. 从不订购的客户

子查询：

```mysql
SELECT
    Name AS Customers
FROM
    Customers
WHERE
    Id NOT IN (
        SELECT DISTINCT CustomerId FROM Orders
    );
```

左连接：

```mysql
SELECT
    Name AS Customers
FROM
    Customers
    LEFT JOIN Orders ON Customers.Id = Orders.CustomerId
WHERE
    Orders.CustomerId IS NULL
```

