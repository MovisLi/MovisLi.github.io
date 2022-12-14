---
title: 「SQL」 - 学习计划 
date: 2022-12-15 02:04:58
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

## 排序 & 修改

### 1873. 计算特殊奖金

两次查询结果合并， `UNION+WHERE` ：

```mysql
SELECT
    employee_id,
    salary AS bonus
FROM
    Employees
WHERE
    employee_id & 1
    AND name NOT LIKE 'M%'
UNION
SELECT
    employee_id,
    0 AS bonus
FROM
    Employees
WHERE
    employee_id & 1 != 1
    OR name LIKE 'M%'
ORDER BY
    employee_id ASC
```

 MySQL 中有个函数 LEFT 也可以：

```mysql
SELECT
    employee_id,
    salary AS bonus
FROM
    Employees
WHERE
    employee_id & 1
    AND LEFT(name, 1) != 'M'
UNION
SELECT
    employee_id,
    0 AS bonus
FROM
    Employees
WHERE
    employee_id & 1 != 1
    OR LEFT(name, 1) = 'M'
ORDER BY
    employee_id ASC
```

CASE WHEN 也可以：

```mysql
SELECT
    employee_id,
    (CASE WHEN employee_id&1 AND LEFT(name, 1) != 'M' THEN salary ELSE 0 END) AS bonus
FROM
    Employees
ORDER BY
    employee_id ASC
```

IF 也可以：

```mysql
SELECT
    employee_id,
    IF(employee_id&1 AND LEFT(name, 1) != 'M', salary, 0) AS bonus
FROM
    Employees
ORDER BY
    employee_id ASC
```

### 627. 变更性别

IF 函数：

```mysql
UPDATE
    Salary
SET
    sex=IF(sex='m', 'f', 'm');
```

CASE WHEN 语句：

```mysql
UPDATE
    Salary
SET
    sex=(CASE sex WHEN 'm' THEN 'f' ELSE 'm' END);
```

还有个 ASCII 码于字符转换的方法：

```mysql
UPDATE
    Salary
SET
    sex=CHAR(109+102-ASCII(sex));
```

### 196. 删除重复的电子邮箱

自连接：

```mysql
DELETE 
    p1
FROM
    Person AS p1,Person AS p2
WHERE
    p1.id > p2.id
    AND p1.email = p2.email;
```

### 1667. 修复表中的名字

CONCAT, UPPER, LOWER, LEFT, SUBSTRING 这几个函数的用法。

```mysql
SELECT
    user_id,
    CONCAT(UPPER(LEFT(name,1)), LOWER(LOWER(SUBSTRING(name, 2)))) AS name
FROM
    Users
ORDER BY
    user_id ASC
```

### 1484. 按日期分组销售产品

刷 SQL 我怎么感觉算是刷 SQL 函数的使用。这道题是讲 GROUP_CONCAT 这个函数：

```mysql
GROUP_CONCAT(DISTINCT expression
    ORDER BY expression
    SEPARATOR sep);
```

还有就是 COUNT 这个函数括号里可以有 DISTINCT 。

```mysql
SELECT
    sell_date,
    COUNT(DISTINCT product) AS num_sold,
    GROUP_CONCAT(DISTINCT product ORDER BY product ASC SEPARATOR ',') AS products
FROM
    Activities
GROUP BY
    sell_date
ORDER BY
    sell_date ASC;
```

### 1527. 患某种疾病的患者

LIKE 的用法，两种情况：

- 第一个元素以 `DIAB1` 开头，对应 `'DIAB1%'` 。
- 非第一个元素以 `DIAB1` 开头，对应 `'% DIAB1%'` 。

```mysql
SELECT
    patient_id,
    patient_name,
    conditions
FROM
    Patients
WHERE
    conditions LIKE 'DIAB1%'
    OR conditions LIKE '% DIAB1%';
```

