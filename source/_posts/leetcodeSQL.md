---
title: 「SQL」 - 学习计划 
date: 2022-12-22 00:35:58
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
    Orders.CustomerId IS NULL;
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
    employee_id ASC;
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
    employee_id ASC;
```

CASE WHEN 也可以：

```mysql
SELECT
    employee_id,
    (CASE WHEN employee_id&1 AND LEFT(name, 1) != 'M' THEN salary ELSE 0 END) AS bonus
FROM
    Employees
ORDER BY
    employee_id ASC;
```

IF 也可以：

```mysql
SELECT
    employee_id,
    IF(employee_id&1 AND LEFT(name, 1) != 'M', salary, 0) AS bonus
FROM
    Employees
ORDER BY
    employee_id ASC;
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

## 字符串处理函数/正则

### 1667. 修复表中的名字

CONCAT, UPPER, LOWER, LEFT, SUBSTRING 这几个函数的用法。

```mysql
SELECT
    user_id,
    CONCAT(UPPER(LEFT(name,1)), LOWER(LOWER(SUBSTRING(name, 2)))) AS name
FROM
    Users
ORDER BY
    user_id ASC;
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

## 组合查询 & 指定选取

### 1965. 丢失信息的雇员

MySQL 居然没有 FULL OUTER JOIN ，所以我们可以使用 LEFT JOIN 加 RIGHT JOIN 这种方式实现。

```mysql
SELECT
    t1.employee_id
FROM
    Employees AS t1
    LEFT JOIN Salaries AS t2 ON t1.employee_id = t2.employee_id
WHERE
    t2.salary IS NULL
UNION
SELECT
    t2.employee_id
FROM
    Employees AS t1
    RIGHT JOIN Salaries AS t2 ON t1.employee_id = t2.employee_id
WHERE
    t1.name IS NULL
ORDER BY
    employee_id ASC;
```

可以用 GROUP BY 的方式实现，讲两张表的 `employee_id` UNION ALL（与 UNION 的区别就是不去重）起来，计数等于 1 的 `employee_id` 就是信息缺失的。

```mysql
SELECT
    t.employee_id
FROM
    (
        SELECT employee_id FROM Employees
        UNION ALL
        SELECT employee_id FROM Salaries
    ) AS t
GROUP BY
    t.employee_id
HAVING
    COUNT(t.employee_id) = 1
ORDER BY
    t.employee_id ASC;
```

### 1795. 每个产品在不同商店的价格

列转行问题，可以用 UNION ALL （这里不会重复，这里主要是比 UNION 快）。

```mysql
SELECT product_id, 'store1' AS store, store1 AS price FROM Products WHERE store1 IS NOT NULL
UNION ALL
SELECT product_id, 'store2' AS store, store2 AS price FROM Products WHERE store2 IS NOT NULL
UNION ALL
SELECT product_id, 'store3' AS store, store3 AS price FROM Products WHERE store3 IS NOT NULL;
```

### 608. 树节点

感觉很多 SQL 的题考察的都是对这门语言的熟练程度而不是逻辑，比如这道题的逻辑很简单，就是：

- `p_id` 为空， Root
- `p_id` 不为空 `id` 不是别人的 `p_id` ，Leaf
- `p_id` 不为空 `id` 是别人的 `p_id` ，Inner

CASE WHEN ELSE 语句：

```mysql
SELECT
    id,
    CASE
        WHEN p_id IS NULL THEN 'Root'
        WHEN id NOT IN (SELECT p_id FROM tree WHERE p_id IS NOT NULL) THEN 'Leaf'
        ELSE 'Inner'
    END AS Type
FROM
    tree
ORDER BY
    id;
```

IF 语句：

```mysql
SELECT
    id,
    IF(p_id IS NULL, 'Root', IF(id NOT IN (SELECT p_id FROM tree WHERE p_id IS NOT NULL), 'Leaf', 'Inner')) AS Type
FROM
    tree
ORDER BY
    id;
```

### 176. 第二高的薪水

除开最高薪水的最高薪水：

```mysql
SELECT
    MAX(salary) AS SecondHighestSalary
FROM
    Employee
WHERE
    salary != (
        SELECT MAX(salary) FROM Employee
    );
```

从大到小排序去重后排第二的薪水：

```mysql
SELECT (
    SELECT DISTINCT
        salary
    FROM
        Employee
    ORDER BY
        salary DESC
    LIMIT 1 OFFSET 1
) AS SecondHighestSalary;
```

## 合并

### 175. 组合两个表

简单的连接，注意要包含 `Person` 表的信息即可。

```mysql
SELECT
    t1.firstName,
    t1.lastName,
    t2.city,
    t2.state
FROM
    Person AS t1
    LEFT JOIN Address AS t2 ON t1.personId = t2.personId;
```

### 1581. 进店却未进行过交易的顾客

首先查满足条件的 `customer_id` ，然后再用 GROUP BY 进行统计。以构建一张中间表的形式在 FROM 里：

```mysql
SELECT
    customer_id,
    COUNT(*) AS count_no_trans
FROM
    (
        SELECT
            t1.customer_id
        FROM
            Visits AS t1
            LEFT JOIN Transactions AS t2 ON t1.visit_id = t2.visit_id
        WHERE
            t2.transaction_id IS NULL
    ) AS t
GROUP BY
    customer_id;
```

放在 WHERE 里：

```mysql
SELECT
    customer_id,
    COUNT(*) AS count_no_trans
FROM
    Visits
WHERE
    visit_id NOT IN(SELECT DISTINCT visit_id FROM Transactions)
GROUP BY
    customer_id;
```

### 1148. 文章浏览 I

DISTINCT 去重。

```mysql
SELECT DISTINCT
    author_id AS id
FROM
    Views
WHERE
    author_id = viewer_id
ORDER BY
    id ASC;
```

### 197. 上升的温度

`ADDDATE` 函数的使用。

```mysql
SELECT
    today.id
FROM
    Weather AS yesterday,
    Weather AS today
WHERE
    today.recordDate = ADDDATE(yesterday.recordDate, INTERVAL 1 DAY)
    AND today.temperature > yesterday.temperature;
```

或者用 `DATE_ADD` 函数。

```mysql
SELECT
    today.id
FROM
    Weather AS yesterday,
    Weather AS today
WHERE
    today.recordDate = DATE_ADD(yesterday.recordDate, INTERVAL 1 DAY)
    AND today.temperature > yesterday.temperature;
```

### 607. 销售员

子查询。

```mysql
SELECT
    name
FROM
    SalesPerson
WHERE
    sales_id NOT IN (
        SELECT
            o.sales_id
        FROM
            Orders AS o
            INNER JOIN Company AS c ON o.com_id = c.com_id
        WHERE
            c.name = 'RED'
    )
```

## 计算函数

### 1141. 查询近30天活跃用户数

这道题考察分组查询统计，首先 COUNT 里是可以用 `DISTINCT user_id` 来去重的，然后可以用 `DATE_SUB` 函数。

```mysql
SELECT
    activity_date AS day,
    COUNT(DISTINCT user_id) AS active_users
FROM
    Activity
WHERE
    activity_date > DATE_SUB("2019-07-27", INTERVAL 30 DAY)
    AND activity_date <= "2019-07-27"
GROUP BY
    activity_date;
```

当然也可以用 `DATE_ADD` 函数。

```mysql
SELECT
    activity_date AS day,
    COUNT(DISTINCT user_id) AS active_users
FROM
    Activity
WHERE
    activity_date > DATE_ADD("2019-07-27", INTERVAL -30 DAY)
    AND activity_date <= "2019-07-27"
GROUP BY
    activity_date;
```

### 1693. 每天的领导和合伙人

这道题应该是考察 GOURP BY 可以按多列分组。

```mysql
SELECT
    date_id,
    make_name,
    COUNT(DISTINCT lead_id) AS unique_leads,
    COUNT(DISTINCT partner_id) AS unique_partners
FROM
    DailySales
GROUP BY
    date_id, make_name;
```

### 1729. 求关注者的数量

其实就是对 `user_id` 进行 GROUP BY：

```mysql
SELECT
    user_id,
    COUNT(*) AS followers_count
FROM
    Followers
GROUP BY
    user_id
ORDER BY
    user_id ASC;
```

### 586. 订单最多的客户

首先找 `customer` 与其的订单数量，然后从中可以找到最多的订单数量，最后再查谁的订单数量等于最多的订单数量。嵌套子查询。

```mysql
SELECT
    MAX(order_count)
FROM(
    SELECT
        customer_number,
        COUNT(*) AS order_count
    FROM
        Orders
    GROUP BY
        customer_number
) AS t
```

也可以直接用 `customer` 的订单数量从大到小排序，只输出 1 个值，只查 `customer_number` 。

```mysql
SELECT
    customer_number
FROM
    Orders
GROUP BY
    customer_number
ORDER BY
    count(*) DESC
LIMIT 1
```

### 511. 游戏玩法分析 I

直接用 GROUP BY 加 MIN 就行了。

```mysql
SELECT
    player_id,
    MIN(event_date) AS first_login
FROM
    Activity
GROUP BY
    player_id;
```

### 1890. 2020年最后一次登录

这道题是 GROUP BY 加 MAX，比上道题多了个 WHERE 筛选年份。

```mysql
SELECT
    user_id,
    MAX(time_stamp) AS last_stamp
FROM
    Logins
WHERE
    time_stamp >= DATE('2020-01-01')
    AND time_stamp < DATE('2021-01-01')
GROUP BY
    user_id;
```

也可以这样

```mysql
SELECT
    user_id,
    MAX(time_stamp) AS last_stamp
FROM
    Logins
WHERE
    YEAR(time_stamp) = 2020
GROUP BY
    user_id;
```

### 1741. 查找每个员工花费的总时间

GROUP BY + SUM 函数，GROUP BY 是可以分组 2 列的。

```mysql
SELECT
    event_day AS day,
    emp_id,
    SUM(out_time-in_time) AS total_time
FROM
    Employees
GROUP BY
    event_day, emp_id
```

## 控制流

### 1393. 股票的资本损益

收益等于每笔卖出减去买入的和，也就等于总卖出减去总买入，所以可以用 SUM 和 IF 。

```mysql
SELECT
    stock_name,
    SUM(IF(operation='Sell', price, 0))-SUM(IF(operation='Buy', price, 0)) AS capital_gain_loss
FROM
    Stocks
GROUP BY
    stock_name;
```

当然可以把买入看作支出，卖出看作收入，那么两个 SUM IF 可以合并成一个。

```mysql
SELECT
    stock_name,
    SUM(IF(operation='Sell', price, -price))AS capital_gain_loss
FROM
    Stocks
GROUP BY
    stock_name;
```

### 1407. 排名靠前的旅行者

IFNULL 函数的使用，ORDER BY 的多列操作。

```mysql
SELECT
    t1.name,
    IFNULL(SUM(t2.distance), 0) AS travelled_distance
FROM
    Users AS t1
    LEFT JOIN Rides AS t2 ON t1.id = t2.user_id
GROUP BY
    t1.id
ORDER BY
    travelled_distance DESC, t1.name ASC;
```

### 1158. 市场分析 I

SUM IF YEAR 三个函数的使用，这题跟 Item 这张表没啥关系。

```mysql
SELECT
    t1.user_id AS buyer_id,
    t1.join_date,
    SUM(IF(YEAR(t2.order_date) = 2019, 1, 0)) AS orders_in_2019
FROM
    Users AS t1
    LEFT JOIN Orders AS t2 ON t1.user_id = t2.buyer_id
GROUP BY
    t1.user_id;
```

还有种，`JOIN table ON condition` 后面居然可以用 AND 再加条件，是我没想到的。

```mysql
SELECT
    t1.user_id AS buyer_id,
    t1.join_date,
    COUNT(t2.buyer_id) AS orders_in_2019
FROM
    Users AS t1
    LEFT JOIN Orders AS t2 ON t1.user_id = t2.buyer_id AND YEAR(t2.order_date) = 2019
GROUP BY
    t1.user_id;
```

而且在 ON 这里加跟在 WHERE 这里加不一样，如下图：

ON 添加条件限定 2019 年：

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202212211008315.png)

WHERE 添加条件限定 2019 年：

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202212211008555.png)

### 182. 查找重复的电子邮箱

GROUP BY + COUNT + HAVING 。

```mysql
SELECT
    Email
FROM
    Person
GROUP BY
    Email
HAVING
    COUNT(*) > 1;
```

### 1050. 合作过至少三次的演员和导演

依然是 GROUP BY + COUNT + HAVING 。

```mysql
SELECT
    actor_id,
    director_id
FROM
    ActorDirector
GROUP BY
    actor_id, director_id
HAVING
    COUNT(*)>=3;
```

### 1587. 银行账户概要 II

子查询 + GROUP BY + IFNULL + SUM 。

```mysql
SELECT
    name,
    balance
FROM(
    SELECT
        t1.name,
        IFNULL(SUM(t2.amount), 0) AS balance
    FROM
        Users AS t1
        LEFT JOIN Transactions AS t2 ON t1.account = t2.account
    GROUP BY
        t1.account
) AS t
WHERE
    balance > 10000;
```

当然直接用 GROUP BY + IFNULL + HAVING 也可以，因为 HAVING 针对的就是 GROUP BY 之后的结果。

```mysql
SELECT
    t1.name,
    IFNULL(SUM(t2.amount), 0) AS balance
FROM
    Users AS t1
    LEFT JOIN Transactions AS t2 ON t1.account = t2.account
GROUP BY
    t1.account
HAVING
    balance > 10000;
```

### 1084. 销售分析III

可以用子查询 + 连接的方式，不过需要去重。

```mysql
SELECT DISTINCT
    t1.product_id,
    t1.product_name
FROM
    Product AS t1
    INNER JOIN Sales AS t2 ON t1.product_id = t2.product_id
WHERE
    t1.product_id NOT IN (
        SELECT product_id
        FROM Sales
        WHERE YEAR(sale_date) != 2019 OR MONTH(sale_date) > 3
    );
```

两个子查询也是可以的。

```mysql
SELECT DISTINCT
    product_id,
    product_name
FROM
    Product
WHERE
    product_id NOT IN (
        SELECT product_id
        FROM Sales
        WHERE YEAR(sale_date) != 2019 OR MONTH(sale_date) > 3
    )
    AND product_id IN (
        SELECT DISTINCT product_id
        FROM Sales
    );
```

当然可以在一个子查询里用 MAX + MIN 两个函数做控制。

```mysql
SELECT DISTINCT
    product_id,
    product_name
FROM
    Product
WHERE
    product_id IN (
        SELECT product_id
        FROM Sales
        GROUP BY product_id
        HAVING MAX(sale_date) <= '2019-03-31' AND MIN(sale_date) >= '2019-01-01'
    );
```

