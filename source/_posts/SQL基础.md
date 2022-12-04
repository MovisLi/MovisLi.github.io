---
title: SQL 基础
date: 2022-08-04 23:34:22
categories: [DataScience, Database]
tags: [sql]
---

> 参考资料：
>
> [Learn SQL | Sololearn](https://www.sololearn.com/learning/1060)
>
> 《SQL 必知必会》（第 5 版）
>
> 本文主要指 MySQL 。

# 基础概念

## 数据库介绍

数据库是一个以便于访问、高效管理与更新的方式组织起来的数据集合。

数据库由存储相关信息的表组成。

表以行和列的格式存储并展示信息，就像 Excel 表格一样。

数据库通常包含多个表，每个表都因特定的目的而设计。

**表可拥有任意数量的行，但只能拥有一定数量的列。**

> MySQL Server 最多只允许 4096 个字段
>
> InnoDB 最多只能有 1017 个字段
>
> [MySQL :: MySQL 8.0 Reference Manual :: 8.4.7 Limits on Table Column Count and Row Size](https://dev.mysql.com/doc/refman/8.0/en/column-count-limit.html)

## 主键 - Primary Keys

主键是表中唯一标识表记录的字段。它关键的两个特点：

- 每行值唯一。
- 非空。

每张表最多只能有一个主键（也可以没有）。

## SQL - Structured Query Language

SQL 中文叫结构化查询语言。

它用于访问和操作数据库。

> SQL 是 ANSI（美国国家标准协会）标准，但 SQL 语言有不同的版本。
>
> 除了 SQL 标准之外，大多数 SQL 数据库程序都有自己的专有扩展，但它们都支持主要命令。
>
> MySQL 是其中的一种。

最主要的 4 大功能：**增删查改**。

## 基础 SQL 命令

### SHOW DATABASES

```mysql
SHOW DATABASES
```

列出了服务器管理的数据库。

### SHOW TABLES

```mysql
SHOW TABLES
```

列出当前数据库里所有表。

### SHOW COLUMNS FROM

```mysql
SHOW COLUMNS FROM 表名
```

列出表里的所有字段的详细信息。

如结果可能是

| Field | Type        | Null | Key  | Default | Extra          |
| ----- | ----------- | ---- | ---- | ------- | -------------- |
| id    | int(11)     | NO   | PRI  | NULL    | auto_increment |
| name  | varchar(30) | YES  |      | NULL    |                |

- Field - 列名，字段名。
- Type - 列数据类型。
- Null - 字段可否是空值。
- Key - 指示列是否被索引。
- Default - 分配给该列的默认值。
- Extra - 可能包含有关给定列的任何其他可用信息。

# 查询

### SELECT

```mysql
SELECT 列名 FROM 表名;
```

从一个表中查询指定的列。

```mysql
SELECT 列名1, 列名2 FROM 表名;
```

从一个表里查询多列的数据，列名之间用 `,` 分隔。

```mysql
SELECT * FROM 表名;
```

如果要查询所有列的数据，可以用 `*` 。

- 对于多条 SQL ，每条后需要添加 `;` 。
- SQL 语言不区分大小写。
- SQL 中忽略空格和多行。
  - 结合适当的间距和缩进，将命令分成逻辑行将使 SQL 语句更易于阅读和维护。

```mysql
SELECT 表名.列名 FROM 表名;
```

同样也可以指定查询某张表的某一列，在表名和列名之间加上 `.` ，当处理可能共享相同列名的多个表时，这种书写形式特别有用。

### DISTINCT

```mysql
SELECT DISTINCT
	列名
FROM
	表名;
```

DISTINCT 关键字可以达到去重的效果。

### LIMIT

```mysql
SELECT
	列名
FROM
	表名
LIMIT 数量;
```

LIMIT 关键字可以指定返回结果的数量。

```mysql
SELECT
	列名
FROM
	表名
OFFSET 数量
LIMIT 数量;
```

可以使用 OFFSET 关键字对所取结果中作筛选偏移，可以理解为 OFFSET 几条就是前几条不要。

### ORDER BY

```mysql
SELECT
	列名
FROM
	表名
ORDER BY
	列名;
```

ORDER BY 关键字可以将结果排序后再返回。

如果是数值型列，默认返回从小到大；如果是字符型列，默认返回字母顺序升序。

```mysql
SELECT
	*
FROM
	表名
ORDER BY
	列名1, 列名2;
```

ORDER BY 后可以跟两列，像这句一样。这样首先满足 `列名1` 的顺序，再满足 `列名2` 的顺序。

- ASC - 升序。

- DESC - 降序。

# 筛选

### WHERE

```mysql
SELECT
	表名
FROM
	列名
WHERE
	条件;
```

WHERE 语句可用于按条件筛选返回结果。

使用文本列时，用单引号 `'` 将语句中出现的任何文本括起来。

并且如果字符串内部有单引号，可以使用单引号来转义。例如：

我有一个查询条件 `值 = I'm` ，写成：

```mysql
WHERE
	列名 = 'I''m'
```

使用比较运算符和逻辑运算符来过滤要选择的数据。

| 比较运算符 | 描述                    |
| ---------- | ----------------------- |
| =          | 等于                    |
| !=         | 不等于                  |
| >          | 大于                    |
| <          | 小于                    |
| >=         | 大于等于                |
| <=         | 小于等于                |
| BETWEEN    | 和 AND 一起筛选一个范围 |

### BETWEEN

```mysql
SELECT
	列名
FROM
	表名
WHERE
	列名 BETWEEN 值1 AND 值2;
```

BETWEEN 运算符选择范围内的值。第一个值必须是下限，第二个值必须是上限。**这两个值都会被包括进去。**

### 逻辑运算符

| 逻辑运算符 | 描述                         |
| ---------- | ---------------------------- |
| AND        | 返回左右两个条件的交集       |
| OR         | 返回左右两个条件的并集       |
| IN         | 返回值在后面跟的括号里的结果 |
| NOT        | 对条件取反                   |

在上述逻辑运算符中，`AND` 优先级是要比 `OR` 高的。

所以 如果要同时用 `AND` 和 `OR` ，最好用括号 `()` 把 `OR` 的括起来以免出错。

### IN

```mysql
SELECT
	列名
FROM
	表名
WHERE
	列名 IN (值1， 值2);
```

比较一个列与多个值时，使用 IN 运算符。其效果等同于：

```mysql
SELECT
	列名
FROM
	表名
WHERE
	列名=值1 OR 列名=值2;
```

### AS

```mysql
SELECT
	列名 AS 新列名
FROM
	表名;
```

用 AS 关键字能够将列名的结果以新列名的标题返回。

### 算术运算符

```mysql
SELECT
	列名+值 AS 列名
FROM
	表名;
```

可以使用算术运算符将每列的值都做运算。

包括四则运算 `+-*/` 和括号，括号可用于强制操作优先于任何其他运算符，还用于提高代码的可读性。

| 操作符 | 说明 |
| ------ | ---- |
| +      | 加   |
| -      | 减   |
| *      | 乘   |
| /      | 除   |

### LIKE

```mysql
SELECT
	列名
FROM
	表名
WHERE
	列名 LIKE 搜索条件;
```

使用 `_` 匹配任何单个字符，使用 `%` 匹配任意数量的字符（包括零个字符）。

- 不要过度使用通配符。如果其他操作符能达到相同的目的，应该使用其他操作符。
- 在确实需要使用通配符时，也尽量不要把它们用在搜索模式的开始处。把通配符置于开始处，搜索起来是最慢的。
- 仔细注意通配符的位置。如果放错地方，可能不会返回想要的数据。

# 函数

> SELECT 语句为测试、检验函数和计算提供了很好的方法。
>
> 虽然 SELECT 通常用于从表中检索数据，但是省略了 FROM 子句后就是简单地访问和 处理表达式，例如 ：
>
> SELECT 3 * 2;将返回 6，
>
> SELECT Trim(' abc '); 将返回 abc，
>
> SELECT Curdate();使用 Curdate()函数返回当前日期和时间。
>
> 可以根据需要使用 SELECT 语句进行检验。

## 文本处理函数

### CONCAT

```mysql
SELECT
	CONCAT(列名1, ',', 列名2)
FROM
	表名;
```

CONCAT 函数用于连接两个或多个文本值并返回连接的字符串。

### RTRIM

```mysql
SELECT
	RTRIM(列名)
FROM
	表名;
```

RTRIM 函数用于去除所取列的值右边所有的空格。

### LTRIM

```mysql
SELECT
	LTRIM(列名)
FROM
	表名;
```

RTRIM 函数用于去除所取列的值**左边**所有的空格。

### TRIM

```mysql
SELECT
	TRIM(列名)
FROM
	表名;
```

RTRIM 函数用于去除所取列的值**左右两边**所有的空格。

### **UPPER**

```mysql
SELECT
	UPPER(列名)
FROM
	表名;
```

UPPER 函数将指定字符串中的所有字母转换为大写。 

### LOWER

```mysql
SELECT
	LOWER(列名)
FROM
	表名;
```

LOWER 函数将字符串转换为小写。

> 如果字符串中有不是字母的字符，这个函数对它们不起作用。

### LEFT

```mysql
SELECT
	LEFT(列名, 数字)
FROM
	表名
```

LEFT 函数将取到该列中值的前 n 个字符。

### RIGHT

```mysql
SELECT
	RIGHT(列名, 数字)
FROM
	表名
```

RIGHT 函数将取到该列中值的后 n 个字符。

### SUBSTRING

```mysql
SELECT
	SUBSTRING(列名 FROM 数字1 FOR 数字2)
FROM
	表名
```

从 `数字1` 开始（字符串下标从 1 开始而不是 0），取 `数字2` 个数字。

### SOUNDEX

```mysql
SELECT
	列名
FROM
	表名
WHERE
	SOUNDEX(列名) = SOUNDEX(字符串)
```

返回与字符串发音相同的列名里的值。

## 数值处理函数

### SQRT

```mysql
SELECT
	SQRT(列名)
FROM
	表名;
```

SQRT 函数返回该列中给定值的平方根。

### ABS

```mysql
SELECT
	ABS(列名)
FROM
	表名;
```

ABS 函数返回该列中给定值的绝对值。

### SIN、COS、TAN、PI、EXP

| 函数 | 说明                       |
| ---- | -------------------------- |
| SIN  | 返回一个角度的正弦         |
| COS  | 返回一个角度的余弦         |
| TAN  | 返回一个角度的正切         |
| PI   | 返回圆周率 $\pi$ 的值      |
| EXP  | 返回一个数的指数值 $e^{x}$ |

## 聚集函数

**这种函数只会返回一个值，此所谓聚集的意思。**

### AVG

```mysql
SELECT
	AVG(列名)
FROM
	表名;
```

AVG 函数返回该列的平均值。

> AVG 函数忽略列值为 NULL 的行。

### COUNT

```mysql
SELECT
	COUNT(列名)
FROM
	表名;
```

COUNT 函数返回该列的行数。

> 如果指定列名，则 COUNT 函数会忽略指定列的值为 NULL 的行，但 如果 COUNT 函数中用的是星号 `*` ，则不忽略。DISTINCT 不能用于 COUNT(*) 。

### MAX

```mysql
SELECT
	MAX(列名)
FROM
	表名;
```

MAX 函数返回该列的最大值。

> MAX 函数忽略列值为 NULL 的行。

### MIN

```mysql
SELECT
	MIN(列名)
FROM
	表名;
```

MIN 函数返回该列的最小值。

> MIN 函数忽略列值为 NULL 的行。

### SUM

```mysql
SELECT
	SUM(列名)
FROM
	表名;
```

SUM 函数返回该列的和。

> SUM 函数忽略列值为 NULL 的行。

# 分组数据

使用分组可以将数据分为多个逻辑组，对每个组进行聚集计算。

## GROUP BY

```mysql
SELECT
	列名, 聚集函数(*)
FROM
	表名
GROUP BY
	列名;
```

GROUP BY 子句指示 DBMS分组数据，然后对每个组而不是整 个结果集进行聚集。

## HAVING

# 子查询

子查询是另一个查询中的查询。它的末尾没有分号。

> 作为子查询的 SELECT 语句只能查询单个列。企图检索多个列将返回 错误。

```mysql
SELECT
	(子查询)
FROM
	表名;
```

子查询作为查询结果。

```mysql
SELECT
	列名
FROM
	表名
WHERE
	列名 in (子查询);
```

子查询作为筛选条件。

# 连接

**在引用的列可能出现歧义时，必须使用完全限定列名（用一个句点分隔表名和列名， `表名.列名` 的形式）。如果引用一个没有用表名限制的具有歧义的列名，数据库会报错。**

> DBMS在运行时关联指定的每个表，以处理联结。这种处理可能非常耗费资源，因此应该注意，不要联结不必要的表。联结的表越多，性能下降越厉害。

## 等值连接

```mysql
SELECT
	列名1, 列名2
FROM
	表名1, 表名2
WHERE
	表名1.列名1 = 表名2.列名2;
```

这样会返回两个表的笛卡尔积，使用 WHERE 语句。

## 内连接

```mysql
SELECT
	列名1, 列名2
FROM
	表名1
	INNER JOIN 表名2 ON 表名1.列名1=表名2.列名2;
```

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202208081810684.png)

## 自连接

```mysql
SELECT
	列名
FROM
	表名 AS 别名1, 表名 AS 别名2
WHERE
	别名1.列名 = 别名2.列名;
```

或

```mysql
SELECT
	列名
FROM
	表名 AS 别名1
	INNER JOIN 表名 AS 别名2 ON 别名1.列名=别名2.列名;
```

> 自联结通常作为外部语句，用来替代从相同表中检索数据的使用子查询语句。
>
> 虽然最终的结果是相同的，但许多DBMS处理联结远比处理子查询快得多。
>
> 应该试一下两种方法，以确定哪一种的性能更好。

## 自然连接

```mysql
SELECT
	列名
FROM
	表名1
	NATURAL JOIN 表名2;
```

自然连接是一种特殊的内连接，它不需要指定连接条件，重复的列会被去掉。

## 外连接

许多联结将一个表中的行与另一个表中的行相关联，但有时候需要包含没有关联行的那些行。

### 左连接

```mysql
SELECT
	列名1, 列名2
FROM
	左表名
	LEFT OUTER JOIN 右表名 ON 左表名.列名1 = 右表名.列名2;
```

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202208081845615.png)

### 右连接

```mysql
SELECT
	列名1, 列名2
FROM
	左表名
	RIGHT OUTER JOIN 右表名 ON 左表名.列名1 = 右表名.列名2;
```

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202208081847770.png)

# 组合查询

> 多数 SQL查询只包含从一个或多个表中返回数据的单条 SELECT 语句。但是，SQL也允许执行多个查询（多条  SELECT 语句），并将结果作为一个查询结果集返回。这些组合查询通常称为并（ union ）或复合查询（ compound query ）。

```mysql
查询1
UNION
查询2;
```

UNION 返回的结果已经是去重了的，如果不需要去重，用 UNION ALL ，并且 UNION ALL 也更快

```mysql
查询1
UNION ALL
查询2;
```

- 在用 UNION 组合查询时，只能使用一条 **ORDER BY** 子句，它必须位于最后一条 SELECT 语句之后。

- 如果列在所有查询中不完全匹配，可以使用 NULL（或任何其他）值（**说明组合查询要求列的数量完全匹配，最终列名会按第一个查询的列名来展示结果**），例如：

```mysql
SELECT
	列名1, 列名2
FROM
	表名1
UNION
SELECT
	列名1, NULL
FROM
	表名2;
```

# 插入数据

## 插入完整的行

```mysql
INSERT INTO
	表名
VALUES
	(值1, 值2);
```

这种写法能够实现插入功能，但是依赖列的顺序，很不安全，不推荐使用。

推荐使用下面的写法：

```mysql
INSERT INTO
	表名(列名1, 列名2)
VALUES
	(值1, 值2);
```

## 插入部分行

其实就是使用写列名加值的写法。但是省略的列必须满足以下两个条件之一：

- 该列定义允许 NULL 值。
- 该列在表定义种给出默认值。

## 插入检索出的数据

```mysql
INSERT INTO
	表名1(列名)
SELECT
	列名
FROM
	表名2;
```

一般用于表的迁移或合并。**并不要求插入的列名和查询的列名一致，DBMS 使用列的位置来插入。**

## 复制表

当然对于迁移到全新的表，也可以选择复制表。

```mysql
CREATE TABLE
	新表名 AS
SELECT
	*
FROM
	旧表名;
```

# 修改数据

```mysql
UPDATE
	表名
SET
	列名1=值名1, 列名2=值名2
WHERE
	条件;
```

**如果不加 WHERE 条件的话，就会更新所有行。**

# 删除数据

```mysql
DELETE FROM
	表名
WHERE
	条件;
```

与更新数据一样，**如果不加 WHERE 条件的话，就会删除所有行。**

# 创建表

```mysql
CREATE TABLE
	表名
(
	列名1 数据类型 NOT NULL DEFAULT 默认值,
	列名2 数据类型 NULL,
    列名3 INT NOT NULL AUTO_INCREMENT,
    PRIMARY KET(列名)
);
```

## 数据类型

常用数据类型如下：

| 列种类     | 数据类型     | 说明                                                         |
| ---------- | ------------ | ------------------------------------------------------------ |
| 数值型     | INT          | 有符号或无符号的正常大小的整数。                             |
|            | FLOAT(M, D)  | 有符号的浮点数。可以选择定义显示长度 (M) 和小数位数 (D)。    |
|            | DOUBLE(M, D) | 有符号的双精度浮点数。可以选择定义显示长度 (M) 和小数位数 (D)。 |
| 日期与时间 | DATE         | YYYY-MM-DD 格式的日期。                                      |
|            | DATETIME     | YYYY-MM-DD HH:MM:SS 格式的日期和时间组合。                   |
|            | TIMESTAMP    | 时间戳，从 1970 年 1 月 1 日午夜开始计算。                   |
|            | TIME         | 以 HH:MM:SS 格式存储时间。                                   |
| 字符型     | CHAR(M)      | 定长字符串。大小在括号中指定。最大 255 字节。                |
|            | VARCHAR(M)   | 变长字符串。最大尺寸在括号中指定。                           |
|            | BLOB         | “二进制大对象”，用于存储大量二进制数据，例如图像或其他类型的文件。 |
|            | TEXT         | 大量的文本数据。                                             |

## 约束

常用约束如下：

| 约束               | 说明                                                         |
| ------------------ | ------------------------------------------------------------ |
| NOT NULL           | 指示列不能包含任何 NULL 值。                                 |
| UNIQUE             | 不允许在列中插入重复值。 UNIQUE 约束保持表中列的唯一性。一个表中可以使用多个 UNIQUE 列。 |
| PRIMARY KEY        | 强制表接受特定列的唯一数据，并且此约束创建唯一索引以更快地访问表。 |
| CHECK              | 根据逻辑表达式确定值是否有效。                               |
| DEFAULT            | 在向表中插入数据时，如果没有为列提供值，则该列将获取设置为 DEFAULT 的值。 |
| **AUTO_INCREMENT** | 自增。                                                       |

# 更新表

## 增加列

```mysql
ALTER TABLE
	表名
ADD COLUMN
	列名 数据类型;
```

## 删除列

```mysql
ALTER TABLE
	表名
DROP COLUMN
	列名;
```

## 重命名

```mysql
ALTER TABLE
	表名
RENAME
	旧列名
TO
	新列名;
```

重命名列。

```mysql
RENAME TABLE
	旧表名
TO
	新表名;
```

重命名表。

# 删除表

```mysql
DROP TABLE
	表名;
```

# 视图

在 SQL 中，视图是一个基于 SQL 语句结果集的虚拟表。每次访问视图都会重新查询，因此视图其实可能会导致性能下降得特别厉害。它的优点在于封装了底层查询，同时也可用作权限的管理。

## 创建视图

```mysql
CREATE VIEW
	视图名 AS
查询;
```

## 更新视图

```mysql
CREATE OR REPLACE VIEW
	视图名 AS
查询;
```

不太推荐，直接删掉重新创建就行了。

## 删除视图

```mysql
DROP VIEW
	视图名;
```

