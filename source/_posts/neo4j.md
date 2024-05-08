---
title: Cypher 与 neo4j
date: 2024-04-16 12:27:11
categories: [DataScience, Database]
tags: [graph, python, neo4j]
---

[toc]

# 基础

## 核心概念

Cypher 是一种专为图设计的查询语言。图由节点和关系构成。也被称为 CQL (Cypher Query Language) 。

### 节点

**节点**在 Cypher 中由括号 `()` 表示，节点的**标签**（类似一个类的类名）由冒号 `:` 表示。

```CQL
(:Loan)
```

这就是一个类型为 Loan 的节点。

> neo4j 推荐节点的标签使用**大驼峰命名**（类似 Python 的类命名）。

节点的**属性**以类似 JSON 的语法指定，不过不一定需要双引号，单引号也可以。**属性名称区分大小写。**

```CQL
(:Loan {loanId: 123, createdAt: '2024-04-10'})
```

这就是一个带有属性的 Loan 节点。

> neo4j 推荐节点和关系的属性使用**小驼峰命名**（类似 Java 的变量名）。

### 关系

节点之间的**关系**由两个破折号表示 `--` ，关系的**方向**由大于符号 `>` 或小于符号 `<` 表示。关系的**类型**用方括号 `[]` 与冒号 `:` 表示。

```CQL
[:USE_IN]
```

> neo4j 推荐关系的类型使用**大写加下划线命名**（类似 Python 的常量命名）。

将节点与关系结合表示：

```CQL
(p:Person)-[:APPLIED]->[l:Loan {loanId: 123, createdAt: '2024-04-10'}]
```

这里表示了一个 Person （ p 和 l 都不用管，这个是变量，相当于在这段 CQL 语句里的别名而已，节点本身跟这个没关系，有点像 SQL 查询时所用的 `AS` ）APPLIED 了一笔 Loan ，这笔 Loan 的 loanId 为 123， createdAt 为 2024-04-10 。

## 数据类型

### 数据类型

neo4j 中有如下数据类型：

- String 字符串

- Integer 整数
  - 在 neo4j 里其实是 Long 。

- Float 浮点数

- Boolean 布尔值

- Date/Datetime 日期/时间
  - 日期类型可以用 `x.year, x.month, x.day` 这样的方式提取信息。

- Point 点

- Lists of values 列表
  - 在 neo4j 里其实是 StringArray 。
  - 列表中的值必须具有相同的数据类型。

### 类型转换函数

| 函数           | 描述                                                        |
| :------------- | :---------------------------------------------------------- |
| `toBoolean(s)` | 将字符串转换为布尔值                                        |
| `toFloat(s)`   | 将字符串转换为浮点数                                        |
| `toInteger(s)` | 将字符串转换为整数                                          |
| `toString(v)`  | 将值转换为字符串                                            |
| `date(s)`      | 将字符串转换为日期                                          |
| `datetime(s)`  | 将字符串转换为时间                                          |
| split(s, x)    | 将字符串拆分为列表，其中 s 表示要拆分的字符串，x 表示分隔符 |

## 读取

### MATCH RETURN 简单查询

使用 `MATCH` 关键字匹配节点，可以在节点里面用 JSON 的方式做条件筛选。准确来讲更像是 `MATCH ... RETURN` 一起实现了 SQL 里类似 `SELECT` 的效果。

```CQL
// 查询在 2024-04-10 创建的 Loan 的 loanId 属性
MATCH (l:Loan {createdAt: '2024-04-10'})
RETURN l.loanId
```

除了匹配节点，也可以匹配关系，**在匹配关系的时候，不一定需要指定方向**。

```CQL
// 查询有人申请的 Loan 的 loanId 属性
MATCH (p:Person)-[:APPLIED]->(l:Loan)
RETURN l.loanId

// 查询所有指向其他节点的节点
MATCH (l)-->() RETURN l
```

可以使用 `;` 来执行多段查询。

### OPTIONAL MATCH 选择性匹配

类似 MATCH ，但是对模式的缺失部分使用空值填充（ MATCH 是直接不匹配）。

### WHERE 条件过滤

#### AND 和 OR

使用 `WHERE` 结合 `AND` ， `OR` 关键字可以进行更复杂的查询。

```CQL
// 查询在 2024-04-10 创建的 Loan 的 loanId 属性
MATCH (l:Loan {createdAt: '2024-04-10'})
RETURN l.loanId

// 查询在 2024-04-10 创建的 Loan 的 loanId 属性
MATCH (l:Loan)
WHERE l.createdAt = '2024-04-10'
RETURN l.loanId

// 查询在 2024-04-10 创建的 Loan 的 loanId 属性
MATCH (l)
WHERE l:Loan AND l.createdAt = '2024-04-10'
RETURN l.loanId
```

上面三段语句是等价的。

#### > 和 < 和 >= 和 <= 和 <>

在 `WHERE` 里还能使用类似 Python 一样的使用范围过滤。

```CQL
// 查询 loanId 大于 122 且小于等于 124 的 Loan 节点
MATCH (l:Loan)
WHERE l.loanId > 122 AND l.loanId <= 124
RETURN l

// 查询 loanId 大于 122 且小于等于 124 的 Loan 节点
MATCH (l:Loan)
WHERE 122 < l.loanId <= 124
RETURN l

// 查询 loanId 不等于 122 的节点
MATCH (l:Loan)
WHERE l.loanId <> 122
RETURN l
```

上面两段语句是等价的。

#### IS NULL 和 IS NOT NULL

属性的值不能为空，如果属性的值为空等价于没有这个属性。这个语法也可以说是查询属性是否存在。

```CQL
// 查询没有 createdAt 属性 Loan 节点
MATCH (l:Loan)
WHERE l.createdAt IS NULL
RETURN l

// 查询有 createdAt 属性的 Loan 节点
MATCH (l:Loan)
WHERE l.createdAt IS NOT NULL
RETURN l
```

#### 字符串函数 STARTS WITH 和 END WITH 和 CONTAINS

`toLower()` 函数和 `toUpper()` 函数能够将查询转换为小写/大写，但是如果查询过程中**使用了大小写转换，索引会失效**。

```CQL
// 返回人名以 mo 开头的 Person 节点
MATCH (p:Person)
WHERE toLower(p.name) STARTS WITH 'mo'
RETURN p
```

#### 使用列表进行过滤 IN

有点类似 SQL 里的 IN 。

```CQL
MATCH (p:Person)
WHERE p.born IN [1965, 1966, 1970]
RETURN p.name
```

#### 查询标签是否存在

```CQL
MATCH (l:Loan)
WHERE l:ApprovedLoan
RETURN l
```

### RETURN 返回结果

#### 排序 ORDER BY

这个和 SQL 的用法一样，ORDER BY 加上 ASC（升序） 或者 DESC（降序） 表示去控制返回顺序，如果不写默认就是升序。

#### LIMIT 与 SKIP

这个用法依然和 SQL 一样，表示对返回数量的限制或者是否执行跳过操作（常用于分页）。

#### 去重 DISTINCT

DISTINCT 用在 RETURN 后面表示去重。

```CQL
MATCH (:Person)-[]->(l:Loan)
RETURN DISTINCT p.name
```

#### 投影

```CQL
// 返回 Loan 节点的所有属性
MATCH (l:Loan)
RETURN l {.*} AS loan

// 仅返回 Loan 节点的 loanId 属性
MATCH (l:Loan)
RETURN l {.loanId} AS loan
```

另外这里还可以额外定义一个不存在的属性作为返回的内容，就像这样：

```CQL
// 返回 Loan 节点的所有属性并加一个 isSuccess 的属性
// 注意这个 isSuccess 的附加属性前面没有 '.'
MATCH (l:Loan)
RETURN l {.*, isSuccess: true} AS loan
```

直接返回节点的时候，实际上每行返回这 4 个内容：

- identity - 唯一标识符
- labels - 列表形式的 Label （因为一个节点可以是多个 Label ）
- elementId - 对象 Id
- properties - 属性的键值对

#### CASE WHEN ... THEN 条件返回

```CQL
// 对不同年龄的 Person 分别返回不一样的标识
MATCH (p:Person)
RETURN
CASE
WHEN p.born.year < 1960 THEN 'old'
WHEN 1960 <= p.born.year < 2000 THEN 'middle'
ELSE p.born.year >= 2000 THEN 'young'
END
AS ageGroup
```

#### 返回路径

```CQL
// 返回所有 Person APPLIED Loan 的路径
MATCH p = ((p:Person)-[:APPLIED]->(l:Loan))
RETURN p
```

有一些可以用于分析路径的函数：

- `length(p)` - 返回路径长度（关系数）。
- `nodes(p)` - 返回一个包含路径上所有节点的列表。
- `relationship(p)` - 返回一个包含路径上所有关系的列表。

### 暂存结果 WITH

`WITH` 关键字类似 `RETURN` 关键字，能够返回中间结果，在需要一个中间变量存储的场景很有用。

另外由于聚合函数不能用在 `WHERE` 条件过滤中，因此这种情况下 `WITH` 会比较有用。

### UNION 与 UNION ALL

使用 `UNION` 关键字连接查询结果，类似 SQL ，`UNION` 要去重，`UNION ALL` 不去重。

### 函数

#### exists 匹配模式

```CQL
MATCH (p:Person)-[:APPLIED]->(l:Loan)
WHERE NOT exists( (p)-[:DEFAULTED]->(l:Loan) )
RETURN p.name
```

#### labels 查看节点标签

```CQL
MATCH (l:Loan)
RETURN labels(l)
```

#### types 查看关系类型

```CQL
MATCH ()-[r:APPLIED]->()
RETURN types(r)
```

#### keys 查看属性名

```CQL
MATCH (l:Loan)
RETURN keys(l)
```

#### count 计数

```CQL
//查看有多少个 Loan 节点
MATCH (l:Loan)
RETURN count(l)
```

`count(n)` 这种写法将不会包括 `n` 为空的情况，`count(*)` 会包括 `n` 为空的情况。

#### type 查看类型

```CQL
MATCH (p:Person)-[r]->(l:Loan)
RETURN p.name AS personName, type(r) AS relationshipType
```

#### collect 转为列表

`collect` 函数将结果聚合为列表。

```CQL
MATCH (l:Loan)
RETURN collect(l.loanId) AS loanIds
```

可以用 `listObj[index]` 这样的方式按索引访问列表的元素，比如上面的 `collect(l.loanId)[0]` 返回第一个 Loan 节点的 loanId 属性。

#### UNWIND 列表展开

`UNWIND` 关键字将列表转换为多行。

#### 日期与时间

```CQL
// 返回当前日期
MATCH (n) RETURN date() LIMIT 1

// 返回当前日期与时间
MATCH (n) RETURN datetime() LIMIT 1

// 返回当前时间
MATCH (n) RETURN time() LIMIT 1
```

这三个函数可以传参，`datetime()` 采用 ISO8601 的标准，假设不传时区，默认为 UTC 。

#### 时间间隔

```CQL
// duration.between(x.date1,x.date2)

// 天数间隔
// duration.inDays(x.datetime1,x.datetime2).days
```

### 分组与聚合

在 Cypher 中，分组是隐性完成的，不需要 SQL 中的 GROUP BY 关键字，一旦使用 `count()` 这样的聚合函数，所有非聚合结果的列就会成为分组键。

下面的生成式也会默认去聚合。

#### 列表生成式 List Comprehension

有点类似于 Python 的列表生成式的语法，生成一个列表作为结果返回。

```CQL
// 以一个列表形式返回名字中有 M 或 V 的人名，否则返回空列表
MATCH (p:Person)
RETURN [x IN p.name WHERE x CONTAINS 'M' OR x CONTAINS'V']
```

#### 模式生成式 Pattern Comprehension

```CQL
// 以一个列表的形式返回申请 Loan 的人名，如果一笔 Loan 没人申请过，就会是一个空列表
MATCH (l:Loan)
RETURN [(a:Person)-[:APPLIED]->(l:Loan)|a.name] AS applier_name
```

### 图查询 CALL

#### 查询图里有哪些属性

```CQL
CALL db.propertyKeys()
```

一旦定义了属性键，即使当前没有节点或关系使用该属性键，该属性键也会保留在图中。

#### 查询图里有哪些标签

```CQL
CALL db.labels()
```

#### 查询图的数据模型

```CQL
CALL db.schema.visualization()
```

#### 查询图里的节点属性类型

```CQL
CALL db.schema.nodeTypeProperties()
```

#### 查询图里的关系属性类型

```CQL
CALL db.schema.relTypeProperties()
```

### 图遍历与查询调优

#### EXPLAIN 与 PROFILE

`EXPLAIN` 和 SQL 一样可以查看查询执行过程，是否走索引等。

`PROFILE` 可以查看检索行数，内存使用，性能调优。

使用 EXPLAIN 和 PROFILE 的区别在于，EXPLAIN 提供的是查询步骤的估计值，而 PROFILE 提供的是查询的确切步骤和检索的行数。

#### 查询的一般过程

1. 选定锚点 - 创建执行计划时，neo4j 会先确定作为查询起点的节点集加载到内存，查询的锚点是节点集中最少的节点数（有的时候可能有多个锚点）。
2. 展开路径 - 如果查询指定了路径，下一步就是沿着该路径前进，这一步被称为展开路径。
3. 返回结果 - 根据条件遍历返回结果（类似深度优先遍历）。

#### 变长遍历

`shortestPath()` 和 `allShortestPath()` 用于查询两个节点间的最短路径，如果有多条最短路径，`shortestPath()` 返回一条（具体哪条不确定），`allShortestPath()` 返回所有，用法如下：

```CQL
// 查找与 loanId 这条 Loan 有关系的所有 Loan 节点
MATCH p = shortestPath((l1:Loan)-[*]-(l2:Loan))
WHERE l1.loanId = 123
RETURN l2
```

变长遍历写在关系里，用法如下：

```CQL
// 两个节点正好能用 2 段关系连接
()-[*2]-()

// 两个节点用于连接的关系数大于等于 1
()-[*1..]-()

// 两个节点用于连接的关系数小于等于 3
()-[*..3]-()

// 两个节点用于连接的关系数大于等于 1 小于等于 3
()-[*1..3]-()

// 两个节点用于连接的关系数大于等于 1 小于等于 3 并且关系都是 APPLIED
()-[:APPLIED*1..3]-()
```

#### 优化点与思路

- 避免指定非锚点的节点的标签（如果逻辑上可以），这一步的目的是避免做类型检查。

### 子查询

```CQL
CALL {
	子查询
}
```

### 参数化

在 Cypher 中，参数名以 `$` 开头。这样做的好处是对于

#### 设置参数

```CQL
// 设置单个参数 personName 为 Movis ，冒号后面一定要空格
:param personName: 'Movis'

// 设置多个参数，以键值对的形式设置
:params {paramOne: 'A', paramTwo: 2}

// 设置整数最好使用 => 强制指定
:param number=>10
```

#### 查看参数

```CQL
:params
```

#### 删除参数（清空）

```CQL
:params {}
```

#### 参数化查询

```CQL
// 设置参数 personName 为 Movis
:param personName: 'Movis'

// 查询 Movis 申请的贷款的 loanId
MATCH (p:Person)-[:APPLIED]->(l:Loan)
WHERE p.name = $personName
RETURN l.loanId
```



## 写入

### 创建 MERGE 和 CREATE

#### 创建节点

建议使用 `MERGE` 关键字创建节点。另外一个也可以创建节点的关键字是 `CREATE` ，使用 `CREATE` 创建节点时，再添加节点之前不查找主键。如果确定数据时干净的，可以用 `CREATE` 获得更快的速度，而 `MERGE` 的优势是解决了重复问题。

```CQL
MERGE (l:Loan {loanId: 123, createdAt: '2024-04-10'})
```

也可以多个语句一起执行：

```CQL
// 创建 Loan 和 Person 节点并返回这两个节点
MERGE (l:Loan {loanId: 123, createdAt: '2024-04-10'})
MERGE (p:Person {name: 'movis', born: 1997})
RETURN l, p
```

#### 创建关系

当为两个节点创建关系的时候，也可以用 `MERGE` ，但这个关系必须满足：

- 有类型
- 有方向

```CQL
MATCH (l:Loan {loanId: 123})
MATCH (p:Person {name: 'movis', born: 1997})
MERGE (p)-[:APPLIED]->(l)
```

关系和节点一样也可以用 JSON 的格式创建属性，也可以同时创建节点和关系：

```CQL
MERGE (l:Loan {loanId: 123, createdAt: '2024-04-10'})
MERGE (p:Person {name: 'movis', born: 1997})
MERGE (p)-[:APPLIED]->(l)
```

甚至同时创建节点和关系可以简化为一句：

```CQL
MERGE (:Loan {loanId: 123, createdAt: '2024-04-10'})<-[:APPLIED]-(:Person {name: 'movis', born: 1997})
```

**默认情况下，不指定方向时， `MERGE` 将按从左到右的方向创建关系。**

```cql
// Person APPLIED Loan
MERGE (:Person {name: 'movis', born: 1997})-[:APPLIED]->(:Loan {loanId: 123, createdAt: '2024-04-10'})

// Person APPLIED Loan
MERGE (:Person {name: 'movis', born: 1997})-[:APPLIED]-(:Loan {loanId: 123, createdAt: '2024-04-10'})
```

上面两段语是等价的。

#### MERGE 与 CREATE

`MERGE` 在创建之前将会去图中寻找是否有相关的模式，如果要创建的数据点存在就不会创建。可以使用下面的方法自定义 `MERGE` 创建时的操作：

```CQL
MERGE (l:Loan {loanId:123})
ON CREATE SET l.createdAt = '2024-04-10'
ON MATCH SET l.createdAt = '2024-04-10'
```

这样重复创建时将会有不同的属性。

### 更新 SET 和 REMOVE

#### 添加和更新 SET

```CQL
MATCH (p:Person {name: 'movis'})-[:APPLIED]->(l:Loan {loanId: 123})
SET p.born = 1965, l.createdAt = '2024-04-11', l:ApprovedLoan
```

可以用逗号分隔，一次性更新或添加多个属性或者标签。

#### 移除 REMOVE 或者 SET

可以用 `REMOVE` 关键字来删除：

- 属性
- 标签

```CQL
// 移除 Person movis 的 born 属性
MATCH (p:Person {name: 'movis'})
REMOVE p.born

// 移除 Person movis 的 born 属性
MATCH (p:Person {name: 'movis'})
SET p.born = NULL

// 移除 Loan 的 ApprovedLoan 标签
MATCH (l:Loan {loanId: 123})
REMOVE l:ApprovedLoan
```

`SET` 一个属性为空和 `REMOVE` 这个属性是等价的，都是移除一个属性。

### 删除 DELETE

可以用 `DELETE` 关键字来删除：

- 节点
- 关系

```CQL
// 删除节点
MATCH (l:Loan {loanId: 123})
DELETE l

// 删除关系
MATCH (:Person {name: 'movis'})-[r:APPLIED]->(:Loan {loanId: 123})
DELETE r

// 同时删除节点和关系
MATCH (p:Person {name: 'movis'})
DETACH DELETE p
```

值得注意的是，一个节点如果还有关系的时候， `DELETE` 需要**先删除关系再删除节点**（节点与自己有关系也不能直接删）。除非使用 `DETACH DELETE` 。

## 约束

### 创建唯一约束

```CQL
CREATE CONSTRAINT [约束名] [IF NOT EXISTS]
FOR (n:Label名称)
REQUIRE n.属性 IS UNIQUE
```

其中加括号表明是可选项：

- 约束名 - 如果不指定约束名，neo4j 会自动生成名称。
- IF NOT EXISTS - 如果不加这句，当约束存在时，会报错。

### 查看约束

```CQL
SHOW CONSTRAINTS
```

### 删除约束

```CQL
DROP CONSTRAINT 约束名 [IF EXISTS]
```

其中加括号表明是可选项：

- IF EXISTS - 如果不加这句，当约束不存在时，会报错。

## 从 CSV 文件建图

### 从 CSV 文件导入数据

从 CSV 文件导入数据的语法为：

```CQL
LOAD CSV [WITH HEADERS] FROM 文件地址 [AS 文件别名] [FIELDTERMINATOR 分隔符（这里要加引号）]
```

其中加括号的表明是可选项：

- `WITH HEADERS` - 声明是否有表头。
- `AS xxx` - 使用 xxx 作为别名。
- `FIELDTERMINATOR` - 指定分隔符，类似 `pandas.load_csv()` 函数里的 `sep` 参数。

但是这里仅仅是将数据导入到图数据库中了，并没有创建节点和关系。

### 创建节点

假设有一张表里有想创建的 Loan 节点：

| loan_id | created_at | amount |
| ------- | ---------- | ------ |
| 123     | 2024-01-02 | 200    |
| 456     | 2024-01-03 | 500    |
| 789     | 2024-01-04 | 1000   |

```CQL
LOAD CSV WITH HEADERS FROM 文件地址 [AS 文件别名]
MERGE (l:Loan {loanId: toInteger(文件别名.loan_id)})
SET
l.createdAt = 文件别名.created_at,
l.amount = toInteger(文件别名.amount)
```

这样就创建了 3 个 Loan 类型的节点，值得注意的是**多行 `SET` 是有逗号的**。

### 创建关系

假设有 Person 来 APPLIED Loan 这样的关系，在 APPLIED 关系上需要记录 appliedAt 这个时间。

从一张 CSV 表格创建关系如下：

| person_id | loan_id | applied_at |
| --------- | ------- | ---------- |
| 1         | 123     | 2024-01-01 |
| 2         | 346     | 2024-01-02 |
| 3         | 789     | 2024-01-03 |

```CQL
LOAD CSV WITH HEADERS FROM 文件地址 AS row
MATCH (p:Person {personId: toInteger(row.person_id)})
MATCH (l:Loan {loanId: toInteger(row.loan_id)})
MERGE (p)-[r:APPLIED]->(l)
SET r.appliedAt = row.applied_at
```

## 附加标签

当查询有 Person APPLIED Loan 这种关系的 Person 时一般这样：

```CQL
MATCH (p:Person)-[:APPLIED]->(:Loan) RETURN p
```

其实可以给 Person 创建一个 Applier 标签，这样在查询这种关系时会更快（查询标签比查询关系快）：

```CQL
// 创建 Applier 标签
MATCH (p:Person)-[:APPLIED]->(:Loan) 
[WITH DISTINCT] SET p:Applier

// 查询 Person APPLIED Loan 的 Person
MATCH (p:Applier) RETURN p
```

其中，使用 `WITH DISTINCT` 会更具有写入效率。

## 存储过程

### 事务

在一个事务中，如果出现了错误，数据就会回滚。但是在单个事务中执行大量写入操作可能会导致性能问题和潜在的故障。

因此 Cypher 提供了自己控制事务的语句。

```CQL
CALL {
  存储过程
} IN TRANSACTIONS [OF 数量 ROWS]
```

其中加括号的表明是可选项：

- OF 数量 ROWS - 指定要处理的行数以进行批处理。

### Eager

neo4j 在导入数据时有一种名为 `Eager` 的机制，大概意思是在导入前会将数据全部加载到内存，这样的话可能会导致内存不足。避免 Eager 的方式是将数据分为很多小部分，比如分别创建节点和关系。
