---
title: sqlmodel 阅读笔记
date: 2022-12-06 16:51:53
categories: [DataScience, Database]
tags: [sqlmodel, fastapi, orm, sqlalchemy]
---

SQLModel 是一个使用 Python 对象与 SQL 数据库交互的库。它与 FastAPI 的作者是同一个人，旨在简化 FastAPI 应用程序中与数据库的交互。

本文是 SQL 官方文档的阅读笔记。用 Python 的 FastAPI 框架， MySQL 提供演示。

# ORM

ORM 的全称为 [Object–Relational Mapping](https://en.wikipedia.org/wiki/Object%E2%80%93relational_mapping) ，中文叫对象关系映射，是一种程序设计技术，用于实现面向对象编程语言与不同类型资料间的转换。经过对象关系映射这种技术后，一个类就对应着一张表，一个对象（这个类的实例）就对应着一条记录（一行数据），一个类的属性就对应着一个字段（一列）。

比如我一张表长这样：

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202212060551369.png)

那么在 Python 里它的数据类是：

```python
from sqlalchemy import Boolean, Column, ForeignKey, Integer, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(256), unique=True, index=True)
    hashed_password = Column(String(256))
    is_active = Column(Boolean, default=True)
```

ORM 这种技术有好处也有坏处。

好处就是方便开发。举个例子，如果你在开发系统中不用 ORM ，那么你后端可能需要写大量的原生 SQL 比如 MySQL，然后当你需要把你的数据库换成 PostgreSQL 时，你需要重写所有的 SQL 代码，这在项目大的时候是一件比较头疼的事情。

坏处是**性能是不如原生 SQL** 的，有的复杂查询可能无法表达。

在 Python 中有很多 ORM 框架，它们可以帮助开发者实现对象关系映射而不必关注细节，如 DjangoORM, SQLAlchemy 以及本文学习的 SQLModel 等。

SQLModel 的官方

# 基础教程

## 使用 SQL 创建表

```mysql
CREATE TABLE `test` (
  `id` bigint NOT NULL,
  `numeric_field` int DEFAULT NULL,
  `string_field` varchar(255) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  PRIMARY KEY (`id`)
)
```

可以看到表已经创建好。

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202212060731781.png)

## 使用 SQLModel 创建表

```python
from typing import Optional
from sqlmodel import create_engine, Field, SQLModel
# mysql 的数据类型，为了和上面一样所以用 mysql 的数据类型
from sqlalchemy.dialects.mysql import BIGINT, VARCHAR


class Test(SQLModel, table=True):
    id: Optional[int] = Field(BIGINT(19), primary_key=True)
    numeric_field: Optional[int]
    string_field: Optional[str] = Field(VARCHAR(256))


MYSQL_URL = 'mysql://root:123456@localhost:3306/test'
engine = create_engine(MYSQL_URL, echo=True)
SQLModel.metadata.create_all(engine)
```

这样就行了。不过这个地方有个坑，如图：

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202212060906280.png)

好像不能创建 BIGINT 类型的字段。然后我看了下源码。

这个 BIGINT 继承了 `sqlalchemy.types.BigInteger` 这个类，在[官方文档](https://docs.sqlalchemy.org/en/14/core/type_basics.html#sqlalchemy.types.BigInteger)里有一段：

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202212060934932.png)

大意就是说 BIGINT 这种写法实现不了它字面意思的效果，这是我没想到的。

经过搜索，发现要这样写才行：

```python
from typing import Optional
from sqlmodel import create_engine, Field, SQLModel, VARCHAR
from sqlalchemy.dialects.mysql import BIGINT
from sqlalchemy import Column


class Test(SQLModel, table=True):
    # primary_key=True 必须放在 Column 里面
    id: Optional[int] = Field(default=None, sa_column=Column(BIGINT(19), primary_key=True))
    numeric_field: Optional[int]
    string_field: Optional[str] = Field(VARCHAR(256))


MYSQL_URL = 'mysql://root:123456@localhost:3306/test'
engine = create_engine(MYSQL_URL, echo=True)
SQLModel.metadata.create_all(engine)
```

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202212060949238.png)

额，还是建议建表这块直接用 DDL 在数据库里建。

## 创建行 - INSERT

SQL 创建行（插入数据）：

```mysql
INSERT INTO test(numeric_field, string_field) VALUES (1, 'raw_sql')
```

用 SQLModel 插入（包含上面的创建）：

```python
from typing import Optional
from sqlmodel import create_engine, Field, SQLModel, Session, VARCHAR
from sqlalchemy.dialects.mysql import BIGINT
from sqlalchemy import Column


class Test(SQLModel, table=True):
    id: Optional[int] = Field(default=None, sa_column=Column(BIGINT(19), primary_key=True))
    numeric_field: Optional[int]
    string_field: Optional[str] = Field(VARCHAR(256))


MYSQL_URL = 'mysql://root:123456@localhost:3306/test'
engine = create_engine(MYSQL_URL, echo=True)


def create_table_test():
    SQLModel.metadata.create_all(engine)


def create_test_record():
    test_1 = Test(numeric_field=2, string_field='sqlmodel')

    # 创建会话
    session = Session(engine)
    # 添加更改
    session.add(test_1)
    # 提交更改
    session.commit()
    # 关闭会话
    session.close()


def create_test_record2():
    ''' 第二种创建行的方法 '''
    test_1 = Test(numeric_field=2, string_field='sqlmodel')

    with Session(engine) as session:
        session.add(test_1)
        session.commit()


if __name__ == '__main__':
    create_table_test()
    create_test_record()
```

结果如下：

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202212061008631.png)

从 `create_test_record` 这个函数我们可以看到添加数据分为 2 步，第一步是 `add` ，第二步是 `commit` ，这个有点像 `git` 。

## 自增 ID、默认值与刷新数据

在上文的代码中：

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202212061031451.png)

这两个框之前，我们都创建了一个 `test_1` 对象，相当于数据库一条记录，并且没有在代码里写主键 `id` 。我们知道主键都是非空的，所以经过打印，发现使用 SQLModel 创建行之前，实际上的 `test_1` 对象的 `id` 是 `None` ，类型就是 `NoneType` 而不是 `int` 。但是在 Python 代码里，这个值是空的，所以很显然设置了主键自增。

看源码可以发现 `SQLAlchemy` 的 `Column` 有一个参数叫 `autoincrement` ，所以这个字段其实就是用来控制是否自增，[原文文档](https://docs.sqlalchemy.org/en/14/core/metadata.html#sqlalchemy.schema.Column.params.autoincrement)。

```python
def create_test_record():
    test_2 = Test(numeric_field=3, string_field='sqlmodel')

    # 创建会话
    session = Session(engine)
    # 添加更改
    print('before add', test_2)
    session.add(test_2)
    # 提交更改
    print('before commit', test_2)
    session.commit()
    # 关闭会话
    print('before close', test_2)
    session.close()
```

把创建新的行写成这样，去观察什么时候更新的自增 `id` ，可以看到：

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202212061100119.png)

自增是数据库的行为，发生在提交代码的 `commit` 之后。

实际上有个主动刷新的功能，代码如下：

```python
def create_test_record():
    test_2 = Test(numeric_field=5, string_field='sqlmodel')

    session = Session(engine)
    
    print('before add', test_2)
    session.add(test_2)
    
    print('before commit', test_2)
    session.commit()
    
    # 主动刷新
    print('before refresh', test_2)
    session.refresh(test_2)

    print('before close', test_2)
    session.close()
```

可以得到：

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202212061105795.png)

## 查询 - SELECT

查询刚刚创建的数据，用 SQL ：

```mysql
SELECT
	id,
	numeric_field,
	string_field
FROM
	test;
```

可以看到：

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202212061115758.png)

在 SQLModel 里只需要再添加如下代码：

```python
from sqlmodel import select

def read_test_record():
    with Session(engine) as session:
        statement = select(Test)
        results = session.exec(statement)
        for test in results:
            print(test)
```

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202212061121588.png)

**注意这里返回的 `test` 就是一个 `Test` 的实例化对象，而不是字符串。**

这里就完成了关系到对象的反映射。

不过一般这么写：

```python
def read_test_record():
    with Session(engine) as session:
        results = session.exec(select(Test)).all()
        print(results)
        print(type(results))
```

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202212061125644.png)

可以看到这次返回的是一个 `list` 。

这里作者专门提到了 SQLModel 与 SQLAlchemy 在这里的不同之处，意思就是 SQLModel 虽然是基于 SQLAlchemy 与 Pydantic 的，依然有自己的封装。除了上文里其实已经遇见的 `Field` 和 `Column` ，还有就是查询的时候方法名并不一样：

```python
# SQLAlchemy
session.execute(select(Test)).scalars().all()

# SQLModel
session.exec(select(Test)).all()
```

但是 SQLModel `Session` 仍然可以访问 `session.execute()` 。

我的感觉就是说如果你在使用 SQLModel 中遇到一些问题，你可以直接去看 SQLAlchemy 的文档（我没有专门学这个框架因为它的文档实在太长了），事实上我已经这么做了（比如上面对表字段的类型定义）。

## 过滤 - WHERE

在 SQL 中，我们可以用 WHERE 来进行数据的过滤与筛选，比如：

```mysql
SELECT
	id,
	numeric_field,
	string_field
FROM
	test
WHERE
	numeric_field = 3;
```

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202212061148484.png)

作者在这里贴心地提示了：

- SELECT - 用于告诉数据库返回哪些列。
- WHERE - 用于告诉数据库返回哪些行。

上述 SQL 用 SQLModel 实现就是：

```python
def read_test_record():
    with Session(engine) as session:
        results = session.exec(select(Test).where(Test.numeric_field == 3)).all()
        print(results)
```

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202212061237791.png)

对于逻辑运算符：

| 逻辑运算符 | SQL                             | SQLModel                                                     |
| ---------- | ------------------------------- | ------------------------------------------------------------ |
| AND        | WHRER condition1 AND condition2 | where(condition1).where(condition2) 或者 where(condition1, condition2) |
| OR         | WHERE condition1 OR condition2  | where(or_(condition1, cindition2))                           |

然后作者这里提到了说 Python 解释器可能会对 `where(Test.id > 3)` 这种写法报错，因为在创建的时候用了 `Optional[int]` 这样的声明。解决方案是写成 `where(col(Test.id) > 3)` 。我没有遇到这个问题，这里就不作展示。

## 索引 - INDEX

索引是一种加速查询的技术，优点：

- 查询快

缺点：

- 需要额外物理空间
- 增删改慢

在 SQL 中，如果是上面的例子，要对 `numeric_field` 创建索引：

```mysql
CREATE INDEX 
	idx_test_numeric_field 
ON 
	test(numeric_field)
```

在 SQLModel 中：

```python
class Test(SQLModel, table=True):
    id: Optional[int] = Field(BIGINT(19), primary_key=True)
    numeric_field: Optional[int] = Field(index=True) # 只需要在这里进行小的修改
    string_field: Optional[str] = Field(VARCHAR(256))
```

> 数据库会为主键创建一个内部索引，因此主键不需要索引。

## 查询一行数据

在上文查询或过滤那里我们都查询了所有数据，这里讲如果我们只需要查询一行数据时的操作和会遇到的情况。

其实很简单，把 `session.exec(select(Test).where(Test.id>3))` 当作一个返回结果的话，在上文调用了 `.all()` 方法返回了一个 `list` ，如果不调用返回的是一个可迭代对象，这里我们可以调用 `.first()` 等方法来查询一行数据，比如：

```python
session.exec(select(Test).where(Test.id>3)).first()
```

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202212070504699.png)

在这里也可能查询不到任何行，也就是没有满足条件的数据，比如：

```python
session.exec(select(Test).where(Test.id<0)).first()
```

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202212070505626.png)

除了 `.first()` ，还有一个 `.one()` 也类似，这里列出它们的差异：

| 情况       | 有多行满足条件时返回        | 只有一行满足条件时返回 | 没有满足条件的数据时返回 |
| ---------- | --------------------------- | ---------------------- | ------------------------ |
| `.first()` | 一个 Test 实例对象          | 一个 Test 实例对象     | None                     |
| `.one()`   | 报错 `MultipleResultsFound` | 一个 Test 实例对象     | 报错 `NoResultFound`     |

如果只用主键选择单行，由于主键是唯一的，还有个 `.get()` 方法：

```python
# get()
session.get(Test, 3)
# first()
session.exec(select(Test).where(Test.id == 3)).first()
# one()
session.exec(select(Test).where(Test.id == 3)).one()
```

上面 3 句话结果是一样的。如果用 `.get()` 方法去查询一个不存在的主键，会返回 `None` 。

## LIMIT 与 OFFSET

限制查询满足某种条件的数据 3 条，在 SQL 里：

```mysql
SELECT
	*
FROM
	test
WHERE
	numeric_field >= 3
LIMIT
	3
```

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202212070532914.png)

在 SQLModel 里：

```python
session.exec(select(Test).where(Test.numeric_field >= 3).limit(3)).all()
```

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202212070532521.png)

这个地方我个人觉得要注意的就是 `limit()` 的位置，是放在 `exec()` 里面的，也就是对 `select().where()` 返回的对象做了 `limit()` 操作。

增加 `offset` 偏移量也是一样，在 SQL 里：

```mysql
SELECT
	*
FROM
	test
WHERE
	numeric_field >= 3
LIMIT
	2
OFFSET
	2
```

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202212070534313.png)

这里注意总共满足条件的数据只有 3 条，在跳过 2 条后（ `OFFSET 2` ）只剩下 1 条数据可查询，所以这里即使 `LIMIT 2` ，也只返回了一条数据。

在 SQLModel 里：

```python
session.exec(select(Test).where(Test.numeric_field >= 3).offset(2).limit(2)).all()
# 或
session.exec(select(Test).where(Test.numeric_field >= 3).limit(2).offset(2)).all()
```

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202212070539528.png)

这里 `offset()` 和 `limit()` 的顺序没关系，不影响结果。

## 更新 - UPDATE

在 SQL 中：

```mysql
UPDATE
	test
SET
	string_field = 'SQL_UPDATE_ID_3'
WHERE
	id = 3
```

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202212071335853.png)

在 SQLModel 中：

```python
def update_test_record():
    with Session(engine) as session:
        result = session.get(Test, 4)
        result.string_field = 'SQLMODEL_UPDATE_ID_4'
        session.add(result)
        session.commit()
```

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202212071339300.png)

可以看到， `id=4` 这条记录的字段已经更新成功。不过这种写法与其说是更新，不如说是覆盖，因为我感觉这代码几乎和 `INSERT` 操作一样。因此我做个尝试，如下：

```python
def update_test_record():
    with Session(engine) as session:
        result = Test(id=5, numeric_field=5, string_field='CUSTOM_ID_5')
        result.string_field = 'SQLMODEL_UPDATE_ID_5'
        session.add(result)
        session.commit()
```

然而报错了，是在 `session.commit()` 时出错的，错误信息如下：

```
sqlalchemy.exc.IntegrityError: (MySQLdb.IntegrityError) (1062, "Duplicate entry '5' for key 'test.PRIMARY'")
[SQL: INSERT INTO test (id, numeric_field, string_field) VALUES (%s, %s, %s)]
[parameters: (5, 5, 'CUSTOM_ID_5')]
(Background on this error at: https://sqlalche.me/e/14/gkpj)
```

SQLAlchemy 与 MySQLdb 官方文档给出的解释是：

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202212071523782.png)

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202212071525723.png)

还是不太明白这里为啥会有问题。

不过如果非要做这样的操作，可以使用 `merge()` ，这样：

```python
def custom_update():
    with Session(engine) as session:
        test = Test(id=5, numeric_field=5, string_field='CUSTOM_ID_5')
        session.merge(test)
        session.commit()
```

这样无论是更新数据还是插入数据都可以。

## 删除 - DELETE

在 SQL 里：

```mysql
DELETE
FROM
	test
WHERE
	id = 7
```

删前：

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202212081124913.png)

删后：

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202212081125097.png)

在 SQLModel 里：

```python
def delete_test_record():
    with Session(engine) as session:
        result = session.get(Test, 6)
        session.delete(result)
        session.commit()
```

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202212081129966.png)

跟 `INSERT` 数据类似，这里在调用 `.delete()` 后也需要 `commit()` 。

## 连接 - JOIN

### 创建连接表

这里主要是讲的外键。

这里比如我需要建一张表叫 `test2` ，其中主键是 `id` ，还有一个字段是 `test_id` ，但这个字段与 `test` 表的字段具有外键关系。

```python
class Test2(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    test_id: int = Field(foreign_key='test.id')
```

这样就可以了。

### 创建

跟正常的一样。

### 查询

