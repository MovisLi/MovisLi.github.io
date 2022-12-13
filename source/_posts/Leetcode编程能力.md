---
title: 「编程能力」 - 学习计划
date: 2022-12-13 08:51:45
categories: [ComputerScience, Algorithm, LeetCode]
tags: [python]
---

# 编程能力入门

## 基本数据类型

### 1523. 在区间范围内统计奇数数目

首先就分为四种情况：

| low  | high | result               |
| ---- | ---- | -------------------- |
| 奇数 | 奇数 | $=(high-low)//2 + 1$ |
| 奇数 | 偶数 | $=(high-low)//2 + 1$ |
| 偶数 | 奇数 | $=(high-low)//2 + 1$ |
| 偶数 | 偶数 | $=(high-low)//2$     |

这就有了：

```python
class Solution:
    def countOdds(self, low: int, high: int) -> int:
        return (high - low)//2 if low&1==0 and high&1==0 else (high - low)//2 + 1
```

问题在于我发现这个算法好像不怎么快。如果要比这个更快的话只能没有这个 `if` 的条件判断了。

如果要合二为一的话，我想到一个问题，[0, 一个数 `x` ] 的奇数个数怎么计算呢？

很容易得出结果是 $(x+1)//2$，按照这个思路，其实就是 `(high+1)//2 - (low+1)//2` ，但是稍有不同的是这个式子其实没有算 `low` 这个数，如果 `low` 这个数是个奇数的话，因此：

```python
class Solution:
    def countOdds(self, low: int, high: int) -> int:
        return (high+1)//2 - low//2
```

### 1491. 去掉最低工资和最高工资后的工资平均值

直接调用函数是比较直接的：

```python
class Solution:
    def average(self, salary: List[int]) -> float:
        return (sum(salary)-max(salary)-min(salary))/(len(salary)-2)
```

我们也可以尽量把操作放在一次循环里。

```python3
class Solution:
    def average(self, salary: List[int]) -> float:
        salary_max = salary[0]
        salary_min = salary[0]
        salary_sum = 0
        for i in salary:
            salary_sum += i
            if i > salary_max:
                salary_max = i
            elif i < salary_min:
                salary_min = i
        return (salary_sum-salary_max-salary_min)/(len(salary)-2)
```

这里有个小坑，假如你的初始最大值是 `1e3` ，也就是理论上最小值，初始最小值是 `1e6` 也就是理论上的最大值，这里不能写 `if elif` ，因为对第一个值是最小值的情况会解答错误。
