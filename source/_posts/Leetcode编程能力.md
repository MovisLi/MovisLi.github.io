---
title: 「编程能力」 - 学习计划
date: 2022-12-16 19:51:45
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

## 运算符

### 191. 位1的个数

首先可以转换成字符串，然后统计字符 `'1'` 的个数。这里取 `[2:]` 主要是用 `bin()` 这个函数转换之后是 `0bxxx` 这种形式。

```python
class Solution:
    def hammingWeight(self, n: int) -> int:
        n_str = bin(n)[2:]
        return n_str.count('1')
```

从位运算的角度考虑，判断奇偶可以用 `n&1==1?` 相当于看最后一位是不是 `1` ，所以可以将数字不断向右移位来判断 `1` 的个数。

```python
class Solution:
    def hammingWeight(self, n: int) -> int:
        res = 0
        while n:
            res += n&1
            n >>= 1
        return res
```

还有个操作 `n&(n-1)` 作用是将最右端的 `1` 置 `0` ：

```python
class Solution:
    def hammingWeight(self, n: int) -> int:
        res = 0
        while n:
            n &= n-1
            res += 1
        return res
```

### 1281. 整数的各位积和之差

模拟。

```python
class Solution:
    def subtractProductAndSum(self, n: int) -> int:
        bit_sum = 0
        bit_mul = 1
        while n:
            temp = n%10
            n //= 10
            bit_sum += temp
            bit_mul *= temp
        return bit_mul - bit_sum
```

## 条件语句

### 976. 三角形的最大周长

最开始想的是列出所有排列组合然后看满足条件的排列组合，找周长最大的那个，然后 `OOT` 了。

后来发觉这个其实是个排序然后滑动窗口的问题，由大到小（逆序）排序后，从最左开始滑动包含 3 个数的窗口，当遇到满足条件的窗口直接返回窗口内数的和，否则返回 0 。

```python
class Solution:
    def largestPerimeter(self, nums: List[int]) -> int:
        nums = list(sorted(nums, reverse=True))
        x = 0
        y = 1
        z = 2
        nums_len = len(nums)
        while z<nums_len:
            if nums[x]+nums[y]>nums[z] and nums[y]+nums[z]>nums[x] and nums[x]+nums[z]>nums[y]:
                return nums[x]+nums[y]+nums[z]
            x += 1
            y += 1
            z += 1
        return 0
```

### 1779. 找到最近的有相同 X 或 Y 坐标的点

简单模拟，这道题的坑在于这里的下标指的是点在 `points` 中的位置而非点的 x 坐标。

```python
class Solution:
    def nearestValidPoint(self, x: int, y: int, points: List[List[int]]) -> int:
        m_dis_min = 10001
        res = -1
        for index,point in enumerate(points):
            if point[0] == x or point[1] == y:
                m_dis = abs(point[0]-x)+abs(point[1]-y)
                if m_dis < m_dis_min:
                    res = index
                    m_dis_min = m_dis
        return res
```

## 循环

### 1822. 数组元素积的符号

没有必要真的去乘，只需要关注数的正负性对结果的影响。循环遍历数组：

- 如果是正数，乘积不变号，进入下次循环。
- 如果是 0 ，乘积为 0 ，跳出循环。
- 如果是负数，乘积变号，进入下次循环。

```python
class Solution:
    def arraySign(self, nums: List[int]) -> int:
        res = 1
        for i in nums:
            if i < 0:
                res *= -1
            elif i == 0:
                return 0
        return res
```

### 1502. 判断能否形成等差数列

先排序，并从前两个数求得差值，在依次遍历，如果发现某连续两个数的差值不等于之前求的差值，则说明不是等差数列，否则是等差数列。

```python
class Solution:
    def canMakeArithmeticProgression(self, arr: List[int]) -> bool:
        arr = list(sorted(arr))
        diff = arr[1]-arr[0]
        for i in range(len(arr)-1):
            if arr[i+1]-arr[i] != diff:
                return False
        return True
```

### 202. 快乐数

这个过程让我想到《火影忍者》里的忍术伊邪那美，我们试想下如果处在一个循环的进程中，你什么时候会发现你循环了呢？那就是过去的东西再次重复的时候。所以这道题很显然就是用一个 `set` 来存储过程中出现的数，如果出现的数在 `set` 里了就说明不是快乐数，否则是快乐数。

```python
class Solution:
    def isHappy(self, n: int) -> bool:
        num_set = {n}
        while n != 1:
            temp = 0
            while n!=0:
                temp += (n%10)*(n%10)
                n //= 10
            n = temp
            if n in num_set:
                return False
            else:
                num_set.add(n)
        return True
```

### 1790. 仅执行一次字符串交换能否使两个字符串相等

定义一个数组记录不相同的值的下标，如果两个字符串不同，当且仅当数组长度为 2 且字符串 `s1` 与 `s2` 不同的数交叉相等时，满足题意。

```python
class Solution:
    def areAlmostEqual(self, s1: str, s2: str) -> bool:
        if s1 == s2:
            return True
        counter = []
        for i,v in enumerate(s2):
            if v != s1[i]:
                counter.append(i)
        if len(counter) == 2 and s2[counter[0]] == s1[counter[1]] and s2[counter[1]] == s1[counter[0]]:
            return True
        return False
```

