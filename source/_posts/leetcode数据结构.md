---
title: 「数据结构」 - 学习计划 
date: 2022-12-15 01:50:41
categories: [ComputerScience, Algorithm, LeetCode]
tags: [python, array, hash, point, sliding window]
---

# 数据结构入门

## 数组

### 217. 存在重复元素

简单 set 来判断是否有元素重复。

```python
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        hashset = set('')
        for i in nums:
            if i not in hashset:
                hashset.add(i)
            else:
                return True
        return False
```

### 53. 最大子数组和

首先数组限定了必有至少 1 个元素，那么我们可以假设最大子数和 `max_sum` 等于数组第 1 个元素。

这时我们思考，假如数组中有一段是最大子数和，那么其左右肯定是 0 或者比 0 小的子数和，所以我们可以定义一个 `temp_sum` ，初始化为 0 ，遍历子数并累加，当它小于 0 时，把它置 0 （代表从这开始不加了），当它大于 `max_sum` 时，`max_sum = temp_sum` 。

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        max_sum = nums[0]
        temp_sum = 0
        for i in nums:
            temp_sum += i
            if temp_sum > max_sum:
                max_sum = temp_sum
            if temp_sum < 0:
                temp_sum = 0
        return max_sum
```

### 1. 两数之和

可以建立一个 `dict` 来记录数组遍历过的值和它的下标，这样每次遍历到一个数时，看一下 `target-x` 是否在 `dict` 里就可以找到之前出现的数和这个数的和等于 `target` 的。

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        hashmap = {}
        for i,v in enumerate(nums):
            temp = target - v
            if temp in hashmap:
                return [hashmap[temp], i]
            else:
                hashmap[v] = i
        return None
```

### 88. 合并两个有序数组

这道题需要反着来，从 `nums1` 后面开始放数，优先放大的数，这个不是很容易想到。放的过程中有 4 种情况：

| Situation                   | Operation                                         |
| --------------------------- | ------------------------------------------------- |
| 两数组都没放完 `nums2` 的大 | 放 `nums2` 的数， `nums2` 的指针左移。            |
| 两数组都没放完 `nums1` 的大 | 放 `nums1` 的数， `nums1` 的指针左移。            |
| `nums1` 的数放完了          | 放 `nums2` 的数， `nums2` 的指针左移。            |
| `nums2` 的数放完了          | 不用放 `nums1` 的数了因为本来就在里面，跳出循环。 |

所以其实跳出循环的条件就是 `nums2` 的数放完了，剩下的情况里只有两数组都没放完 `nums1` 的大这种情况需要放 `nums1` 的数，因此：

```python
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        while n >= 1:
            if m>=1 and nums1[m-1] > nums2[n-1]:
                nums1[m+n-1] = nums1[m-1]
                m -= 1
            else:
                nums1[m+n-1] = nums2[n-1]
                n -= 1
        return None
```

### 350. 两个数组的交集 II

用两个 `dict` 分别计数数组中每个数出现的次数，然后遍历其中一个 `dict` ，如果数满足两个 `dict` 里 `key` 都有取 `value` 的最小值，然后将 `value` 个 `key` 加入结果。

```python
class Solution:
    def intersect(self, nums1: List[int], nums2: List[int]) -> List[int]:
        counter1 = {}
        counter2 = {}
        for i in nums1:
            if i not in counter1:
                counter1[i] = 1
            else:
                counter1[i] += 1
        for i in nums2:
            if i not in counter2:
                counter2[i] = 1
            else:
                counter2[i] += 1
        res = []
        for k,v in counter1.items():
            if k in counter2:
                t = min(v, counter2[k])
                res += [k]*t
        return res
```

### 121. 买卖股票的最佳时机

动态规划的入门题，卖是建立在买的基础上的。在找到买入更低点时更新最大利润直到找到下个买入最低点。

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if not prices:
            return 0
        price_min = 1e5
        profit_max = 0
        for i in prices:
            if i < price_min:
                price_min = i
            if i-price_min > profit_max:
                profit_max = i-price_min
        return profit_max
```

