---
title: 「数据结构」 - 学习计划 
date: 2022-12-12 11:34:41
categories: [ComputerScience, Algorithm, LeetCode]
tags: [python, array]
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

