---
title: 「算法」 - 学习计划
date: 2022-12-12 13:06:31
categories: [ComputerScience, Algorithm, LeetCode]
tags: [python, binary search]
---

# 算法入门

## 二分查找

### 704. 二分查找

经典二分查找。

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left = 0
        right = len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] > target:
                right = mid - 1
            else:
                left = mid + 1
        return -1
```

### 278. 第一个错误的版本

第一个错误的版本就是左边是正确的版本，右边是错误的版本时的右边的版本，也就是满足这个条件的时候就该跳出循环。

因此我们可以二分去查找，但是注意如果 `mid` 是错误的版本，`right` 等于它，反之 `left` 等于它。

```python
class Solution:
    def firstBadVersion(self, n: int) -> int:
        left = 0
        right = n
        while right-left != 1:
            mid = (left+right) // 2
            if isBadVersion(mid):
                right = mid
            else:
                left = mid
        return right
```

### 35. 搜索插入位置

经典二分法，搜不到的时候左指针就是该插入的位置。

```python
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        left = 0
        right = len(nums) - 1
        while left <= right:
            mid = (left+right) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] > target:
                right = mid - 1
            else:
                left = mid + 1
        return left
```

## 双指针

### 977. 有序数组的平方

这道题在解释里面是先平方再排序，也就是 `O(NlogN)` 的时间复杂度。

```python
class Solution:
    def sortedSquares(self, nums: List[int]) -> List[int]:
        return sorted(map(lambda x:x*x, nums))
```

这里我就不展开写排序算法了因为 `Python` 有的时候算法比内置函数慢在内置是用 `C` 写的。

也可以像二分查找的一样类似每遍历一个元素，平方，再搜索插入位置。

```python
class Solution:
    def sortedSquares(self, nums: List[int]) -> List[int]:

        def searchInsert(nums: List[int], target: int) -> int:
            left = 0
            right = len(nums) - 1
            while left <= right:
                mid = (left+right) // 2
                if nums[mid] == target:
                    return mid
                elif nums[mid] > target:
                    right = mid - 1
                else:
                    left = mid + 1
            return left
        
        res = []
        for i in nums:
            temp = i*i
            res.insert(searchInsert(res, temp), temp)
        return res
```

然而这个时间真的是慢得不行。

### 189. 轮转数组

在我的印象里这道题就是反转三次解决的题：

1. 反转素有元素
2. (0, k-1) 反转
3. (k, n-1) 反转

此外我们需要考虑下 `k >= n` ：

- k = n 时 - 结果和不移动一样。
- k > n 时 - 结果和 `k%n` 一样。

因此：

```python
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        n = len(nums)
        k %= n
        if k == 0:
            return None
        for i in range(n//2):
            nums[i], nums[n-i-1] = nums[n-i-1], nums[i]
        for i in range(k//2):
            nums[i], nums[k-i-1] = nums[k-i-1], nums[i]
        for i in range(k, (k+n)//2):
            nums[i], nums[n-i-1+k] = nums[n-i-1+k], nums[i]
        return None
```

但是这道题对 Python 的意义应该是深入理解切片的使用。

- `[start:stop：step]` 
  - 表示从下标 `start` 开始取到 `stop` 为止（不包括下标 `stop` 的值）的值，也就是前闭后开。
  - `step` 为正代表向右取数，`step` 为负代表向左取数，如果取不到结果就返回空子集（字符串是 `''` ，列表是 `[]`，元组是 `()` ），举个例子，字符串 `temp = "string"` ：
    - `temp[-1:1]` - 1 元素在 -1 元素右边，取不到。
    - `temp[-2:4]` - 等同于 `temp[4:4]` ，取不到因为不包括右边的值。
  - `step` 为 1 时可以省略 `step` 。对于 `start` 和 `stop` 来讲，省略相当于在这个方向上无限取数，这个方向和 `step` 有关。

```python
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        k %= len(nums)
        if k == 0:
            return None
        nums[::] = nums[::-1]
        nums[k::] = nums[:k-1:-1]
        nums[:k:] = nums[k-1::-1]
        return None
```

不过对于切片来讲，这道题其实可以一次切完。

```python
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        n = len(nums)
        k %= n
        if k == 0:
            return None
        nums[:k], nums[k:] = nums[n-k:], nums[:n-k]
        return None
```

