---
title: 「数据结构」 - 学习计划 
date: 2022-12-17 02:10:41
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

### 566. 重塑矩阵

比较直观的解法，假设有一个点 `(a,b)` 在源矩阵 `m*n` 里面，如果源矩阵用一维数组表示，那么它的位置就是 `b+a*n` ，对于一个新矩阵 `r*c` 来讲，它对应 `r` 的位置就应该是 `((b+a*n)//c, (b+a*n)%c)` 这个点：

```python
class Solution:
    def matrixReshape(self, mat: List[List[int]], r: int, c: int) -> List[List[int]]:
        n = len(mat[0])
        m = len(mat)
        if m*n != r*c:
            return mat
        res = [[0]*c for _ in range(r)]
        for row in range(m):
            for col in range(n):
                res[(col+row*n)//c][(col+row*n)%c] = mat[row][col]
        return res
```

这里我们用了两层循环，其实如果用一维数组做循环也可以，重点变成双方都找 `(loc//column, loc%column)` 的位置了：

```python
class Solution:
    def matrixReshape(self, mat: List[List[int]], r: int, c: int) -> List[List[int]]:
        n = len(mat[0])
        m = len(mat)
        if m*n != r*c:
            return mat
        res = [[0]*c for _ in range(r)]
        for loc in range(m*n):
                res[loc//c][loc%c] = mat[loc//n][loc%n]
        return res
```

### 118. 杨辉三角

又是一道考察多维数组的题，我们可以模拟题目描述中的动画：

```python
class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        res = [[1]]
        for i in range(numRows-1):
            temp = [1]
            for k in range(len(res[-1])-1):
                temp.append(res[-1][k]+res[-1][k+1])
            temp += [1]
            res.append(temp)
        return res
```

然后我们也可以想象下这个假如是个二维数组坐标系：

|         | **col** |      |      |      |      |
| ------- | ------- | ---- | ---- | ---- | ---- |
| **row** | 1       |      |      |      |      |
|         | 1       | 1    |      |      |      |
|         | 1       | 2    | 1    |      |      |
|         | 1       | 3    | 3    | 1    |      |
|         | 1       | 4    | 6    | 4    | 1    |

在 `row-col` 坐标系上非行首行尾点 `(a,b)` 其实等于 `(a-1, b-1)+(a-1, b)` ：

```python
class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        res = [[1]*(row+1) for row in range(numRows)]
        for row in range(2, numRows):
            for col in range(1, row):
                res[row][col] = res[row-1][col-1]+res[row-1][col]
        return res
```

### 36. 有效的数独

我们首先遍历一遍数独，得到一个每个数的存储信息的 `dict` ：

- `dict` 的 `key` - 存放每个数，除了 `'.'` 这个代表空白的值。
- `dict` 的 `value` - 存放一个 `list` ，这个 `list` 里存放的是 `(row_index, col_index, block_index)` 的 `tuple` 用来记录这个数的位置信息。

得到信息后对字典每个 `value` 进行判断，需要当前 `list` 里的 `row_index` ， `col_index` ， `block_index` 都不相同，任意一个相同时返回错误，否则返回正确。

```python
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        num_loc = {}
        for i in range(9):
            for j in range(9):
                if board[i][j] != '.':
                    if board[i][j] not in num_loc:
                        num_loc[board[i][j]] = [(i,j,((i//3)*3+(j//3)))]
                    else:
                        num_loc[board[i][j]].append((i,j,((i//3)*3+(j//3))))
        for v in num_loc.values():
            row_set = set('')
            col_set = set('')
            block_set = set('')
            for i in v:
                if i[0] not in row_set:
                    row_set.add(i[0])
                else:
                    return False
                if i[1] not in col_set:
                    col_set.add(i[1])
                else:
                    return False
                if i[2] not in block_set:
                    block_set.add(i[2])
                else:
                    return False
        return True
```

然后就可以发现上面的逻辑可以优化，上述逻辑是先放后比，边放边比会更快。

```python
class Solution:
    def isValidSudoku(self, board: List[List[str]]) -> bool:
        row_record = [set('') for _ in range(9)]
        col_record = [set('') for _ in range(9)]
        block_record = [set('') for _ in range(9)]
        for i in range(9):
            for j in range(9):
                if board[i][j] != '.':
                    if board[i][j] in row_record[i]:
                        return False
                    if board[i][j] in col_record[j]:
                        return False
                    if board[i][j] in block_record[(i//3)*3+(j//3)]:
                        return False
                    row_record[i].add(board[i][j])
                    col_record[j].add(board[i][j])
                    block_record[(i//3)*3+(j//3)].add(board[i][j])
        return True
```

### 73. 矩阵置零

和上面这道题思路比较像，先统计一下 0 的位置信息，再修改。

```python
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        row_record = set('')
        col_record = set('')
        m = len(matrix)
        n = len(matrix[0])
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == 0:
                    row_record.add(i)
                    col_record.add(j)
        for i in range(m):
            if i in row_record:
                matrix[i][:] = [0]*n
            else:
                for j in col_record:
                    matrix[i][j] = 0
        return None
```
