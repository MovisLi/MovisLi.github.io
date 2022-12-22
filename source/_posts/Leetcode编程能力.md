---
title: 「编程能力」 - 学习计划
date: 2022-12-23 01:45:45
categories: [ComputerScience, Algorithm, LeetCode]
tags: [python, hash, point]
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

## 函数

### 589. N 叉树的前序遍历

递归，前序遍历（又称 NLR ）其实很简单，就是先遍历根节点，遍历子节点，和深度优先搜索是一样的，所以：

```python
class Solution:
    def preorder(self, root: 'Node') -> List[int]:
        res = []
        def dfs(node):
            if not node:
                return None
            res.append(node.val)
            for child in node.children:
                dfs(child)
        dfs(root)
        return res
```

非递归，用栈的思想，从根节点开始，先添加节点值进结果，在反序添加孩子的值进栈，每次循环出栈一个节点并打印，再添加。这个是更直观的深度优先搜索。

```python
class Solution:
    def preorder(self, root: 'Node') -> List[int]:
        if root is None:
            return None
        res = []
        stack = [root]
        while stack:
            temp = stack.pop()
            res.append(temp.val)
            for child in reversed(temp.children):
                stack.append(child)
        return res
```

### 496. 下一个更大元素 I

我们逆序遍历 `nums2` ，可以用一个 `stack` 来记录比当前所有数都大的数，如果比当前数小就出栈直到比当前数大或者非空，那么遍历到下一个更大元素实际上就是栈顶元素。

```python
class Solution:
    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        hashmap = {}
        stack = []
        for i in reversed(nums2):
            while stack and i>stack[-1]:
                stack.pop()
            hashmap[i] = stack[-1] if stack else -1
            stack.append(i)
        return [hashmap[i] for i in nums1]
```

### 1232. 缀点成线

一条直线上的点斜率相等。

```python
class Solution:
    def checkStraightLine(self, coordinates: List[List[int]]) -> bool:
        dy = coordinates[1][1]-coordinates[0][1]
        dx = coordinates[1][0]-coordinates[0][0]
        for point in coordinates[2:]:
            if (point[1]-coordinates[0][1])*dx != (point[0]-coordinates[0][0])*dy:
                return False
        return True
```

这样写是为了避免除数为 0 的情况。

## 数组

### 1588. 所有奇数长度子数组的和

很显然这必然是一道数学题。我们先用的解法，遍历数组模拟流程：

```python
class Solution:
    def sumOddLengthSubarrays(self, arr: List[int]) -> int:
        res = 0
        arr_len = len(arr)
        for i in range(arr_len):
            temp = 1
            while i+temp <= arr_len:
                for v in arr[i:i+temp]:
                    res += v
                temp += 2
        return res
```

之后我们想一种数学关系，先列出一个矩阵：

| 长度 | 数组                        | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    |
| ---- | --------------------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 1    | [1]                         | 1    |      |      |      |      |      |      |      |      |
| 2    | [1, 2]                      | 1    | 1    |      |      |      |      |      |      |      |
| 3    | [1, 2, 3]                   | 2    | 2    | 2    |      |      |      |      |      |      |
| 4    | [1, 2, 3, 4]                | 2    | 3    | 3    | 2    |      |      |      |      |      |
| 5    | [1, 2, 3, 4, 5]             | 3    | 4    | 5    | 4    | 3    |      |      |      |      |
| 6    | [1, 2, 3, 4, 5, 6]          | 3    | 5    | 6    | 6    | 5    | 3    |      |      |      |
| 7    | [1, 2, 3, 4, 5, 6, 7]       | 4    | 6    | 8    | 8    | 8    | 6    | 4    |      |      |
| 8    | [1, 2, 3, 4, 5, 6, 7, 8]    | 4    | 6    | 8    | 9    | 9    | 8    | 6    | 4    |      |
| 9    | [1, 2, 3, 4, 5, 6, 7, 8, 9] | 5    | 8    | 11   | 12   | 13   | 12   | 11   | 8    | 5    |

可以发现多项式的系数呈现对称关系。

以 5 为例，呈现这样的分布经过了 1, 3, 5 这 3 个奇数向量的相加，其中：

- 1 - 1, 1, 1, 1, 1
- 3 - 1, 2, 3, 2, 1
- 5 - 1, 1, 1, 1, 1

以 7 为例，呈现这样的分布经过了 1, 3, 5, 7 这 4 个奇数向量的相加，其中：

- 1 - 1, 1, 1, 1, 1, 1, 1
- 3 - 1, 2, 3, 3, 3, 2, 1
- 5 - 1, 2, 3, 3, 3, 2, 1
- 7 - 1, 1, 1, 1, 1, 1, 1

再以 9 为例，呈现这样的分布经过了 1, 3, 5, 7, 9 这 5 个奇数向量的相加，其中：

- 1 - 1, 1, 1, 1, 1, 1, 1, 1, 1
- 3 - 1, 2, 3, 3, 3, 3, 3, 2, 1
- 5 - 1, 2, 3, 4, 5, 4, 3, 2, 1
- 7 - 1, 2, 3, 3, 3, 3, 3, 2, 1
- 9 - 1, 1, 1, 1, 1, 1, 1, 1, 1

那可以看到 5 和 9 还算是相同的规律，7 好像有所不同和 3 有点像，那么我们去寻找 7 哪里不同，结合上文的对称关系，我们首先找每个数组的中间的数有什么不同：

- 3 的中间数 2 - 左右各有 1 个数，此系数为 2 = 1\*1 + 1\*1。左右各有 1 个奇数， 0 个偶数。

- 5 的中间数 3 - 左右各有 2 个数，此系数为 5 = 1\*1 + 2\*2。左右各有 1 个奇数， 1 个偶数。
- 7 的中间数 4 - 左右各有 3 个数，此系数为 8 = 2\*2 + 2\*2。左右各有 2 个奇数， 1 个偶数。
- 9 的中间数 5 - 左右各有 4 个数，此系数为 13 = 2\*2 + 3\*3。左右各有 2 个奇数， 2 个偶数。

可以看到其实每个中间数的系数其实=左右奇数数量的平方+左右偶数数量加 1 的平方。

然后再观察 9 的其他项发现这个公式容易推广成每个数的系数 = 左边奇数个数\*右边奇数个数 + (左边偶数个数+1)\*(右边偶数个数+1)。

```python
class Solution:
    def sumOddLengthSubarrays(self, arr: List[int]) -> int:
        res = 0
        arr_len = len(arr)
        for i in range(arr_len):
            left_count, right_count = i, arr_len-i-1
            left_odds = (left_count+1)//2
            right_odds = (right_count+1)//2
            left_evens = left_count//2 + 1
            right_evens = right_count//2 + 1
            res += arr[i]*(left_odds*right_odds+left_evens*right_evens)
        return res
```

### 283. 移动零

这道快慢指针问题在算法专题里刷过，其核心就是慢指针记录，快指针遍历。

```python
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        slow = 0
        for fast, value in enumerate(nums):
            if value:
                nums[slow] = value
                slow += 1
        nums[slow::] = [0 for i in range(slow, len(nums))]
        return None
```

### 1672. 最富有客户的资产总量

这道题就是遍历每位客户，直接求每位客户资产和求最大值就行了。

```python
class Solution:
    def maximumWealth(self, accounts: List[List[int]]) -> int:
        return max(map(sum, accounts))
```

### 1572. 矩阵对角线元素的和

这道题注意一下矩阵行列数为奇数和偶数时的不同情况就行了。

```python
class Solution:
    def diagonalSum(self, mat: List[List[int]]) -> int:
        n = len(mat)
        res = 0
        if n&1:
            res = -mat[n//2][n//2]
        for i in range(n):
            res += mat[i][i]+mat[i][-i-1]
        return res
```

### 566. 重塑矩阵

在数据结构的专项计划里做过。

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

## 字符串

### 1768. 交替合并字符串

简单模拟。

```python
class Solution:
    def mergeAlternately(self, word1: str, word2: str) -> str:
        word1_len = len(word1)
        word2_len = len(word2)
        min_len = min(word1_len, word2_len)
        res = []
        for i in range(min_len):
            res.append(word1[i])
            res.append(word2[i])
        if min_len == word1_len:
            res.append(word2[min_len:word2_len])
        else:
            res.append(word1[min_len:word1_len])
        return ''.join(res)
```

### 1678. 设计 Goal 解析器

简单替换。

```python
class Solution:
    def interpret(self, command: str) -> str:
        command = command.replace('G','G')
        command = command.replace('()','o')
        command = command.replace('(al)','al')
        return command
```

### 389. 找不同

对两个字符串计数找不同。

```python
class Solution:
    def findTheDifference(self, s: str, t: str) -> str:
        s_dict = collections.Counter(s)
        t_dict = collections.Counter(t)
        return list(t_dict-s_dict)[0]
```

巧用 ASCII 码。

```python
class Solution:
    def findTheDifference(self, s: str, t: str) -> str:
        return chr(sum(ord(_) for _ in t)-sum(ord(_) for _ in s))
```

### 709. 转换成小写字母

Python 里面的最直接用法当然是调用 `lower()` 函数了。

```python
class Solution:
    def toLowerCase(self, s: str) -> str:
        return s.lower()
```

不过做题的话我觉得可以用 ASCII 码的方式，用 `list` 和 `str.join()` 来代替 `Java` 里类似 `StringBuffer, StringBuilder` 这样的东西，如下：

```python
class Solution:
    def toLowerCase(self, s: str) -> str:
        res = []
        for i in s:
            c_ascii = ord(i)
            if 65<=c_ascii<=90:
                res.append(chr(c_ascii+32))
            else:
                res.append(i)
        return ''.join(res)
```

当然字符串相加也行：

```python
class Solution:
    def toLowerCase(self, s: str) -> str:
        res = ''
        for i in s:
            c_ascii = ord(i)
            if 65<=c_ascii<=90:
                res += chr(c_ascii+32)
            else:
                res += i
        return res
```

### 1309. 解码字母到整数映射

依然是 ASCII 码完整数字和字母的转换，需要一个前探指针探一下后两位是不是 `#` 。

```python
class Solution:
    def freqAlphabets(self, s: str) -> str:
        res = ''
        s_len = len(s)
        i = 0
        while i<s_len:
            if (s[i] == '1' or s[i] == '2') and i+2<s_len and s[i+2]=='#':
                res += chr(int(s[i:i+2])+96)
                i += 3
            else:
                res += chr(ord(s[i])+48)
                i += 1
        return res
```

### 953. 验证外星语词典

模拟逻辑。

```python
class Solution:
    def isAlienSorted(self, words: List[str], order: str) -> bool:
        hashmap = {}
        for i,v in enumerate(order):
            hashmap[v] = i
        for i in range(len(words)-1):
            s1 = words[i]
            s2 = words[i+1]
            flag = False
            if len(s1) > len(s2):
                flag = True
                min_len = len(s2)
            else:
                min_len = len(s1)
            j = 0
            while j<min_len:
                if hashmap[s2[j]] > hashmap[s1[j]]:
                    flag = False
                    break
                elif hashmap[s2[j]] < hashmap[s1[j]]:
                    return False
                j += 1
            if flag:
                return False
        return True
```

## 链表 & 树

### 1290. 二进制链表转整数

最直观的用 `list` 存储链表每个节点的值，最后再调用进制转换的方法。

```python
class Solution:
    def getDecimalValue(self, head: ListNode) -> int:
        res = []
        while head:
            res.append(str(head.val))
            head = head.next
        return int(''.join(res), 2)
```

好像用字符串快一些。

```python
class Solution:
    def getDecimalValue(self, head: ListNode) -> int:
        res = []
        while head:
            res.append(str(head.val))
            head = head.next
        return int(''.join(res), 2)
```

递归。

```python
class Solution:
    def getDecimalValue(self, head: ListNode) -> int:
        def get_dec(node:ListNode)->tuple:
            if not node.next:
                return node.val, 0
            res, count = get_dec(node.next)
            count += 1
            res += node.val<<count
            return res, count
        return get_dec(head)[0]
```

还有就是从前到后找时，每次先把前面的和乘 2，相当于一个 n 位的二进制数，最左已经乘了 n-1 次。

```python
class Solution:
    def getDecimalValue(self, head: ListNode) -> int:
        res = 0
        while head:
            res = (res<<1) + head.val
            head = head.next
        return res
```

### 876. 链表的中间结点

在算法的学习计划里做过，就是快慢指针。

```python
class Solution:
    def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
        slow = head
        fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        return slow
```

### 104. 二叉树的最大深度

广度优先搜索。

```python
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        queue = collections.deque()
        queue.append(root)
        depth = 0
        while queue:
            for _ in range(len(queue)):
                node = queue.popleft()
                if node.left: queue.append(node.left)
                if node.right: queue.append(node.right)
            depth += 1
        return depth
```

深度优先搜索，递归。

```python
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if root:
            return max(self.maxDepth(root.left), self.maxDepth(root.right))+1
        else:
            return 0
```

### 404. 左叶子之和

广度优先搜索加一个标志位。

```python
class Solution:
    def sumOfLeftLeaves(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        queue = collections.deque()
        queue.append((root, 0))
        res = 0
        while queue:
            node, is_left = queue.popleft()
            if node.left: queue.append((node.left, 1))
            if node.right: queue.append((node.right, 0))
            if not node.left and not node.right and is_left: res += node.val
        return res
```

## 容器 & 库

### 1356. 根据数字二进制下 1 的数目排序

我们需要一个 `dict` ，以 1 的数量为 `key` ，`value` 则是一个 `list` 用来记录同 1 的数量的数字。

首先我们遍历数组可以将这个 `dict` 填充好数据。

然后对 `dict` 的 `value` 里每个 `list` 排序，处理相同 1 数量的顺序。

然后根据 `dict` 的 `key` 的顺序将 `list` 放入结果中。

```python
class Solution:
    def sortByBits(self, arr: List[int]) -> List[int]:
        def one_count(num: int)->int:
            res = 0
            while num:
                res += 1
                num &= num-1
            return res
        
        hashmap = {}
        for i in arr:
            count = one_count(i)
            if count not in hashmap:
                hashmap[count] = [i]
            else:
                hashmap[count].append(i)
        for k in hashmap.keys():
            unsort_value = hashmap[k]
            hashmap[k] = list(sorted(unsort_value))
        res = []
        for i in sorted(hashmap, key=lambda x:x):
            res += hashmap[i]
        return res
```

也可以调用 Python 的 `sorted` 函数修改排序规则。

```python
class Solution:
    def sortByBits(self, arr: List[int]) -> List[int]:     
        return sorted(arr, key=lambda x: (bin(x).count('1'), x))
```

### 232. 用栈实现队列

在数据结构学习计划里做过了，这里不再赘述。

```python
class MyQueue:

    def __init__(self):
        self.stack_in = []
        self.stack_out = []

    def push(self, x: int) -> None:
        while self.stack_out:
            self.stack_in.append(self.stack_out.pop())
        self.stack_in.append(x)
        while self.stack_in:
            self.stack_out.append(self.stack_in.pop())

    def pop(self) -> int:
        return self.stack_out.pop()

    def peek(self) -> int:
        return self.stack_out[-1]

    def empty(self) -> bool:
        return len(self.stack_out) == 0
```

### 242. 有效的字母异位词

也在数据结构学习计划里做过，这里展示一下用 `collections.Counter` 的方法，这种计数的问题其实都可以用这个方式，效果挺好。

```python
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        return collections.Counter(s)==collections.Counter(t)
```

### 217. 存在重复元素

这个也在数据结构学习计划里做过，考察是否知道 `set` 这个概念。

```py
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

