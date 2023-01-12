---
title: 「编程能力」 - 学习计划
date: 2023-01-12 23:58:45
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

## 类 & 对象

### 1603. 设计停车系统

这个应该是考察面向对象的简单设计。

```python
class ParkingSystem:

    def __init__(self, big: int, medium: int, small: int):
        self.big = big
        self.medium = medium
        self.small = small

    def addCar(self, carType: int) -> bool:
        match carType:
            case 1:
                if self.big>0:
                    self.big -= 1
                    return True
                else:
                    return False
            case 2:
                if self.medium>0:
                    self.medium -= 1
                    return True
                else:
                    return False
            case 3:
                if self.small>0:
                    self.small -= 1
                    return True
                else:
                    return False
            case _:
                return False    
```

尽量写优雅一点吧。

```python
class ParkingSystem:

    def __init__(self, big: int, medium: int, small: int):
        self.park = [big, medium, small]

    def addCar(self, carType: int) -> bool:
        if carType>len(self.park) or carType<1:
            return False
        if self.park[carType-1] > 0:
            self.park[carType-1] -= 1
            return True
        else:
            return False
```

### 303. 区域和检索 - 数组不可变

正常来讲应该是这样的。

```python
class NumArray:

    def __init__(self, nums: List[int]):
        self.nums = nums

    def sumRange(self, left: int, right: int) -> int:
        return sum(self.nums[left:right+1])
```

但是很慢，为什么呢？作为一个类来讲，查询的次数是很多的。如果每次查询都得重新累加，就会非常慢。

```python
class NumArray:

    def __init__(self, nums: List[int]):
        self.nums = [0]

        for i in nums:
            self.nums.append(self.nums[-1]+i)

    def sumRange(self, left: int, right: int) -> int:
        return self.nums[right+1] - self.nums[left]
```

这里体现出了前缀和的思想，在初始化遍历的时候就把前缀和写好，查询会非常快。

# 编程能力基础

### 896. 单调数列

我们用 `flag` 来记录数列的单调性，但未知晓数列的单调性时不将它初始化，知晓之后再初始化。

```python
class Solution:
    def isMonotonic(self, nums: List[int]) -> bool:
        flag = 0
        for i in range(len(nums)-1):
            if flag:
                if flag == 1 and nums[i]>nums[i+1]:
                    return False
                elif flag == -1 and nums[i]<nums[i+1]:
                    return False
            else:
                if nums[i]>nums[i+1]:
                    flag = -1
                elif nums[i]<nums[i+1]:
                    flag = 1
        return True
```

### 28. 找出字符串中第一个匹配项的下标

切片去匹配。

```python
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        m = len(haystack)
        n = len(needle)
        for i in range(m-n+1):
            if haystack[i:i+n] == needle:
                return i
        return -1
```

或者用双指针去比较。

```python
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        m = len(haystack)
        n = len(needle)
        for i in range(m-n+1):
            t = 0
            flag = False
            if haystack[i] == needle[t]:
                flag = True
                k = i
                while t<n:
                    if haystack[k] != needle[t]:
                        flag = False
                    t += 1
                    k += 1
            if flag:
                return i
        return -1
```

### 110. 平衡二叉树

第一种是自顶向下判断是否左右子树平衡，相当于我们从根节点开始，先查找左右子数的最大高度，差值 <= 1 则说明根节点是平衡的，再去找左子节点是否平衡，右子节点是否平衡，依次遍历完。

```python
class Solution:
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        def get_height(node):
            if not node:
                return 0
            left = get_height(node.left)
            right = get_height(node.right)
            return 1+max(left, right)

        if not root:
            return True
        left = get_height(root.left)
        right = get_height(root.right)
        if abs(left-right) > 1:
            return False
        return self.isBalanced(root.left) and self.isBalanced(root.right)
```

当然自顶向下的遍历会有很多重复的计算，实际上在求最大高度的时候可以携带一个信息，来表明是否是 AVL 树，比如如果已经不是 AVL 树就直接返回 -1 ，而不再返回当前节点的最大高度。这就叫自底向上的遍历。

```python
class Solution:
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        def get_height(node):
            if not node:
                return 0
            left = get_height(node.left)
            if left == -1:
                return -1
            right = get_height(node.right)
            if right == -1 or abs(left-right)>1:
                return -1
            return 1+max(left, right)

        if not root:
            return True
        return get_height(root)>=0
```

### 459. 重复的子字符串

遍历扫描字符串，实际上只需要扫描一半即可，因为如果扫描一半还没有发现有重复的子字符串其实就没有了。

如果左 `i` 个字符与右 `i` 个字符相等，并且字符串等于左 `i` 个字符的重复时，说明可以通过重复子字符串构成。

```python
class Solution:
    def repeatedSubstringPattern(self, s: str) -> bool:
        s_len = len(s)
        for i in range(s_len//2):
            if s_len%(i+1) == 0 and s[:i+1] == s[s_len-i-1:] and s[:i+1]*(s_len//(i+1)) == s:
                return True
        return False
```

如果字符串能有重复子字符串构成，比如像 `ababab` 这种，移除左边的 `a` 和右边的 `b` 其实也还有构成的元素 `ab` 。将字符串乘 2 ，即 `abababababab` 移除左边的 `a` 和右边的 `b` ，可以发现 `ababab` 仍然在 `bababababa` 里面。

```python
class Solution:
    def repeatedSubstringPattern(self, s: str) -> bool:
        return s in (s+s)[1:2*len(s)-1]
```

### 150. 逆波兰表达式求值

后缀表达式求值，我记得是当时学数据结构时栈的测试题，当时涉及中缀表达式转后缀再求值，除了数字栈还需要有符号栈，这里一个数字栈就够了。

```python
class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        num_stack = []
        res = 0
        for t in tokens:
            match t:
                case '+':
                    a = num_stack.pop()
                    b = num_stack.pop()
                    num_stack.append(b+a)
                case '-':
                    a = num_stack.pop()
                    b = num_stack.pop()
                    num_stack.append(b-a)
                case '*':
                    a = num_stack.pop()
                    b = num_stack.pop()
                    num_stack.append(b*a)
                case '/':
                    a = num_stack.pop()
                    b = num_stack.pop()
                    num_stack.append(int(b/a))
                case _:
                    num_stack.append(int(t))
        return num_stack.pop()
```

### 66. 加一

可以考虑先求加一再生成结果数组。

```python
class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        res = 0
        for i in digits:
            res = res*10+i
        return [int(i) for i in str(res+1)]
```

但是其实在加的过程中就可以得到答案，那就是 digits 不进位的时候。

```python
class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        for i in range(len(digits)-1, -1, -1):
            digits[i] += 1
            if digits[i] >= 10:
                digits[i] -= 10
            else:
                return digits
        return [1] + digits
```

### 1367. 二叉树中的链表

遍历树的时候存储下每个节点的路径，如果一个节点路径后 `n` 位与长度为 `n` 的链表相同，那么就满足题意，否则遍历完树之后就不存在。

```python
class Solution:
    def isSubPath(self, head: Optional[ListNode], root: Optional[TreeNode]) -> bool:
        stack = []
        stack.append((root, [root.val]))
        linked_lst = []
        linked_len = 0
        while head:
            linked_lst.append(head.val)
            linked_len += 1
            head = head.next
        while stack:
            node, path = stack.pop()
            if path[-linked_len:] == linked_lst:
                return True
            if node.left: stack.append((node.left, path+[node.left.val]))
            if node.right: stack.append((node.right, path+[node.right.val]))
        return False
```

### 43. 字符串相乘

如果转成整数形式的话：

```python
class Solution:
    def multiply(self, num1: str, num2: str) -> str:
        return str(int(num1)*int(num2))
```

或者模拟多位数乘法运算。

```python
class Solution:
    def multiply(self, num1: str, num2: str) -> str:
        res = 0
        flag1 = 1
        for i in num1[::-1]:
            adv = 0
            temp = 0
            flag2 = 1
            i = ord(i)-48
            for j in num2[::-1]:
                j = ord(j)-48
                mul = (i*j+adv)
                adv = mul//10
                temp += (mul%10)*flag2
                flag2 *= 10
            res += (temp+adv*flag2) * flag1
            flag1 *= 10
        return str(res)
```

### 67. 二进制求和

转成数求和。

```python
class Solution:
    def addBinary(self, a: str, b: str) -> str:
        return bin(int(a,2)+int(b,2))[2:]
```

模拟。

```python
class Solution:
    def addBinary(self, a: str, b: str) -> str:
        carry = False
        res = ''
        len_a = len(a)
        len_b = len(b)
        for i in range(-1, -max(len_a,len_b)-1, -1):
            if i<-len_a:
                temp = int(b[i])
            elif i<-len_b:
                temp = int(a[i])
            else:
                temp = int(a[i])+int(b[i])
            if carry:
                temp += 1
            if temp >= 2:
                carry = True
                temp -= 2
            else:
                carry = False
            res = str(temp) + res
        if carry:
            res = '1' + res
        return res
```

### 989. 数组形式的整数加法

这道题考察的是对各种情况的处理，算是一道简单的模拟。

```python
class Solution:
    def addToArrayForm(self, num: List[int], k: int) -> List[int]:
        carry = 0
        for i in range(len(num)-1, -1, -1):
            num[i] += (k%10)+carry
            if num[i] >= 10:
                num[i] -= 10
                carry = 1
            else:
                carry = 0
            k //= 10
        k += carry
        res = []
        while k:
            res.append(k%10)
            k //= 10
        return res[::-1]+num
```

### 739. 每日温度

首先上暴力，但是超时了。然后我想可能是个前缀和的问题（准确地说应该叫后缀差），然而依然没有头绪，因此从 [代码随想录](https://programmercarl.com/0739.%E6%AF%8F%E6%97%A5%E6%B8%A9%E5%BA%A6.html) 前辈这里学习了一下这种应该用地数据结构叫单调栈。简单来说就是我们需要维护一个单调递增栈（或者叫最小栈，从栈头到栈尾单调递增），这个栈其实记的是下标，每次遍历到一个元素有三种情况，对应操作如下：

- 如果遍历到的元素小于栈顶元素 - 当前元素入栈。
- 如果遍历到的元素等于栈顶元素 - 当前元素入栈。
- 如果遍历到的元素大于栈顶元素 
  - 这种情况下如果当前元素入栈会破坏栈的单调性。
  - 首先我们需要将比当前元素小的元素都出栈，在出栈时就可以说明当前元素是出栈那个元素的下一个更大值，因此可以对结果数组赋值。
  - 当前元素进栈。

```python
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        temperatures_len = len(temperatures)
        stack = [0]
        res = [0] * temperatures_len
        for i in range(1, temperatures_len):
            if temperatures[i] <= temperatures[stack[-1]]:
                stack.append(i)
            else:
                while len(stack)!=0 and temperatures[i]>temperatures[stack[-1]]:
                    res[stack[-1]] = i - stack[-1]
                    stack.pop()
                stack.append(i)
        return res
```

### 58. 最后一个单词的长度

简单的模拟计数。

```python
class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        count_start = False
        res = 0
        for i in s[::-1]:
            if i != ' ':
                count_start = True
                res += 1
            elif count_start:
                break
        return res
```

### 48. 旋转图像

两次轴对称。

```python
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        n = len(matrix)
        for c in range(1, n):
            for r in range(0, c):
                matrix[r][c], matrix[c][r] = matrix[c][r], matrix[r][c]
        for r in range(n):
            matrix[r][:] = matrix[r][::-1]
```

### 1886. 判断矩阵经轮转后是否一致

应该算是上道题的进阶版吧。判断矩阵本身和旋转 3 次有没有和目标矩阵一样就行了。

```python
class Solution:
    def findRotation(self, mat: List[List[int]], target: List[List[int]]) -> bool:
        def rotate(matrix):
            n = len(matrix)
            for c in range(1, n):
                for r in range(0, c):
                    matrix[r][c], matrix[c][r] = matrix[c][r], matrix[r][c]
            for r in range(n):
                matrix[r][:] = matrix[r][::-1]
            return matrix
        if mat == target:
            return True
        for _ in range(3):
            mat = rotate(mat)
            if mat == target:
                return True
        return False
```

### 54. 螺旋矩阵

这是一道模拟题，考察对边界的控制。

```python
class Solution:
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        m, n = len(matrix), len(matrix[0])
        printed = [[0]*n for _ in range(m)]
        r,c = 0, 0
        res = []
        direction = 'r'
        while True:
            res.append(matrix[r][c])
            printed[r][c] = 1
            match direction:
                case 'r':
                    if c+1<n and not printed[r][c+1]:
                        c += 1
                    elif r+1<m and not printed[r+1][c]:
                        direction = 'b'
                        r += 1
                    else:
                        break
                case 'b':
                    if r+1<m and not printed[r+1][c]:
                        r += 1
                    elif c>0 and not printed[r][c-1]:
                        direction = 'l'
                        c -= 1
                    else:
                        break
                case 'l':
                    if c>0 and not printed[r][c-1]:
                        c -= 1
                    elif r>0 and not printed[r-1][c]:
                        direction = 't'
                        r -= 1
                    else:
                        break
                case 't':
                    if r>0 and not printed[r-1][c]:
                        r -= 1
                    elif c+1<n and not printed[r][c+1]:
                        direction = 'r'
                        c += 1
                    else:
                        break
        return res
```

### 973. 最接近原点的 K 个点

用 `dict` 来统计某距离所有的点，然后对 `dict` 进行排序，从前往后往结果里更新点，最终当更新到 k 个点时返回答案。

```python
class Solution:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        hashmap = {}
        res = []
        for i in points:
            distance = i[0]*i[0]+i[1]*i[1]
            if distance not in hashmap:
                hashmap[distance] = [[i[0], i[1]]]
            else:
                hashmap[distance] += [[i[0], i[1]]]
        sorted_dis = sorted(hashmap)
        for i in sorted_dis:
            if len(res) < k:
                res += hashmap[i]
        return res
```

不过既然都用自带的排序了也可以直接根据距离排序后取前 k 个点。

```python
class Solution:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        return sorted(points, key=lambda x:x[0]*x[0]+x[1]*x[1])[:k]
```

### 1630. 等差子数组

暴力，对每一组取出来依次判断，判断的方法就是找到数列第一个差值，然后依次遍历看是否都等差。

```python
class Solution:
    def checkArithmeticSubarrays(self, nums: List[int], l: List[int], r: List[int]) -> List[bool]:
        res = []
        for i,j in zip(l,r):
            temp = nums[i:j+1]
            temp = sorted(temp)
            diff = temp[1]-temp[0]
            flag = True
            for k in range(j-i):
                if temp[k+1]-temp[k] != diff:
                    flag = False
                    break
            res.append(flag)
        return res
```

### 429. N 叉树的层序遍历

经典树的广度优先搜索，跟二叉树层次遍历区别不大。

```python
class Solution:
    def levelOrder(self, root: 'Node') -> List[List[int]]:
        if not root:
            return []
        queue = collections.deque()
        queue.append(root)
        res = []
        while queue:
            temp = []
            for _ in range(len(queue)):
                node = queue.popleft()
                temp.append(node.val)
                if node.children:
                    for child in node.children:
                        queue.append(child)
            res.append(temp)
        return res
```

### 503. 下一个更大元素 II

单调栈。解决循环数组的方式可以是将数组看成 2 倍长，然后从尾端开始遍历，只要遍历完成每一个数都应该找过下一个更大元素了。我们假设理论索引是 `i` ，实际索引是 `j=i%len(nums)` 代表实际应该判断的位置，然后用一个单调递增（栈顶 < 栈底）的单调栈。单调栈的思路见 [496. 下一个更大元素 I](###496. 下一个更大元素 I) 与 [739. 每日温度](###739. 每日温度) 。

```python
class Solution:
    def nextGreaterElements(self, nums: List[int]) -> List[int]:
        nums_len = len(nums)
        res = [-1] * nums_len
        stack = []
        for i in range(2*nums_len-1, -1, -1):
            j = i%nums_len
            while stack and nums[j]>=stack[-1]:
                stack.pop()
            res[j] = stack[-1] if stack else -1
            stack.append(nums[j])
        return res
```

### 556. 下一个更大元素 III

首先 `n` 如果小于 12 ，肯定是不存在满足题意的数的，直接返回 -1 。

然后将 `n` 转换为一个 `List[str]` 的列表，从后往前遍历，如果找到了某一位比它后一位的值小，记录这个索引，退出遍历，我们设这个索引为 `flag` 。当然，也可能存在找不到索引的情况，也返回 -1 。

此时下一个更大的值实际上是把 `flag` 位上的数和**在 `flag` 后面并且比他大的最小值交换**，再对 `str[flag+1:]` 从小到大排序。

```python
class Solution:
    def nextGreaterElement(self, n: int) -> int:
        if n < 12:
            return -1
        digits = [_ for _ in str(n)]
        flag = -1
        for i in range(len(digits)-1, 0, -1):
            if digits[i] > digits[i-1]:
                flag = i-1
                break
        if flag == -1:
            return -1
        temp = digits[flag:]
        target = None
        for i in range(1, len(temp)):
            if temp[i] > temp[0]:
                target = i
        temp[0], temp[target] = temp[target], temp[0]
        temp = temp[:1] + sorted(temp[1:])
        res = int(''.join(digits[:flag]+temp))
        return -1 if res>=1<<31 else res
```

### 1376. 通知所有员工所需的时间

这道题算是一个加了权的 N 叉树，或者说是一个加了权的有向无环图。

首先我们不考虑树的思想，直接使用暴力解法，就是对每一个人进行遍历，去推总裁找到他们所需时间。这个时间的最大值显然就是通知所有员工所需的时间。

```python
class Solution:
    def numOfMinutes(self, n: int, headID: int, manager: List[int], informTime: List[int]) -> int:
        direct_info = [_ for _ in informTime]
        for i in range(n):
            j = i
            temp = informTime[j]
            while manager[j] != -1:
                j = manager[j]
                temp += informTime[j]
            direct_info[i] = temp
        return max(direct_info)
```

这个代码是很慢的，因为会去计算很多重复的通知时间，即回溯过程中上层树结构都一样了，这就是自底向上的计算思路。当然也可以优化，因为实际上去统计的是底层员工的时间，因此我们只需要计算每个底层员工就行了。

```python
class Solution:
    def numOfMinutes(self, n: int, headID: int, manager: List[int], informTime: List[int]) -> int:
        direct_info = [_ for _ in informTime]
        for i in range(n):
            if informTime[i] != 0:
                continue
            j = i
            temp = informTime[j]
            while manager[j] != -1:
                j = manager[j]
                temp += informTime[j]
            direct_info[i] = temp
        return max(direct_info)
```

当然其实还能再优化，我们可以用空间换时间，开一个数组专门记录当前员工是否被搜过。

```python
class Solution:
    def numOfMinutes(self, n: int, headID: int, manager: List[int], informTime: List[int]) -> int:      
        direct_info = [0]*n
        def stats_time(i):
            if manager[i] != -1 and direct_info[i] == 0:
                direct_info[i] = informTime[manager[i]] + stats_time(manager[i])
            return direct_info[i]
        
        res = 0
        for i in range(n):
            if informTime[i] == 0:
               res = max(res, stats_time(i)) 
        return res
```

为了避免重复计算，我们在这里还可以采用自顶向下的思路，就是树的搜索，搜过了就不再搜了。这里可以考虑用一个 `dict` 来辅助记录上级与下级的关系。

```python
class Solution:
    def numOfMinutes(self, n: int, headID: int, manager: List[int], informTime: List[int]) -> int:
        hashmap = {}
        for staff, boss in enumerate(manager):
            if boss not in hashmap:
                hashmap[boss] = [staff]
            else:
                hashmap[boss] += [staff]
        stack = [(headID, informTime[headID])]
        max_time = 0
        while stack:
            p,t = stack.pop()
            max_time = max(max_time, t)
            if p in hashmap:
                for staff in hashmap[p]:
                    stack.append((staff, t+informTime[staff]))
        return max_time
```

### 49. 字母异位词分组

这道题之前在数据结构里计划里做过，核心在于创造一个合适的 `hashmap` 的 `key` 。

第一种是数组记录（因为只有 26 个小写英文字母）转元组作为 `key` 的方式。

```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        hashmap = {}
        for t in strs:
            temp = [0]*26
            for i in t:
                temp[ord(i)-97] += 1
            temp = tuple(temp)
            if temp not in hashmap:
                hashmap[temp] = [t]
            else:
                hashmap[temp] += [t]
        return [_ for _ in hashmap.values()]
```

第二种是排序的方式。

```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        hashmap = {}
        for i in strs:
            temp = ''.join(sorted([_ for _ in i]))
            if temp not in hashmap:
                hashmap[temp] = [i]
            else:
                hashmap[temp] += [i]
        return [_ for _ in hashmap.values()]
```

### 438. 找到字符串中所有字母异位词

其实这道题就是哈希计数的题，首先最直接的比较计数结果是否一样。

```python
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        p_count = collections.Counter(p)
        p_len = len(p)
        s_len = len(s)
        res = []
        for i in range(s_len-p_len+1):
            if collections.Counter(s[i:i+p_len]) == p_count:
                res.append(i)
        return res
```

这样能过，但是很慢，这里有一个显著的问题，那就是如果 `s` 的子串和 `p` 不是字母异位词，不需要统计完这个子串，因此我们需要在这里剪枝。

但是很遗憾，速度慢似乎并不是因为剪枝引起的。因为剪枝过后居然超出时间限制了，这里可以说明 `collections.Counter` 比我想象中要快很多。

```python
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        p_count = dict(collections.Counter(p))
        p_len = len(p)
        s_len = len(s)
        res = []
        for i in range(s_len-p_len+1):
            temp = deepcopy(p_count)
            flag = True
            for j in s[i:i+p_len]:
                if j not in temp or temp[j] == 0:
                    flag = False
                    break
                temp[j] -= 1
            if flag:
                res.append(i)
        return res
```

然后我考虑了滑动窗口，也是一种优化。

```python
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        p_count = collections.Counter(p)
        s_len = len(s)
        l, r = 0, len(p)
        count = collections.Counter(s[:r])
        res = []
        while r<=s_len:
            if count == p_count:
                res.append(l)
            count.subtract({s[l]:1})
            if r<s_len:
                count.subtract({s[r]:-1})
            l += 1
            r += 1
            count = +count
        return res
```

看起来不在 hash 上面做文章，这个效率是高不了了。

这里建议采用 [49. 字母异位词分组](###49. 字母异位词分组) 中数组记录（因为只有 26 个小写英文字母）转元组作为 `key` 的方式。

```python
class Solution:
    def findAnagrams(self, s: str, p: str) -> List[int]:
        s_len = len(s)
        l, r = 0, len(p)
        p_count = [0]*26
        for t in p:
            p_count[ord(t)-97] += 1
        count = [0]*26
        for t in s[l:r]:
            count[ord(t)-97] += 1
        res = []
        while r<s_len+1:
            if count == p_count:
                res.append(l)
            if r<s_len:
                count[ord(s[l])-97] -= 1
                count[ord(s[r])-97] += 1
            l += 1
            r += 1
        return res
```

### 713. 乘积小于 K 的子数组

我觉得这个都不能算滑动窗口了，应该是一个双指针问题，右指针指向子数组的右边，并且在每一轮循环中右指针的位置是不变的。左指针指向的位置是**当前右指针下满足条件的最左位置**。最后进行一个对 `k` 值的剪枝就行了。

```python
class Solution:
    def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
        if k <= 1:
            return 0
        res, left, temp = 0, 0, 1
        for right, value in enumerate(nums):
            temp *= value
            while temp>=k:
                temp //= nums[left]
                left += 1
            res += right-left+1
        return res
```

### 304. 二维区域和检索 - 矩阵不可变

经典前缀和设计加快查询求和速度。

```python
class NumMatrix:

    def __init__(self, matrix: List[List[int]]):
        m = len(matrix)
        n = len(matrix[0])
        presum = [[0]*(n+1) for _ in range(m+1)]
        for i in range(1, m+1):
            for j in range(1, n+1):
                presum[i][j] = matrix[i-1][j-1] + presum[i-1][j] + presum[i][j-1] - presum[i-1][j-1]
        self.presum = presum

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        return self.presum[row2+1][col2+1] + self.presum[row1][col1] - self.presum[row2+1][col1] - self.presum[row1][col2+1]
```

### 910. 最小差值 II

观察一下能想到这道题就是一组数大于某个值就 `-k` ，小于某个值就 `+k` ，我最开始的猜测是平均值。发现不对，至少处理不了一个数等于平均值的情况。昨天刚看了一下概率论的几个流派，按照古典派的思想，当一个事件有两种可能性但我们并不清楚的时候，它的可能性都是一半。所以取特定的值肯定与这个思想冲突很大。因此我选择排序之后逐一判断，结果过了。

```python
class Solution:
    def smallestRangeII(self, nums: List[int], k: int) -> int:
        nums = sorted(nums)
        res = nums[-1] - nums[0]
        for i in range(1, len(nums)):
            num_min = min(nums[0]+k, nums[i]-k)
            num_max = max(nums[-1]-k, nums[i-1]+k)
            res = min(num_max-num_min, res)
        return res
```

### 143. 重排链表

这道题可以用双端队列解决，先扫描链表进队，然后左出一个右出一个直至队空。

```python
class Solution:
    def reorderList(self, head: Optional[ListNode]) -> None:
        queue = collections.deque()
        while head:
            queue.append(head)
            head = head.next
        dummy = ListNode()
        node = dummy
        while queue:
            if queue:
                node.next = queue.popleft()
                node = node.next
            if queue:
                node.next = queue.pop()
                node = node.next
        node.next = None
        head = dummy.next
```

### 138. 复制带随机指针的链表

Python 中的 `deepcopy()` 。

```python
class Solution:
    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        return deepcopy(head)
```

先遍历 `head` ，同时构建新节点，同时将节点的对应关系加入 `hashmap` 中。之后再根据 `hashmap` 添加 `random` 。

```python
class Solution:
    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        dummy = head
        res = Node(0)
        node = res
        hashmap = {}
        while dummy:
            node.next = Node(dummy.val)
            node = node.next
            hashmap[dummy] = node
            dummy = dummy.next
        dummy = head
        node = res.next
        while dummy:
            node.random = hashmap[dummy.random] if dummy.random else None
            dummy = dummy.next
            node = node.next
        return res.next
```

### 2. 两数相加

在数据结构里做过。

```python
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode()
        node = dummy
        carry = 0
        while l1 or l2:
            temp = carry
            if l1:
                temp += l1.val
                l1 = l1.next
            if l2:
                temp += l2.val
                l2 = l2.next
            carry = temp // 10
            temp %= 10
            node.next = ListNode(temp)
            node = node.next
        if carry:
            node.next = ListNode(1)
        return dummy.next
```

### 445. 两数相加 II

可以采用的一种思路就是先对链表取数相加再构建新链表。

```python
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        num1 = 0
        while l1:
            num1 = num1*10 + l1.val
            l1 = l1.next
        num2 = 0
        while l2:
            num2 = num2*10 + l2.val
            l2 = l2.next
        dummy = ListNode()
        node = dummy
        for i in str(num1+num2):
            node.next = ListNode(int(i))
            node = node.next
        return dummy.next
```

### 61. 旋转链表

用一个 `list` 记录每个节点位置，再相应操作。

```python
class Solution:
    def rotateRight(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        if not head:
            return None
        lst = []
        while head:
            lst.append(head)
            head = head.next
        k %= len(lst)
        if k == 0:
            return lst[0]
        node_1 = lst[-k-1]
        node_2 = lst[-k]
        node_3 = lst[-1]
        node_4 = lst[0]
        node_1.next = None
        node_3.next = node_4
        return node_2
```

### 173. 二叉搜索树迭代器

数据结构里做过。首先将搜索树结果中序遍历，之后再创建迭代器输出。

```python
class BSTIterator:

    def __init__(self, root: Optional[TreeNode]):
        nums = []
        stack = []
        while stack or root:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            nums.append(root.val)
            root = root.right
        self.index = 0
        self.nums = nums
        self.len = len(nums)

    def next(self) -> int:
        res = self.nums[self.index]
        self.index += 1
        return res

    def hasNext(self) -> bool:
        if self.index<self.len:
            return True
        else:
            return False
```

### 1845. 座位预约管理系统

可以采用单调栈解决问题，预约的过程是出栈，解除预约是进栈。

```python
class SeatManager:

    def __init__(self, n: int):
        self.stack = [i for i in range(n, 0, -1)]

    def reserve(self) -> int:
        return self.stack.pop()

    def unreserve(self, seatNumber: int) -> None:
        temp = []
        while self.stack and seatNumber>self.stack[-1]:
            temp.append(self.stack.pop())
        self.stack.append(seatNumber)
        while temp:
            self.stack.append(temp.pop())
```

另一个需要用到堆队列算法，可以参考 [heapq — Heap queue algorithm — Python 3.11.1 documentation](https://docs.python.org/3/library/heapq.html) 。

```python
class SeatManager:

    def __init__(self, n: int):
        self.heap = list(range(1,n+1))

    def reserve(self) -> int:
        return heappop(self.heap)

    def unreserve(self, seatNumber: int) -> None:
        heappush(self.heap, seatNumber)
```

### 860. 柠檬水找零

利用 `dict` 做一个对零钱的统计模拟。

```python
class Solution:
    def lemonadeChange(self, bills: List[int]) -> bool:
        hashmap = {
            5:0,
            10:0
        }
        for bill in bills:
            match bill:
                case 5:
                    hashmap[5] += 1
                case 10:
                    hashmap[10] += 1
                    if hashmap[5] > 0:
                        hashmap[5] -= 1
                    else:
                        return False
                case 20:
                    if hashmap[10] > 0:
                        hashmap[10] -= 1
                    elif hashmap[5] > 1:
                        hashmap[5] -= 2
                    else:
                        return False
                    if hashmap[5] > 0:
                        hashmap[5] -= 1
                    else:
                        return False
        return True
```

### 155. 最小栈

数据结构里做过，用两个栈解决问题，一个存顺序，一个存最小值。

```python
class MinStack:

    def __init__(self):
        self.stack = []
        self.minstack = []

    def push(self, val: int) -> None:
        if self.minstack:
            _min = self.minstack[-1]
            _min = min(val, _min)
            self.minstack.append(_min)
            self.stack.append(val)
        else:
            self.minstack.append(val)
            self.stack.append(val)

    def pop(self) -> None:
        self.stack.pop()
        self.minstack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.minstack[-1]
```

### 341. 扁平化嵌套列表迭代器

深度优先搜索，但是重点在于我们必须在 `hasNext()` 里去维护栈，否则结果可能会有无效空值，比如 `[[]]` 这种输入。

```python
class NestedIterator:
    def __init__(self, nestedList: [NestedInteger]):
        self.stack = [t for t in nestedList][::-1]
    
    def next(self) -> int:
        return self.stack.pop().getInteger()

    
    def hasNext(self) -> bool:
        while self.stack:
            node = self.stack.pop()
            if node.isInteger():
                self.stack.append(node)
                return True
            else:
                self.stack += node.getList()[::-1]
                return self.hasNext()
        return False
```

### 1797. 设计一个验证系统

重点在于题目中这句，过期事件**优先于**其他操作。 `dict` 哈希实现。

```python
class AuthenticationManager:

    def __init__(self, timeToLive: int):
        self.timeToLive = timeToLive
        self.hashmap = {}

    def generate(self, tokenId: str, currentTime: int) -> None:
        self.hashmap[tokenId]=self.timeToLive+currentTime

    def renew(self, tokenId: str, currentTime: int) -> None:
        if tokenId not in self.hashmap or self.hashmap[tokenId] <= currentTime:
            return None
        self.hashmap[tokenId]=self.timeToLive+currentTime

    def countUnexpiredTokens(self, currentTime: int) -> int:
        res = 0
        for k,v in self.hashmap.items():
            if v > currentTime:
                res += 1
        return res
```

### 707. 设计链表

数据结构里做过。

```python
class MyLinkedList:

    def __init__(self):
        self.len = 0
        self.nums = []

    def get(self, index: int) -> int:
        if index < 0 or index >= self.len:
            return -1
        return self.nums[index]

    def addAtHead(self, val: int) -> None:
        self.addAtIndex(0, val)

    def addAtTail(self, val: int) -> None:
        self.addAtIndex(self.len, val)

    def addAtIndex(self, index: int, val: int) -> None:
        if index > self.len:
            return None
        index = max(index, 0)
        self.len += 1
        self.nums.insert(index, val)

    def deleteAtIndex(self, index: int) -> None:
        if index < 0 or index >= self.len:
            return None
        self.len -= 1
        del self.nums[index]
```

### 380. O(1) 时间插入、删除和获取随机元素

用 `set` , `random` 实现插入、删除和随机获取的逻辑。

```python
class RandomizedSet:

    def __init__(self):
        self.set = set('')

    def insert(self, val: int) -> bool:
        if val not in self.set:
            self.set.add(val)
            return True
        else:
            return False

    def remove(self, val: int) -> bool:
        if val in self.set:
            self.set.remove(val)
            return True
        else:
            return False

    def getRandom(self) -> int:
        return random.choice(list(self.set))
```

### 622. 设计循环队列

用 `list` 模拟队列。

```python
class MyCircularQueue:

    def __init__(self, k: int):
        self.queue = []
        self.max_len = k
        self.len = 0

    def enQueue(self, value: int) -> bool:
        if self.len == self.max_len:
            return False
        self.queue.append(value)
        self.len += 1
        return True

    def deQueue(self) -> bool:
        if self.len == 0:
            return False
        self.queue.pop(0)
        self.len -= 1
        return True

    def Front(self) -> int:
        if self.len == 0:
            return -1
        return self.queue[0]

    def Rear(self) -> int:
        if self.len == 0:
            return -1
        return self.queue[-1]

    def isEmpty(self) -> bool:
        return self.len == 0

    def isFull(self) -> bool:
        return self.len == self.max_len
```

使用 `deque` 双端队列。

```python
class MyCircularQueue:

    def __init__(self, k: int):
        self.queue = collections.deque()
        self.max_len = k

    def enQueue(self, value: int) -> bool:
        if len(self.queue) == self.max_len:
            return False
        self.queue.append(value)
        return True

    def deQueue(self) -> bool:
        if len(self.queue) == 0:
            return False
        self.queue.popleft()
        return True

    def Front(self) -> int:
        if len(self.queue) == 0:
            return -1
        res = self.queue.popleft()
        self.queue.appendleft(res)
        return res

    def Rear(self) -> int:
        if len(self.queue) == 0:
            return -1
        res = self.queue.pop()
        self.queue.append(res)
        return res

    def isEmpty(self) -> bool:
        return len(self.queue) == 0

    def isFull(self) -> bool:
        return len(self.queue) == self.max_len
```

### 729. 我的日程安排表 I

用一个 `list` 里，存储元组进行简单模拟。

```python
class MyCalendar:

    def __init__(self):
        self.calendar = []

    def book(self, start: int, end: int) -> bool:
        flag = True
        for i in self.calendar:
            if start < i[1] and end > i[0]:
                flag = False
                break
        if flag:
            self.calendar.append((start, end))
        return flag
```

