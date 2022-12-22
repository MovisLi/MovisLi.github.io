---
title: 「数据结构」 - 学习计划 
date: 2022-12-23 02:45:41
categories: [ComputerScience, Algorithm, LeetCode]
tags: [python, array, tree, linked list, stack, queue]
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

## 字符串

### 387. 字符串中的第一个唯一字符

`dict` 存储频次。

```python
class Solution:
    def firstUniqChar(self, s: str) -> int:
        hashmap = {}
        for i in s:
            if i not in hashmap:
                hashmap[i] = 1
            else:
                hashmap[i] += 1
        for i,v in enumerate(s):
            if hashmap[v] == 1:
                return i
        return -1
```

`dict` 存储唯一元素的下标，否则存储 -1。

```python
class Solution:
    def firstUniqChar(self, s: str) -> int:
        hashmap = {}
        for i,v in enumerate(s):
            if v not in hashmap:
                hashmap[v] = i
            else:
                hashmap[v] = -1
        res = len(s)
        for i in hashmap.values():
            if i != -1 and i < res:
                res = i
        if res == len(s):
            return -1
        return res
```

### 383. 赎金信

用 `dict` 做一个简单的计数，再判断。

```python
class Solution:
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        magazine_dict = {}
        ransomNote_dict = {}
        for i in magazine:
            if i not in magazine_dict:
                magazine_dict[i] = 1
            else:
                magazine_dict[i] += 1
        for i in ransomNote:
            if i not in ransomNote_dict:
                ransomNote_dict[i] = 1
            else:
                ransomNote_dict[i] += 1
        for k,v in ransomNote_dict.items():
            if k not in magazine_dict:
                return False
            else:
                if v > magazine_dict[k]:
                    return False
        return True
```

### 242. 有效的字母异位词

用 `dict` 做一个简单的计数统计，再判断。

```python
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        hashmap = {}
        for i in s:
            if i not in hashmap:
                hashmap[i] = 1
            else:
                hashmap[i] += 1
        for i in t:
            if i not in hashmap:
                return False
            else:
                hashmap[i] -= 1
        for v in hashmap.values():
            if v != 0:
                return False
        return True
```

## 链表

### 141. 环形链表

快慢双指针能否相遇的问题，如果相遇了就是有环，没有相遇就是没有环。

```python
class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        fast = slow = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if fast == slow:
                return True
        return False
```

### 21. 合并两个有序链表

迭代，链表的修改需要找到待插入节点的上一个节点，因此，我们需要定义一个 `pre` 节点方便修改，同时为了方便返回结果，我们还需要一个 `dummy` 节点用于记录一开始的位置。

```python
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode()
        pre = dummy
        while list1 and list2:
            if list1.val <= list2.val:
                pre.next = list1
                list1 = list1.next
            else:
                pre.next = list2
                list2 = list2.next
            pre = pre.next
        pre.next = list2 if list1 is None else list1
        return dummy.next
```

递归。

```python
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        if list1 is None:
            return list2
        elif list2 is None:
            return list1
        elif list1.val <= list2.val:
            list1.next = self.mergeTwoLists(list1.next, list2)
            return list1
        else:
            list2.next = self.mergeTwoLists(list1,list2.next)
            return list2
```

### 203. 移除链表元素

迭代，同样一个 `pre` 节点方便删元素，一个 `dummy` 节点方便返回头指针。

```python
class Solution:
    def removeElements(self, head: Optional[ListNode], val: int) -> Optional[ListNode]:
        dummy = ListNode()
        pre = dummy
        while head:
            while head and head.val == val:
                head = head.next
            if head:
                pre.next = head
                head = head.next
            else:
                pre.next = None
            pre = pre.next
        return dummy.next
```

虽然过了，但是我仔细一看上面的代码其实逻辑有问题，最外层的 `while head` 根本不应该写成循环，因为推动循环变化的是 `pre` 这个点，优化一下：

```python
class Solution:
    def removeElements(self, head: Optional[ListNode], val: int) -> Optional[ListNode]:
        dummy = ListNode(0, head)
        pre = dummy
        while pre.next:
            if pre.next.val != val:
                pre = pre.next
            else:
                pre.next = pre.next.next
        return dummy.next
```

递归。这里也注意递归条件。

```python
class Solution:
    def removeElements(self, head: Optional[ListNode], val: int) -> Optional[ListNode]:
        if head is None:
            return head
        head.next = self.removeElements(head.next, val)
        return head if head.val != val else head.next
```

### 206. 反转链表

用栈进行反转。

```python
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        stack = []
        while head:
            stack.append(head.val)
            head = head.next
        dummy = ListNode()
        node = dummy
        while stack:
            node.next = ListNode(stack.pop())
            node = node.next
        return dummy.next
```

但是这个都把值提出来了，不太正规。

我们可以使用双指针迭代，一个指针在前一个指针在后，遍历链表时修改指针的方向。

```python
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        pre = None
        cur = head
        while cur:
            temp = cur.next
            cur.next = pre
            pre = cur
            cur = temp
        return pre
```

还可以用递归，其中递归是参考的 [206. 反转链表 - 力扣（Leetcode）](https://leetcode.cn/problems/reverse-linked-list/solutions/36710/dong-hua-yan-shi-206-fan-zhuan-lian-biao-by-user74/) 。

```python
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if head is None or head.next is None:
            return head
        cur = self.reverseList(head.next)
        head.next.next = head
        head.next = None
        return cur
```

### 83. 删除排序链表中的重复元素

可以使用 `set` 只保留重复元素。

```python
        hashset = set('')
        dummy = ListNode()
        pre = dummy
        while head:
            if head.val not in hashset:
                hashset.add(head.val)
                pre.next = head
                pre = pre.next
            head = head.next
        pre.next = None
        return dummy.next
```

不过，因为这道题是已经排序的链表，也就是说重复元素都是连着出现的，所以也可以这样，根据下一个元素是否重复来决定是移动指针还是插值：

```python
class Solution:
    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if head is None:
            return head
        node = head
        while node.next:
            if node.val == node.next.val:
                node.next = node.next.next
            else:
                node = node.next
        return head
```

## 栈 / 队列

### 20. 有效的括号

用栈处理，左符号进栈右符号出栈匹配，出现任何错误或者最后栈非空都是 `False` ，否则是 `True` 。

```python
class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        for i in s:
            match i:
                case '(': stack.append(i)
                case ')': 
                    if not stack or stack.pop() != '(':
                        return False
                case '{': stack.append(i)
                case '}':
                    if not stack or stack.pop() != '{':
                        return False
                case '[': stack.append(i)
                case ']':
                    if not stack or stack.pop() != '[':
                        return False
        return not stack
```

### 232. 用栈实现队列

我的做法是有两个栈分别叫 `stack_in` 和 `stack_out` ，`stack_in` 负责用于转换数据的顺序（因为队列和栈顺序是反着的），`stack_out` 存着正确的出队，查看队顶，查看队空的顺序。

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

## 树

### 144. 二叉树的前序遍历

二叉树的遍历顺序（前中后）指的是根节点遍历是在前中后哪个位置，比如这道题，前序，根节点在前。

递归。

```python
class Solution:
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        res = []
        if not root:
            return res

        def NLR(node:Optional[TreeNode]):
            if not node:
                return None
            res.append(node.val)
            NLR(node.left)
            NLR(node.right)
            
        NLR(root)
        return res
```

非递归，类似深度优先搜索。

```python
class Solution:
    def preorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        res = []
        if not root:
            return res
        stack = [root]
        while stack:
            node = stack.pop()
            res.append(node.val)
            if node.right: stack.append(node.right)
            if node.left: stack.append(node.left)
        return res
```

### 94. 二叉树的中序遍历

递归。

```python
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        res = []
        if not root:
            return res

        def LNR(node:Optional[TreeNode]):
            if not node:
                return None
            LNR(node.left)
            res.append(node.val)
            LNR(node.right)
        
        LNR(root)
        return res
```

非递归的方式，我们可以看上面递归的方法，其实在打印之前，是不对右节点进行处理的，也就是说是针对每一次循环都是先处理左节点到打印，想明白这件事，我们对栈的变化情况其实心里就有数了。

值得注意的是这个中序遍历其实用的是指针去遍历，而不是用栈遍历。

```python
class Solution:
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        res = []
        if not root:
            return res
        stack = []
        while stack or root:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            res.append(root.val)
            root = root.right
        return res
```

### 145. 二叉树的后序遍历

递归。

```python
class Solution:
    def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        res = []
        if not root:
            return res

        def LRN(node:Optional[TreeNode]):
            if not node:
                return None
            LRN(node.left)
            LRN(node.right)
            res.append(node.val)

        LRN(root)
        return res
```

这个有点骚，已知前序遍历是 NLR，后续遍历是 LRN，我们可以将前序遍历魔改一下成为，NRL，再反转一下结果就变成了 LRN。

```python
class Solution:
    def postorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        res = []
        if not root:
            return res
        stack = [root]
        while stack:
            node = stack.pop()
            res.append(node.val)
            if node.left: stack.append(node.left)
            if node.right: stack.append(node.right)
        return res[::-1]
```

### 102. 二叉树的层序遍历

经典广度优先搜索。

```python
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        queue = collections.deque()
        res = []
        queue.append(root)
        while queue:
            temp = []
            for _ in range(len(queue)):
                node = queue.popleft()
                temp.append(node.val)
                if node.left: queue.append(node.left)
                if node.right: queue.append(node.right)
            res.append(temp)
        return res
```

### 104. 二叉树的最大深度

昨天刚在编程能力学习计划里做了。递归。

```python
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if root:
            return max(self.maxDepth(root.left), self.maxDepth(root.right))+1
        else:
            return 0
```

广度优先搜索。

```python
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        res = 0
        queue = collections.deque()
        queue.append(root)
        while queue:
            res += 1
            for _ in range(len(queue)):
                node = queue.popleft()
                if node.left: queue.append(node.left)
                if node.right: queue.append(node.right)
        return res
```

### 101. 对称二叉树

我们可以用二叉树层次遍历的思想进行广度优先搜索。每一层都必须是对称的否则直接不是对称二叉树。同时为了处理空节点的情况，即使空节点也要添加值 `None` 。

```python
class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        queue = collections.deque()
        queue.append(root)
        while queue:
            temp = []
            for i in range(len(queue)):
                node = queue.popleft()
                if node.left:
                    temp.append(node.left.val)
                    queue.append(node.left)
                else:
                    temp.append(None)
                if node.right:
                    temp.append(node.right.val)
                    queue.append(node.right)
                else:
                    temp.append(None)
            if temp != temp[::-1]:
                return False
        return True
```

递归，这里主要考虑终止递归的条件：

- 左节点和右节点都为空 - 对称，返回 `True` 。
- 左右节点空了一个 - 非对称，返回 `False` 。
- 左右节点都非空但值不相等 - 非对称，返回 `False` 。
- 左右节点都非空且值相等 - 需要递归比较（左节点的左子节点，右节点的右子节点）和（左节点的右子节点，右节点的左子节点）。

```python
class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        def DFS(left, right):
            if not (left or right):
                return True
            if not (left and right):
                return False
            if left.val != right.val:
                return False
            return DFS(left.left, right.right) and DFS(left.right, right.left)
        return DFS(root.left, root.right)
```

