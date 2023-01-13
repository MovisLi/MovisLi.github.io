---
title: 「数据结构」 - 学习计划 
date: 2023-01-13 17:19:41
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

### 226. 翻转二叉树

递归翻转。

```python
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:
            return None
        root.left, root.right = self.invertTree(root.right), self.invertTree(root.left)
        return root
```

层次反转，广度优先搜索的思想：

```python
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:
            return None
        queue = collections.deque()
        queue.append(root)
        while queue:
            node = queue.popleft()
            node.left, node.right = node.right, node.left
            if node.left: queue.append(node.left)
            if node.right: queue.append(node.right)
        return root
```

深度优先搜索的思想：

```python
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:
            return None
        stack = [root]
        while stack:
            node = stack.pop()
            node.left, node.right = node.right, node.left
            if node.left: stack.append(node.left)
            if node.right: stack.append(node.right)
        return root
```

### 112. 路径总和

深度优先搜索，非递归方法。实质上就是每次往栈里填节点的时候填上当前路径的和，当节点为叶子节点的时候可以看一下路径和是否等于目标值。

```python
class Solution:
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        if not root:
            return False
        stack = [(root, root.val)]
        while stack:
            node, temp = stack.pop()
            if node.right: stack.append((node.right, temp+node.right.val))
            if node.left: stack.append((node.left, temp+node.left.val))
            if not node.left and not node.right and temp == targetSum:
                return True
        return False
```

广度优先搜索也能解决。

```python
class Solution:
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        if not root:
            return False
        queue = collections.deque()
        queue.append((root, root.val))
        while queue:
            node, temp = queue.popleft()
            if node.right: queue.append((node.right, temp+node.right.val))
            if node.left: queue.append((node.left, temp+node.left.val))
            if not node.left and not node.right and temp == targetSum:
                return True
        return False
```

递归，递归似乎比上面两种快很多。

```python
class Solution:
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        if not root:
            return False
        if root.left and root.right:
            return self.hasPathSum(root.left, targetSum-root.val) or self.hasPathSum(root.right, targetSum-root.val)
        if root.left:
            return self.hasPathSum(root.left, targetSum-root.val)
        if root.right:
            return self.hasPathSum(root.right, targetSum-root.val)
        return root.val == targetSum
```

### 700. 二叉搜索树中的搜索

二叉搜索树的重要特征就是左子树的所有值 <= 根节点的值 <= 右子树的所有值，因此我们可以通过值的比较快速定位到目标值的节点。

递归。

```python
class Solution:
    def searchBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        if root.val == val:
            return root
        if root.val < val and root.right:
            return self.searchBST(root.right, val)
        if root.val > val and root.left:
            return self.searchBST(root.left, val)
        return None
```

非递归。

```python
class Solution:
    def searchBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        while root:
            if root.val == val:
                return root
            elif root.val < val:
                root = root.right
            elif root.val > val:
                root = root.left
        return None
```

### 701. 二叉搜索树中的插入操作

在不考虑树的深度的情况下插入新节点到二叉搜索树中还是非常容易的，与上题相似。

递归。

```python
class Solution:
    def insertIntoBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        if not root:
            return TreeNode(val)
        if root.val > val:
            if root.left:
                root.left = self.insertIntoBST(root.left, val)
            else:
                root.left = TreeNode(val)
        if root.val < val:
            if root.right:
                root.right = self.insertIntoBST(root.right, val)
            else:
                root.right = TreeNode(val)
        return root
```

非递归，在循环里插入：

```python
class Solution:
    def insertIntoBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        if not root:
            return TreeNode(val)
        dummy = root
        while True:
            if root.val > val:
                if root.left:
                    root = root.left
                else:
                    root.left = TreeNode(val)
                    break
            elif root.val < val:
                if root.right:
                    root = root.right
                else:
                    root.right = TreeNode(val)
                    break
        return dummy
```

非递归，在循环外插入：

```python
class Solution:
    def insertIntoBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        if not root:
            return TreeNode(val)
        dummy = root
        while True:
            if root.val > val and root.left:
                root = root.left
            elif root.val < val and root.right:
                root = root.right
            else:
                break
        if root.val > val:
            root.left = TreeNode(val)
        else:
            root.right = TreeNode(val)
        return dummy
```

### 98. 验证二叉搜索树

自定义一个递归方法去设置上下限。

```python
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def DST(root, min_val, max_val):
            if root.left and root.right:
                if min_val < root.left.val < root.val < root.right.val < max_val:
                    return DST(root.left, min_val, root.val) and DST(root.right, root.val, max_val)
                else:
                    return False
            if root.left:
                if min_val < root.left.val < root.val < max_val:
                    return DST(root.left, min_val, root.val)
                else:
                    return False
            if root.right:
                if min_val < root.val < root.right.val < max_val:
                    return DST(root.right, root.val, max_val)
                else:
                    return False
            return True
        return DST(root, -1<<32, 1<<32)
```

也可以中序遍历完了再做比较。

```python
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        res = []
        def LNR(node):
            if not node:
                return None
            LNR(node.left)
            res.append(node.val)
            LNR(node.right)
        LNR(root)
        for i in range(len(res)-1):
            if res[i] >= res[i+1]:
                return False
        return True
```

对于中序遍历来讲，可以边遍历边比较会更快。

```python
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        pre = float('-inf')
        stack = []
        while stack or root:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            if root.val <= pre:
                return False
            pre = root.val
            root = root.right
        return True
```

### 653. 两数之和 IV - 输入二叉搜索树

我们可以用 hash 加遍历树的方式来寻找有无两数之和。非递归。

```python
class Solution:
    def findTarget(self, root: Optional[TreeNode], k: int) -> bool:
        hashset = set('')
        stack = [root]
        while stack:
            node = stack.pop()
            if node.val in hashset:
                return True
            hashset.add(k-node.val)
            if node.left: stack.append(node.left)
            if node.right: stack.append(node.right)
        return False
```

递归。

```python
class Solution:
    def findTarget(self, root: Optional[TreeNode], k: int) -> bool:
        hashset = set('')
        def DFS(node):
            if not node:
                return False
            if node.val in hashset:
                return True
            hashset.add(k-node.val)
            return DFS(node.left) or DFS(node.right) 
        return DFS(root)
```

### 235. 二叉搜索树的最近公共祖先

上道题感觉其实与二叉搜索树没什么关系，这道是真有。

我们先不管这个二叉搜索树的性质，使用树的层次遍历并开一个祖先列表，将一个节点所有祖先都放入列表中，直到找到 `p` ， `q` 两个节点然后再从后往前去比较两个祖先列表。

```python
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        queue = collections.deque()
        queue.append((root,[root]))
        p_ancestors = None
        q_ancestors = None
        while queue:
            node, ancestors = queue.pop()
            if node == p:
                p_ancestors = ancestors
            if node == q:
                q_ancestors = ancestors
            if p_ancestors and q_ancestors:
                break
            if node.left: queue.append((node.left, ancestors+[node.left]))
            if node.right: queue.append((node.right, ancestors+[node.right]))
        for p_ancestor in reversed(p_ancestors):
            for q_ancestor in reversed(q_ancestors):
                if p_ancestor == q_ancestor:
                    return p_ancestor
        return None
```

这个效率确实低了很多。

我们可以利用二叉搜索树的性质，如果两个节点的值都比某个节点值大，这两个节点都应该在这个节点右边。如果两个节点值都比某个节点值小，这两个节点都应该在某个节点左边。否则这个节点就是两个节点的最近公共祖先。

```python
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if p.val < root.val and q.val < root.val:
            return self.lowestCommonAncestor(root.left, p, q)
        if p.val > root.val and q.val > root.val:
            return self.lowestCommonAncestor(root.right, p, q)
        return root
```

# 数据结构基础

## 数组

### 136. 只出现一次的数字

异或运算的性质。

```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        return reduce(lambda x,y:x^y, nums)
```

### 169. 多数元素

对元素计数，找出大于 `len(nums)//2` 的元素。

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        nums_len_half = len(nums)//2
        counter = collections.Counter(nums)
        for k in counter:
            if counter[k] > nums_len_half:
                return k
        return None
```

### 15. 三数之和

强行三数之和转两数之和（ hash ）+ 去重。

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        def twoSum(nums, target, begin_index):
            hashset = set('')
            res = []
            for i in range(begin_index, len(nums)):
                temp = target-nums[i]
                if temp in hashset:
                    res.append([nums[i],temp])
                else:
                    hashset.add(nums[i])
            return res
        
        res = []
        zero = False
        for i,v in enumerate(nums):
            two = twoSum(nums, -v, i+1)
            if len(two) != 0:
                for j in two:
                    if v==0 and j[0]==0 and j[1]==0:
                        zero = True
                        continue
                    flag = True
                    for r in res:
                        if v in r and j[0] in r and j[1] in r:
                            flag = False
                    if flag:
                        res.append([v]+j)
        if zero:
            res.append([0, 0, 0])
        return res
```

太慢了，我甚至怀疑不是 Python 都过不了测试用例：

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202212260937219.png)

双指针法，见注释。

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums = sorted(nums)	# 靠排序达到去重效果
        nums_len = len(nums)
        res = []
        if nums_len < 3 or nums[0]>0 or nums[-1]<0:
            return res
        for i in range(nums_len-2):	# 寻找元素 a
            if nums[i]>0:	# 剪枝
                break
            if i>0 and nums[i] == nums[i-1]:	# 对元素 a 去重
                continue
            left = i+1			# 寻找元素 b
            right = nums_len-1	# 寻找元素 c
            while left<right:
                temp_sum = nums[i]+nums[left]+nums[right]	# 这里和下面的 if-elif-else 语句都在寻找三数之和为 0
                if temp_sum > 0:
                    right -= 1
                elif temp_sum < 0:
                    left += 1
                else:
                    res.append([nums[i], nums[left], nums[right]])
                    while left<right and nums[left] == nums[left+1]: left += 1		# 对元素 b 去重
                    while left<right and nums[right] == nums[right-1]: right -= 1	# 对元素 c 去重
                    left += 1
                    right -= 1
        return res
```

### 75. 颜色分类

这道题是一道盲点问题，其实不需要排序，只需要统计，统计完之后根据统计结果直接替换就行了。

```python
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        counter = collections.Counter(nums)
        index = 0
        for i in (0, 1, 2):
            for _ in range(counter[i]):
                nums[index] = i
                index += 1
```

### 56. 合并区间

暴力法，用一个 `record` 数组记录有数的位置，这种思路的问题在于要处理 `[1,1], [2,2]` 与 `[1,2], [2,2]` 的区别，因为在 `record` 数组上的记录都是 `record[1]=1, record[2]=1` 。这里采取的方式是将 `record` 数组扩大两倍，这样 `[1,1], [2,2]` 就表现为 `record[2]=1, record[4]=1` ，而 `[1,2], [2,2]` 则表现为 `record[2]=1, record[3]=1, record[4]=1` ，得以区分。

```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        res = []
        record = [0] * 20002
        for i in intervals:
            for j in range(i[0]*2, i[1]*2+1):
                record[j] = 1
        i = 0
        while i<=20001:
            if record[i] == 0:
                i += 1
            else:
                pre = i+1
                while pre<=20001 and record[pre]:
                    pre += 1
                res.append([i//2, pre//2])
                i = pre
        return res
```

贪心。首先需要将 `intervals` 数组按 `i[0]` 排序，排序之后进行遍历，这时会有两种情况：

- 当前遍历的 `t[0]` 大于结果数组 `res` 最后一个 `i[1]` 的值，代表从现在开始就不连续了，直接将 `t` 加入结果数组。
- 当前遍历的 `t[0]` 小于等于结果数组 `res` 最后一个 `i[1]` 的值，也就是连续，这时候应该对 `res[-1]` 和 `t` 取并集，不过由于已经排了序，所以其实就是 `res[-1][1]` 右边边界取 `t[1]` 和 `res[-1][1]` 的更大值。

```python
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        arr_sort = sorted(intervals, key=lambda x: x[0])
        res = [arr_sort[0]]
        for t in arr_sort:
            if t[0]>res[-1][1]:
                res.append(t)
            else:
                res[-1][1] = max(res[-1][1], t[1])
        return res
```

### 706. 设计哈希映射

调用 `dict` 。

```python
class MyHashMap:

    def __init__(self):
        self.hashmap = {}

    def put(self, key: int, value: int) -> None:
        self.hashmap[key] = value

    def get(self, key: int) -> int:
        if key not in self.hashmap:
            return -1
        else:
            return self.hashmap[key]

    def remove(self, key: int) -> None:
        if key in self.hashmap:
            del self.hashmap[key]
```

### 119. 杨辉三角 II

与构建杨辉三角很类似，直接沿用构建杨辉三角的思路，取结果数组里最后一行的数组就行了。

```python
class Solution:
    def getRow(self, rowIndex: int) -> List[int]:
        if rowIndex == 0:
            return [1]
        res = [[1]*(row+1) for row in range(rowIndex+1)]
        for row in range(2, rowIndex+1):
            for col in range(1, row):
                res[row][col] = res[row-1][col-1]+res[row-1][col]
        return res[-1]
```

### 48. 旋转图像

先沿右对角线做一次轴对称，再沿中线做一次轴对称。

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

### 59. 螺旋矩阵 II

模拟过程：

```python
class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        res = [[1]*n for _ in range(n)]
        direction = ('t', 'r', 'b', 'l')
        direct = ['r']*(n-1) + ['b']*(n-1) + ['l']*(n-1)
        index = 0
        while n>1:
            for _ in range(2):
                direct += direction[index]*(n-2)
                index = (index+1)%4
            n -= 1
        count = 2
        r = 0
        c = 0
        for d in direct:
            match d:
                case 'r':
                    c += 1
                case 'b':
                    r += 1
                case 'l':
                    c -= 1
                case 't':
                    r -= 1
            res[r][c] = count
            count += 1
        return res
```

### 240. 搜索二维矩阵 II

直接使用暴力法：

```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        m = len(matrix)
        n = len(matrix[0])
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == target:
                    return True
        return False
```

？？？打败了 90 % 的人？

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202212291126962.png)

说实话我肯定不能接受。这题肯定跟二分有关系的。

对每一行都使用二分查找搜索。

```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        m = len(matrix)
        n = len(matrix[0])
        for i in range(m):
            l, r = 0, n-1
            while l<=r:
                mid = (l+r) // 2
                if matrix[i][mid] == target:
                    return True
                elif matrix[i][mid] < target:
                    l = mid + 1
                else:
                    r = mid - 1
        return False
```

从右上角或者左下角开始搜，思路差不多。以右上角开始为例，可以发现向左值都在减小，向下值都在增加，所以可以利用这个性质搜。

```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        m = len(matrix)
        n = len(matrix[0])
        i = 0
        j = n-1
        while i<m and j>=0:
            if matrix[i][j] == target:
                return True
            elif matrix[i][j] > target:
                j -= 1
            else:
                i += 1
        return False
```

### 435. 无重叠区间

先排序之后用贪心的思想过滤每个区间。

```python
class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        intervals = sorted(intervals, key=lambda x:x[0])
        bound = -1e5
        res = 0
        for t in intervals:
            if t[0] < bound:
                res += 1
                bound = min(bound, t[1])
            else:
                bound = t[1]
        return res
```

### 334. 递增的三元子序列

首先使用暴力，不幸地超出时间限制了。一看 `nums.length` 哦 5e5 啊，O(n3) 那肯定 OOT 了。

ok，试了一些基础方法，比如转 `hashmap` 存下标这样的方式还是没能解这个题。但是能感觉到这个题解的代码是一个动态的过程，但又列不出状态转移方程，那肯定就是贪心了。定义 `min_1` 记录最小值，定义 `min_2` 记录第二小值，那么当一个数比目前的第二小值大时显然就得到了解。

```python
class Solution:
    def increasingTriplet(self, nums: List[int]) -> bool:
        min_1 = float('inf')
        min_2 = float('inf')
        for i in nums:
            if i <= min_1:
                min_1 = i
            elif i<= min_2:
                min_2 = i
            else:
                return True
        return False
```

### 238. 除自身以外数组的乘积

前缀和问题，分别正序和逆序遍历数组找出当前元素之前的积得到 `pre` 与 `suf` 数组，将 `suf` 反序（因为是逆序遍历的），然后对于结果来说就是当前位置的前缀积与后缀积相乘。

```python
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        pre = []
        suf = []
        temp = 1
        for i in nums:
            pre.append(temp)
            temp *= i
        temp = 1
        for i in nums[::-1]:
            suf.append(temp)
            temp *= i
        suf[:] = suf[::-1]
        res = []
        for i in range(len(nums)):
            res.append(pre[i]*suf[i])
        return res
```

### 560. 和为 K 的子数组

前缀和 + hashmap 。

```python
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        pre_sum = [0]
        for i in nums:
            pre_sum.append(pre_sum[-1]+i)
        hashmap = {0:1}
        res = 0
        for i in range(len(nums)):
            temp = pre_sum[i+1]-k
            if temp in hashmap:
                res += hashmap[temp]
            if pre_sum[i+1] not in hashmap:
                hashmap[pre_sum[i+1]] = 1
            else:
                hashmap[pre_sum[i+1]] += 1
        return res
```

## 字符串

### 415. 字符串相加

模拟。

```python
class Solution:
    def addStrings(self, num1: str, num2: str) -> str:
        res = ""
        i, j, carry = len(num1) - 1, len(num2) - 1, 0
        while i >= 0 or j >= 0:
            n1 = int(num1[i]) if i >= 0 else 0
            n2 = int(num2[j]) if j >= 0 else 0
            temp = n1 + n2 + carry
            carry = temp // 10
            res = str(temp % 10) + res
            i, j = i - 1, j - 1
        return "1" + res if carry else res
```

### 409. 最长回文串

用 `dict` 去统计每个字符出现的次数，如果是偶数可以直接用 `n` 个字符构成回文串，如果是奇数则可以用 `n-1` 个字符来构成，不过这时要标记遇到了奇数，之后如果标记位为真说明遇到了奇数，结果就会加 1 。

```python
class Solution:
    def longestPalindrome(self, s: str) -> int:
        res = 0
        hashmap = {}
        flag = False
        for i in s:
            if i not in hashmap:
                hashmap[i] = 1
            else:
                hashmap[i] += 1
        for value in hashmap.values():
            if value&1:
                res += value - 1
                flag = True
            else:
                res += value
        return res+1 if flag else res
```

### 290. 单词规律

用 `dict` 去计数匹配。

```python
class Solution:
    def wordPattern(self, pattern: str, s: str) -> bool:
        dict_pos = {}
        dict_nag = {}
        lst_ptn = list(pattern)
        lst_s = s.split()
        if len(lst_ptn) != len(lst_s):
            return False
        for k,v in zip(lst_ptn, lst_s):
            if k not in dict_pos:
                dict_pos[k] = v
            dict_nag[v] = k
        if len(dict_pos) != len(dict_nag):
            return False
        for k,v in dict_pos.items():
            if k != dict_nag[v]:
                return False
        return True
```

### 763. 划分字母区间

用 `dict` 去记录每个字母首次出现尾次出现的索引，然后根据首次出现的索引排序，遍历排序后的索引，并记录首次索引和尾次索引。会遇到 3 种情况：

- 遍历的首次索引大于当前记录的尾次出现索引 - 将当前结果添加，更新当前记录的首次索引和尾次索引。
- 遍历的尾次索引大于当前记录的尾次索引 - 更新当前记录的尾次索引。
- 遍历的首尾区间被当前记录的索引包含 - 不处理。

```python
class Solution:
    def partitionLabels(self, s: str) -> List[int]:
        hashmap = {}
        for i,v in enumerate(s):
            if v not in hashmap:
                hashmap[v] = (i, i)
            else:
                start, end = hashmap[v]
                hashmap[v] = (start, i)
        sorted_index = sorted(hashmap.items(), key=lambda x:x[1][0])
        res = []
        start = -1
        end = -1
        for i in sorted_index:
            if i[1][0] > end:
                res.append(end-start+1)
                start = i[1][0]
                end = i[1][1]
            elif i[1][1] > end:
                end = i[1][1]
        res.append(end-start+1)
        return res[1:]
```

### 49. 字母异位词分组

这道题主要考察如何将具有相同特征（字母异位词）作为 `dict` 的 `key` ，毕竟如果只是对每个字母做计数得到的子 `dict` 是不能作为 `key` 的。因此首先我想到的是计数之后再双循环去处理，不幸地超时了。然后可以将每个单词重新排序后作为 `key` 。

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

官解里提到的另一种方法实际也是再找寻合适的 `key` ，也是不排序直接计数的方法，我最开始也这样想但是没想到。官解这里采用长度为 26 的 `list` 来记录字母出现的次数，`ord(x)-ord('a')` 即为对应数组下标，再将 `list` 转为 `tuple` 以实现哈希。

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

### 43. 字符串相乘

这题前几天在编程能力计划里做过，不再赘述。

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

### 187. 重复的DNA序列

用 2 个 `set` 来记录，一个存放已经出现过的序列，另一个用来存放结果，如果已经出现过，就添加进结果中。

```python
class Solution:
    def findRepeatedDnaSequences(self, s: str) -> List[str]:
        hashset = set('')
        res = set('')
        for i in range(len(s)):
            temp = s[i:i+10]
            if temp not in hashset:
                hashset.add(temp)
            else:
                res.add(temp)
        return list(res)
```

### 5. 最长回文子串

暴力，过了。从长到短取序列，如果某个序列是回文串，直接返回他。

```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        s_len = len(s)
        c_len = s_len
        while c_len > 0:
            for i in range(s_len-c_len+1):
                temp = s[i:i+c_len]
                if temp == temp[::-1]:
                    return temp
            c_len -= 1
        return None
```

中心扩散算法，对每个点开始从中心往左右两边扩散直到扩散结果不是回文串。这个算法的难点在于有两种情况，比如 `aba` 与 `abba` 这两个字符串，都从第一个 `b` 开始扩散，很难同时进行处理，所以这里需要假设两种情况扩散的结果，选取最大回文串。

```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        def get_palindrome(s, l, r):
            while l>=0 and r<len(s) and s[l]==s[r]:
                l -= 1
                r += 1
            return l,r
        
        s_len = len(s)
        left, right = 0, 0
        for i in range(s_len):
            l1, r1 = get_palindrome(s, i, i)
            l2, r2 = get_palindrome(s, i, i+1)
            if r1-l1 > right-left:
                left, right = l1,r1
            if r2-l2 > right-left:
                left, right = l2,r2
        return s[left+1:right]
```

中心扩散这个算法其实有点动态规划的思想在里面，我之前其实很少遇到二维 dp ，这里用 `dp[i][j]` 表示字符串 `s[i:j]` 是否是回文字符串，可以列出如下状态转移方程：
$$
dp[i][j]=\begin{cases}
dp[i+1][j-1]\&(s[i]==s[j])\ \ \ \ if\ j>i+1\\
s[i]==s[j]\ \ \ \ if\ j=i+1\\
True\ \ \ \ if\ i==j
\end{cases}
$$
但是得注意，这里跟暴力不太一样，是从小往大推，不是从大往小推（否则算 `dp[i][j]` 时根本不知道 `dp[i+1][j-1]` ）。

```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        s_len = len(s)
        dp = [[False]*s_len for _ in range(s_len)]
        for i in range(s_len):
            dp[i][i] = True
        start, max_len = 0, 1
        for c_len in range(2,s_len+1):
            for i in range(s_len-c_len+1):
                j = i+c_len-1
                if s[i] != s[j]:
                    dp[i][j] = False
                else:
                    if j>i+1:
                        dp[i][j] = dp[i+1][j-1]
                    else:
                        dp[i][j] = True
                if dp[i][j] and c_len>max_len:
                    start = i
                    max_len = c_len
        return s[start:start+max_len]
```

但是动态规划，这里也不快。

## 链表

### 2. 两数相加

一种方式是先把 `l1` 与 `l2` 的数都取出来相加之后再根据结果生成新的链表。

```python
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        num1 = 0
        digit = 1
        while l1:
            num1 += l1.val*digit
            l1 = l1.next
            digit *= 10
        num2 = 0
        digit = 1
        while l2:
            num2 += l2.val*digit
            l2 = l2.next
            digit *= 10
        res_num = num1+num2
        dummy = ListNode()
        temp = dummy
        for i in str(res_num)[::-1]:
            temp.next = ListNode(int(i))
            temp = temp.next
        return dummy.next
```

否则我们需要哨兵节点用于返回最终的链表，并需要前置节点用于将当前数位两数之和结果相加。

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
            if temp >= 10:
                temp -= 10
                carry = 1
            else:
                carry = 0
            node.next = ListNode(temp)
            node = node.next
        if carry:
            node.next = ListNode(1)
        return dummy.next
```

### 142. 环形链表 II

用 `set` 存储每个节点然后遍历链表，如果节点已经存在，那么这个节点就是入环第一个节点。

```python
class Solution:
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        hashset = set('')
        while head:
            if head not in hashset:
                hashset.add(head)
                head = head.next
            else:
                return head
        return None
```

快慢指针，快指针走的步数减去慢指针走的步数一定是环的整数倍。

```python
class Solution:
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        fast, slow = head, head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
            if fast == slow:
                temp = head
                while temp != slow:
                    temp = temp.next
                    slow = slow.next
                return temp
        return None
```

### 160. 相交链表

用 `set` 存储 `headA` 的每个节点，然后遍历 `headB` 看节点是否在 `set` 中。

```python
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        setA = set('')
        while headA:
            setA.add(headA)
            headA = headA.next
        while headB:
            if headB in setA:
                return headB
            headB = headB.next
        return None
```

或者用两个 `set` 存储，在时间效率上有一定的优化。

```python
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        setA, setB = set(''), set('')
        while headA or headB:
            if headA:
                if headA in setB: return headA
                setA.add(headA)
                headA = headA.next
            if headB:
                if headB in setA: return headB
                setB.add(headB)
                headB = headB.next
        return None
```

双指针，一个以 `headA+headB` 顺序遍历，一个以 `headB+headA` 顺序遍历，相遇的时候就是相交节点。

```python
class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        if not headA or not headB:
            return None
        nodeA = headA
        nodeB = headB
        while nodeA != nodeB:
            nodeA = headB if not nodeA else nodeA.next
            nodeB = headA if not nodeB else nodeB.next
        return nodeA
```

### 82. 删除排序链表中的重复元素 II

一个哨兵节点用于返回结果链表，另一个用于记录新链表节点。循环遍历 `head` ，假设 `temp` 是 `head` 节点的下个节点：

- 如果 `temp` 是空节点或者 `temp.val != head.val` 则将当前 `head` 节点添加至记录。
- 否则 `head` 移到不是重复元素的位置。

```python
class Solution:
    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode()
        node = dummy
        while head:
            temp = head.next
            if not temp or temp.val != head.val:
                node.next = head
                head = head.next
                node = node.next
            else:
                while temp and temp.val == head.val:
                    temp = temp.next
                head = temp
        node.next = head
        return dummy.next
```

### 24. 两两交换链表中的节点

模拟。

```python
class Solution:
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head:
            return None
        if not head.next:
            return head
        dummy = ListNode()
        node = dummy
        while head and head.next:
            p1 = head
            p2 = head.next
            p1.next, p2.next = p2.next, p1
            node.next = p2
            node = node.next.next
            head = head.next
        return dummy.next
```

### 707. 设计链表

`list` 实现。

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

单向链表。

```python
class ListNode:
    def __init__(self, val=0):
        self.val = val
        self.next = None


class MyLinkedList:

    def __init__(self):
        self.size = 0
        self.head = ListNode(0)

    def get(self, index: int) -> int:
        if index < 0 or index >= self.size:
            return -1
        node = self.head
        for _ in range(index+1):
            node = node.next
        return node.val

    def addAtHead(self, val: int) -> None:
        self.addAtIndex(0, val)

    def addAtTail(self, val: int) -> None:
        self.addAtIndex(self.size, val)

    def addAtIndex(self, index: int, val: int) -> None:
        if index > self.size:
            return None
        index = max(index, 0)
        self.size += 1
        node = self.head
        for _ in range(index):
            node = node.next
        temp = ListNode(val)
        temp.next = node.next
        node.next = temp

    def deleteAtIndex(self, index: int) -> None:
        if index < 0 or index >= self.size:
            return None
        self.size -= 1
        node = self.head
        for _ in range(index):
            node = node.next
        node.next = node.next.next
```

### 25. K 个一组翻转链表

用栈可以很轻松的解决，直接每次 `k` 个节点进栈，然后出栈生成新链表，为了防止节点不够进栈的情况，每次进栈前先记录一下节点。

```python
class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        res = ListNode()
        node = res
        stack = []
        while True:
            pre = head
            for _ in range(k):
                if not head:
                    node.next = pre
                    return res.next
                stack.append(head)
                head = head.next
            while stack:
                temp = stack.pop()
                node.next = temp
                node = node.next
        return None
```

### 143. 重排链表

在编程能力里做过。

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

### 155. 最小栈

最开始读这道题没读懂，后面发现原来这个题既想记录元素入栈的顺序又想记录最小值，那么比较合理的就是采用空间换时间的方法，两个栈，一个正常存数，一个存最小值。

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

### 1249. 移除无效的括号

可以采用一个栈来记录有效的括号，遍历字符串时，如果遇到左括号，下标入栈，如果遇到右括号，那么判断是否可出栈，能出则出，不能说明这个右括号是无效的，把下标记录一下。遍历完成后，还在栈里的下标说明这些左括号找不到匹配的右括号，也记录。最后凡是记录的位置不添加到结果里。

```python
class Solution:
    def minRemoveToMakeValid(self, s: str) -> str:
        stack = []
        error = set('')
        for i,v in enumerate(s):
            if v == '(':
                stack.append(i)
            elif v == ')':
                if stack:
                    stack.pop()
                else:
                    error.add(i)
        for i in stack:
            error.add(i)
        res = []
        for i,v in enumerate(s):
            if i not in error:
                res.append(v)
        return ''.join(res)
```

### 1823. 找出游戏的获胜者

可以采用一个队列来模拟实际游戏情况。

```python
class Solution:
    def findTheWinner(self, n: int, k: int) -> int:
        queue = collections.deque()
        for i in range(1, n+1):
            queue.append(i)
        while len(queue) != 1:
            for i in range(k):
                temp = queue.popleft()
                if i!=k-1:
                    queue.append(temp)
        return queue.popleft()
```

但是其实我们是知道哪个位置的元素被删的，是 `start+k-1` 这个位置，所以可以直接删这个元素。

```python
class Solution:
    def findTheWinner(self, n: int, k: int) -> int:
        flag = [i for i in range(1, n+1)]
        start = 0
        count = n
        while count != 1:
            index = (start+k-1)%count
            flag.pop(index)
            start = index
            count -= 1
        return flag[0]
```

### 108. 将有序数组转换为二叉搜索树

从数组中点开始递归构建二叉平衡树。

```python
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        def insert_BST(l, r):
            mid = (l+r) // 2
            node = TreeNode(nums[mid])
            if l<=mid-1:
                node.left = insert_BST(l,mid-1)
            if r>=mid+1:
                node.right = insert_BST(mid+1,r)
            return node
        
        return insert_BST(0, len(nums)-1)
```

### 105. 从前序与中序遍历序列构造二叉树

二叉树的前序遍历是 `NLR` ，中序遍历是 `LNR` ，因此，`preorder` 的首个元素一定是 `root` 节点，然后根据 `root` 在 `inorder` 中的位置可以区分出左右子树的 `inorder` ，然后可以根据左右子树的 `inorder` 数量来找出左右子树的 `preorder` 。递归构造二叉树。

```python
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        if not preorder:
            return None
        node = TreeNode(preorder[0])
        index_node = inorder.index(preorder[0])
        left_num = index_node
        node.left = self.buildTree(preorder[1:1+left_num], inorder[:index_node])
        node.right = self.buildTree(preorder[1+left_num:], inorder[index_node+1:])
        return node
```

我们可以使用 `dict` 对于查找根节点进行一个优化，但是就不能像上面那样传新的 `list` 了。

```python
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:    
        def _build(preorder_left, preorder_right, inorder_left, inorder_right):
            if preorder_left > preorder_right:
                return None
            preorder_root = preorder_left
            inorder_root = hashmap[preorder[preorder_root]]
            root = TreeNode(preorder[preorder_root])
            size_left_subtree = inorder_root - inorder_left
            root.left = _build(preorder_left + 1, preorder_left + size_left_subtree, inorder_left, inorder_root - 1)           
            root.right = _build(preorder_left + size_left_subtree + 1, preorder_right, inorder_root + 1, inorder_right)
            return root

        hashmap={v:i for i,v in enumerate(inorder)}
        return _build(0, len(preorder)-1, 0, len(inorder)-1)
```

### 103. 二叉树的锯齿形层序遍历

简单的层次遍历，根据不同层数添加不同遍历方向的值。

```python
class Solution:
    def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        queue = [root]
        res = []
        count = 1
        while queue:
            temp = []
            for _ in range(len(queue)):
                node = queue.pop(0)
                temp.append(node.val)
                if node.left: queue.append(node.left)
                if node.right: queue.append(node.right)
            if count&1:
                res.append(temp)
            else:
                res.append(temp[::-1])
            count += 1
        return res
```

### 199. 二叉树的右视图

依然是二叉树的层次遍历，我们在添加节点时从右往左添加并且只将第一个出队的值添加进结果中即可。

```python
class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []
        queue = [root]
        res = []
        while queue:
            for i in range(len(queue)):
                node = queue.pop(0)
                if i == 0:
                    res.append(node.val)
                if node.right: queue.append(node.right)
                if node.left: queue.append(node.left)
        return res
```

### 113. 路径总和 II

带其它信息的搜索，深度优先搜索和广度优先搜索都可以。

```python
class Solution:
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
        if not root:
            return []
        queue = collections.deque()
        res = []
        queue.append((root, [root.val], root.val))
        while queue:
            node, path, path_sum = queue.popleft()
            if node.left: queue.append((node.left, path+[node.left.val], path_sum+node.left.val))
            if node.right: queue.append((node.right, path+[node.right.val], path_sum+node.right.val))
            if not node.left and not node.right:
                if path_sum == targetSum:
                    res.append(path)
        return res
```

### 450. 删除二叉搜索树中的节点

在查询并删除某个节点中，可能会遇到以下几种情况。

- 当前节点为空，返回空节点。
- 当前节点值大于目标值，那么应该去左子树寻找目标节点，相当于对左子树运用同样的函数删一次，因此 `root.left = self.deleteNode(root.left, key)` 。
- 当前节点值小于目标值，那么应该去右子树寻找目标节点。
- 当前节点值等于目标值，也就是找到了。
  - 如果没有左子树，那么直接用右子节点覆盖当前节点就行了。
  - 同样如果没有右子树，那么直接用左子节点覆盖当前节点就行了。
  - 如果都有的话，可以将左子树接到右子树的最左节点，然后用右子树覆盖当前节点。

```python
class Solution:
    def deleteNode(self, root: Optional[TreeNode], key: int) -> Optional[TreeNode]:
        if not root:
            return None
        if root.val > key:
            root.left = self.deleteNode(root.left, key)
        elif root.val < key:
            root.right = self.deleteNode(root.right, key)
        else:
            if not root.left: return root.right
            if not root.right: return root.left
            node = root.right
            while node.left:
                node = node.left
            node.left = root.left
            root = root.right
        return root
```

还有一种方法，将二叉搜索树中序遍历，并且如果遇到目标值就跳过。在根据 `list` 构造二叉搜索树的方法生成新的树。

```python
class Solution:
    def deleteNode(self, root: Optional[TreeNode], key: int) -> Optional[TreeNode]:
        nums = []
        if not root:
            return None
        def LNR(node):
            if not node: return None
            LNR(node.left)
            if node.val != key: nums.append(node.val)
            LNR(node.right)
        LNR(root)
        nums_len = len(nums)
        if nums_len == 0: return None
        def insert_BST(l, r):
            mid = (l+r)//2
            node = TreeNode(val=nums[mid])
            if l<=mid-1:
                node.left = insert_BST(l, mid-1)
            if r>=mid+1:
                node.right = insert_BST(mid+1, r)
            return node
        return insert_BST(0, nums_len-1)
```

### 230. 二叉搜索树中第K小的元素

暴力解法，先将二叉搜索树中序遍历，然后返回第 `K` 小的元素。递归。

```python
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        nums = []
        def LNR(node):
            if not node: return None
            LNR(node.left)
            nums.append(node.val)
            LNR(node.right)
        LNR(root)
        return nums[k-1]
```

非递归。

```python
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        nums = []
        stack = []
        while stack or root:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            nums.append(root.val)
            root = root.right
        return nums[k-1]
```

其实在非递归这里可以看出来，只要数组里有 `k` 个元素就行了，根本不需要遍历完。

```python
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        counter = 0
        stack = []
        while stack or root:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            counter += 1
            if counter == k: return root.val
            root = root.right
        return None
```

### 173. 二叉搜索树迭代器

在初始化 `__init__` 里把二叉搜索树的遍历结果写好，之后就很简单。

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

### 236. 二叉树的最近公共祖先

暴力解法，搜索 `p` , `q` 。遍历找出这两个节点，并且记录路径，逆序比较后得到最近公共祖先。

```python
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        queue = collections.deque()
        queue.append((root,[root]))
        p_ancestors = None
        q_ancestors = None
        while queue:
            for _ in range(len(queue)):
                node, ancestors = queue.pop()
                if node == p:
                    p_ancestors = ancestors
                if node == q:
                    q_ancestors = ancestors
                if p_ancestors and q_ancestors:
                    break
                if node.left: queue.append((node.left, ancestors+[node.left]))
                if node.right: queue.append((node.right, ancestors+[node.right]))
        for p_ancestor in reversed(p_ancestors):
            for q_ancestor in reversed(q_ancestors):
                if p_ancestor == q_ancestor:
                    return p_ancestor
        return None
```

### 297. 二叉树的序列化与反序列化

都可以使用二叉树的遍历去解决。

```python
class Codec:

    def serialize(self, root):
        if not root:
            return ''
        res = []
        queue = collections.deque()
        queue.append(root)
        while queue:
            node = queue.popleft()
            if node:
                res.append(str(node.val))
                queue.append(node.left)
                queue.append(node.right)
            else:
                res.append('')
        return ','.join(res)

    def deserialize(self, data):
        if not data:
            return None
        data = data.split(',')
        root = TreeNode(data[0])
        queue = collections.deque()
        queue.append(root)
        i = 1
        while queue:
            node = queue.popleft()
            if data[i] != '':
                node.left = TreeNode(int(data[i]))
                queue.append(node.left)
            i += 1
            if data[i] != '':
                node.right = TreeNode(int(data[i]))
                queue.append(node.right)
            i += 1
        return root
```

### 997. 找到小镇的法官

用 `dict` 记录每个人的信任其人数，用 `set` 记录信任过别人的人，如果满足信任某个人人数等于 `n-1` 并且他没有信任的人，他就是法官，否则没有法官。

```python
class Solution:
    def findJudge(self, n: int, trust: List[List[int]]) -> int:
        hashmap = {i:0 for i in range(1,n+1)}
        hashset = set('')
        for p,t in trust:
            hashset.add(p)
            if t not in hashmap:
                hashmap[t] = 1
            else:
                hashmap[t] += 1
        for p,count in hashmap.items():
            if count == n-1 and p not in hashset:
                return p
        return -1
```

### 1557. 可以到达所有点的最少点数目

推演几次可以发现可以到达所有点的最少点其实就是不能由其它点到达的点。

```python
class Solution:
    def findSmallestSetOfVertices(self, n: int, edges: List[List[int]]) -> List[int]:
        hashset = {t for f,t in edges}
        return [i for i in range(n) if i not in hashset]
```

### 841. 钥匙和房间

这个和上道题不一样的是这道题的图是有可能有环的，因此直接沿用上道题的方法可能不行，可以加一个栈或队列进行图的搜索，搜索结果就是 `0` 能到达的点。

```python
class Solution:
    def canVisitAllRooms(self, rooms: List[List[int]]) -> bool:
        hashset = set('')
        queue = collections.deque()
        queue.append(0)
        while queue:
            node = queue.popleft()
            for i in rooms[node]:
                if i not in hashset:
                    queue.append(i)
                    hashset.add(i)
        for i in range(1,len(rooms)):
            if i not in hashset:
                return False
        return True
```

