---
title: 「算法」 - 学习计划
date: 2022-02-16 01:48:12
categories: [ComputerScience, Algorithm, LeetCode]
tags: [python, binary search, point]
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

### 283. 移动零

快慢指针问题，分两步：

1. 填非零值，快指针负责遍历数组找到非零值，填入慢指针指向位置，然而慢指针前进一位，否则不处理。
2. 填零值，从慢指针开始移动到数组尾端都填 0 。

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

### 167. 两数之和 II - 输入有序数组

其实和第一题 `两数之和` 还是挺像的，依然是用 `dict` 记录是否存在满足条件的数。两个小坑，第一个 index 从 1 开始，第二个说不能使用重复的元素是指不能用同一个数而不是同一个值（如果数组里有两个一样的依然可以用）。

```python
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        hashmap = {}
        for i,v in enumerate(numbers):
            if target-v in hashmap:
                return [hashmap[target-v]+1, i+1]
            else:
                hashmap[v] = i
        return None
```

### 344. 反转字符串

这是真的经典双指针，一个指向头一个指向尾，然后交换值再都向对方移动一位，直到相遇。

```python
class Solution:
    def reverseString(self, s: List[str]) -> None:
        left = 0
        right = len(s)-1
        while left<right:
            s[left], s[right] = s[right], s[left]
            left += 1
            right -= 1
        return None
```

或者用 `for` 循环。

```python
class Solution:
    def reverseString(self, s: List[str]) -> None:
        s_len = len(s)
        for i in range(s_len//2):
            s[i], s[s_len-i-1] = s[s_len-i-1], s[i]
        return None
```

当然 Python 里面用切片是最方便的：

```python
class Solution:
    def reverseString(self, s: List[str]) -> None:
        s[:] = s[::-1]
        return None
```

### 557. 反转字符串中的单词 III

`join()` 加列表生成式加切片。

```python
class Solution:
    def reverseWords(self, s: str) -> str:
        s_lst = s.split()
        return ' '.join([i[::-1] for i in s_lst])
```

### 876. 链表的中间结点

这道题被归到双指针题目里面，显然就是一道快慢指针的问题，逻辑很简单，快指针走两次，慢指针走一次。最后慢指针的位置就是中间结点。

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

### 19. 删除链表的倒数第 N 个结点

第一种思路，一次扫描，之后再删。注意要删一个结点需要找到的是它的前序结点（而不是它自己）。

```python
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        counter = 0
        node = head
        while node:
            node = node.next
            counter += 1
        dummy = ListNode(0, head)
        node = dummy
        for i in range(counter-n):
            node = node.next
        node.next = node.next.next
        return dummy.next
```

这里我感觉有个坑就是测试用例好像是异步跑的 。我曾经想过用：

```python
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        counter = 0
        node = head
        while node:
            node = node.next
            counter += 1
        node = head
        for i in range(counter-n-1):
            node = node.next
        node.next = node.next.next
        return head
```

会得到如下报错，假如你打印错误的话你会很迷，其实这是第二个测试用例的错误。

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202212171014954.png)

应该写成：

```python
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        counter = 0
        node = head
        while node:
            node = node.next
            counter += 1
        node = head
        for i in range(counter-n-1):
            node = node.next
        if not node.next:
            return head.next
        node.next = node.next.next
        return head
```

第二种是快慢指针的方式：

```python
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        dummy = ListNode(0, head)
        fast = head
        slow = dummy
        for i in range(n):
            fast = fast.next
        while fast:
            fast = fast.next
            slow = slow.next
        slow.next = slow.next.next
        return dummy.next
```

## 滑动窗口

### 3. 无重复字符的最长子串

我们采用队列来实现滑动窗口，遍历字符串，当队里没有当前字符时，当前字符入队；当队里有当前字符时，先统计队列长度进而看情况更新最大子字符串，出队直到队里没有当前字符，再添加当前字符到队尾。考虑到字符串字符都不一样的情况，也就是没有更新最大子字符串长度，遍历完之后还应该更新一次。

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        queue = []
        res = 0
        for i in s:
            if i in queue:
                res = max(res, len(queue))
                while queue[0] != i:
                    del queue[0]
                del queue[0]
            queue.append(i)
        return max(res, len(queue))
```

### 567. 字符串的排列

`s1` 排列之一是 `s2` 的字串，这句话的意思就是 `s1` 的 `dict` 计数结果和 `s2` 的某字串 `dict` 计数结果是一样的，那么显然 `s2` 这个字串长度和 `s1` 也就一样了。所以我们可以模拟这个计数过程。

```python
class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        s1_dict = {}
        for i in s1:
            if i not in s1_dict:
                s1_dict[i] = 1
            else:
                s1_dict[i] += 1
        s1_len = len(s1)
        for i in range(len(s2)-s1_len+1):
            subs = s2[i:i+s1_len]
            subs_dict = {}
            for v in subs:
                if v not in subs_dict:
                    subs_dict[v] = 1
                else:
                    subs_dict[v] += 1
            flag = True
            for k,v in subs_dict.items():
                if k not in s1_dict or v != s1_dict[k]:
                    flag = False
                    break
            if flag:
                return True
        return False
```

那么上面的代码是很慢的，我们可以发现根本没有必要每次创建 `dict` ，维护一个 `dict` 就行了。

```python
class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        s1_len = len(s1)
        s2_len = len(s2)
        if s1_len > s2_len:
            return False
        s1_dict = {}
        for i in s1:
            if i not in s1_dict:
                s1_dict[i] = 1
            else:
                s1_dict[i] += 1
        s2_dict = {}
        for i in range(s1_len):
            if s2[i] not in s2_dict:
                s2_dict[s2[i]] = 1
            else:
                s2_dict[s2[i]] += 1
        flag = True
        for k,v in s1_dict.items():
            if k not in s2_dict or v != s2_dict[k]:
                flag = False
                break
        if flag:
            return True
        for i in range(s1_len, s2_len):
            s2_dict[s2[i-s1_len]] -= 1
            if s2[i] not in s2_dict:
                s2_dict[s2[i]] = 1
            else:
                s2_dict[s2[i]] += 1
            flag = True
            for k,v in s1_dict.items():
                if k not in s2_dict or v != s2_dict[k]:
                    flag = False
                    break
            if flag:
                return True
        return False
```

如果用 collections 的 Counter 看起来就很简洁，但是似乎变慢了：

```python
class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        s1_len = len(s1)
        s1_dict = collections.Counter(s1)
        s2_dict = collections.Counter(s2[:s1_len])
        if s1_dict == s2_dict:
            return True
        for i in range(s1_len, len(s2)):
            s2_dict[s2[i-s1_len]] -= 1
            s2_dict.update({s2[i]:1})
            if s1_dict == s2_dict:
                return True
        return False
```

## 广度优先搜索 / 深度优先搜索

### 733. 图像渲染

深度优先搜索的递归解法。为了防止无限循环，我们需要一个 `record` 数组来记录已经遍历过的点。

```python
class Solution:
    def floodFill(self, image: List[List[int]], sr: int, sc: int, color: int) -> List[List[int]]:
        m = len(image)
        n = len(image[0])
        record = [[1]*n for _ in range(m)]
        def _floodFill(image, sr, sc):
            if record[sr][sc]:
                temp, image[sr][sc], record[sr][sc] = image[sr][sc], color, 0
            else:
                return image
            if sr>=1 and image[sr-1][sc]==temp:
                image = _floodFill(image, sr-1, sc)
            if sr+1<m and image[sr+1][sc]==temp:
                image = _floodFill(image, sr+1, sc)
            if sc>=1 and image[sr][sc-1]==temp:
                image = _floodFill(image, sr, sc-1)
            if sc+1<n and image[sr][sc+1]==temp:
                image = _floodFill(image, sr, sc+1)
            return image
        image = _floodFill(image, sr, sc)
        return image
```

当然我们也可以用栈来实现非递归算法。

```python
class Solution:
    def floodFill(self, image: List[List[int]], sr: int, sc: int, color: int) -> List[List[int]]:
        m = len(image)
        n = len(image[0])
        record = [[1]*n for _ in range(m)]
        stack = [(sr,sc)]
        while stack:
            i,j = stack.pop()
            if record[i][j]:
                temp, image[i][j], record[i][j] = image[i][j], color, 0
                if j+1<n and image[i][j+1] == temp:
                    stack.append((i, j+1))
                if j>=1 and image[i][j-1] == temp:
                    stack.append((i, j-1))
                if i+1<m and image[i+1][j] == temp:
                    stack.append((i+1, j))
                if i>=1 and image[i-1][j] == temp:
                    stack.append((i-1, j))
        return image
```

### 695. 岛屿的最大面积

与上题相似，我们需要一个 `record` 来记录某个点是否被算过。

```python
class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])
        record = [[1]*n for _ in range(m)]
        max_count = 0

        def dfs(i, j):
            stack = [(i, j)]
            res = 0
            while stack:
                i,j = stack.pop()
                if record[i][j]:
                    record[i][j] = 0
                    res += 1
                    if j+1<n and grid[i][j+1] == 1:
                        stack.append((i, j+1))
                    if j>=1 and grid[i][j-1] == 1:
                        stack.append((i, j-1))
                    if i+1<m and grid[i+1][j] == 1:
                        stack.append((i+1, j))
                    if i>=1 and grid[i-1][j] == 1:
                        stack.append((i-1, j))
            return res

        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    max_count = max(max_count, dfs(i, j))

        return max_count
```

### 617. 合并二叉树

深度优先搜索，递归。

```python
class Solution:
    def mergeTrees(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> Optional[TreeNode]:
        if root1 and root2:
            root1.val += root2.val
            root1.left = self.mergeTrees(root1.left, root2.left)
            root1.right = self.mergeTrees(root1.right, root2.right)
            return root1
        return root1 or root2
```

也可以不用递归，这时我们需要栈。假设我们的目的是把 `root2` 合并到 `root1` 里，我们会有如下情况：

- 首先我们需要出栈拿到 `root1` , `root2` 的节点。
- 将 `root1` 和 `root2` 的值相加，这里我们需要保证除非根节点是空的，否则不会遇到 `root1` 或 `root2` 为空的情况。
- 如果 `root1` 和 `root2` 都有右子树或都有左子树，则我们按顺序进行压栈。
- 如果 `root1` 没有右子树或者没有左子树而 `root2` 有，只需要把 `root2` 的接过来，不需要进栈操作。
- 剩余情况我们都不用做处理。

```python
class Solution:
    def mergeTrees(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> Optional[TreeNode]:
        if root1 is None or root2 is None:
            return root1 or root2
        stack = [(root1, root2)]
        while stack:
            t1, t2 = stack.pop()
            if t1 and t2:
                t1.val += t2.val
            if t1.right and t2.right:
                stack.append((t1.right, t2.right))
            elif t1.right is None:
                t1.right = t2.right
            if t1.left and t2.left:
                stack.append((t1.left, t2.left))
            elif t1.left is None:
                t1.left = t2.left
        return root1
```

广度优先搜索。其实改动非常小，因为这道题本身就跟顺序没什么关系。把上面代码的 `pop()` （出栈）换成 `pop(0)`（出队）就行了。

```python
class Solution:
    def mergeTrees(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> Optional[TreeNode]:
        if root1 is None or root2 is None:
            return root1 or root2
        queue = [(root1, root2)]
        while queue:
            t1, t2 = queue.pop(0)
            if t1 and t2:
                t1.val += t2.val
            if t1.right and t2.right:
                queue.append((t1.right, t2.right))
            elif t1.right is None:
                t1.right = t2.right
            if t1.left and t2.left:
                queue.append((t1.left, t2.left))
            elif t1.left is None:
                t1.left = t2.left
        return root1
```

### 116. 填充每个节点的下一个右侧节点指针

这个填充其实是父节点在填充子节点的指针。分为左右子节点两种情况：

- 左子节点，`next` 指针指向位置其实就是右子节点的位置。
- 右子节点，`next` 指针指向位置是此节点 `next` 指针指向节点的左子节点。

想明白这个，用递归实现就很简单了：

```python
class Solution:
    def connect(self, root: Optional[Node]) -> Optional[Node]:
        if root and root.left and root.right:
            root.left.next = root.right
            if root.next:
                root.right.next = root.next.left
            root.left = self.connect(root.left)
            root.right = self.connect(root.right)
        return root
```

对于非递归来讲，采用广度优先搜索。

```python
class Solution:
    def connect(self, root: Optional[Node]) -> Optional[Node]:
        if root is None:
            return root
        queue = [root]
        while queue:
            t = queue.pop(0)
            if t.left and t.right:
                t.left.next = t.right
                if t.next:
                    t.right.next = t.next.left
                queue.append(t.left)
                queue.append(t.right)
        return root
```

也可以一直找最左节点与 `next` 指针：

```python
class Solution:
    def connect(self, root: Optional[Node]) -> Optional[Node]:
        if root is None:
            return root
        leftmost = root
        while leftmost.left:
            node = leftmost
            while node:
                node.left.next = node.right
                if node.next:
                    node.right.next = node.next.left
                node = node.next
            leftmost = leftmost.left
        return root
```

### 542. 01 矩阵

这道题第一眼应该是一道广度优先搜索的题目，然而居然超时了。

然后又想到一招，先记录 0 的位置，之后直接遍历距离 0 的最小值，毕竟这个距离可以用 `abs(x1-x0)+abs(y1-y0)` 得到，比上一个好像快了一点，但是依然超时。

然后可以转换一下，从找每个点到 0 的距离变成 0 到每个点的距离，也就是所谓的多源广度优先搜索，把 0 看作一个整体，首先找距它们 0 个位置的点（自身），再找距它们 1 个位置的点，以此类推直到找到所有点。然后我成功 AC 了一次。

```python
class Solution:
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        m = len(mat)
        n = len(mat[0])
        res = [[-1]*n for _ in range(m)]
        queue = []
        for r in range(m):
            for c in range(n):
                if not mat[r][c]:
                    queue.append((r, c, 0))
        while queue:
            i, j, count = queue.pop(0)
            if res[i][j] == -1:
                res[i][j] = count
                for ni, nj in ((i-1, j), (i+1, j), (i, j-1), (i, j+1)):
                    if 0<=ni<m and 0<=nj<n and res[ni][nj] == -1:
                        queue.append((ni, nj, count+1))
        return res
```

这个通过时间显然是哪里有问题。然后打印了一下循环的信息，我发现这里有个问题，我用 `res` 去分辨哪些点的值被搜索过，而 `res` 是在出队的时候更改搜索信息的，因此，同一层（离多个 0 同一距离）的点可能会被重复添加，也就是说除了 `res` 之外，我们最好再来一个记录值是否被搜索过的，比如 `set` ？

```python
class Solution:
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        m = len(mat)
        n = len(mat[0])
        res = [[0]*n for _ in range(m)]
        queue = []
        searched = set('')
        for r in range(m):
            for c in range(n):
                if not mat[r][c]:
                    queue.append((r, c, 0))
                    searched.add((r,c))
        while queue:
            i, j, count = queue.pop(0)
            for ni, nj in ((i-1, j), (i+1, j), (i, j-1), (i, j+1)):
                if 0<=ni<m and 0<=nj<n and (ni, nj) not in searched:
                    res[ni][nj] = count+1
                    queue.append((ni, nj, count+1))
                    searched.add((ni, nj))
        return res
```

优化一下代码：

```python
class Solution:
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        m = len(mat)
        n = len(mat[0])
        res = [[0]*n for _ in range(m)]
        queue = [(r,c) for r in range(m) for c in range(n) if mat[r][c]==0]
        searched = set(queue)
        while queue:
            i, j = queue.pop(0)
            for ni, nj in ((i-1, j), (i+1, j), (i, j-1), (i, j+1)):
                if 0<=ni<m and 0<=nj<n and (ni, nj) not in searched:
                    res[ni][nj] = res[i][j]+1
                    queue.append((ni, nj))
                    searched.add((ni, nj))
        return res
```

仍然很慢，难道是 `list` 的问题？数据量大之后 `list` 不够高效？把 `list` 换成 `collections.deque()` 双端队列。

```python
class Solution:
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        m, n = len(mat), len(mat[0])
        queue = collections.deque()
        searched = [[0]*n for _ in range(m)]
        res = [[0]*n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                if mat[i][j] == 0:
                    queue.append((i, j))
                    searched[i][j] = 1
        while queue:
            i, j = queue.popleft()
            for ni, nj in ((i-1, j), (i+1, j), (i, j-1), (i, j+1)):
                if 0<=ni<m and 0<=nj<n and not searched[ni][nj]:
                    res[ni][nj] = res[i][j]+1
                    queue.append((ni, nj))
                    searched[ni][nj] = 1
        return res
```

还真是，应该是 `list.pop(0)` 这个方法的事件复杂度是 `O(n)` 。

看了一下官解里提的动态规划方法，原理。

状态转移方程：
$$
f(i,j)=\begin{cases}
1+min(f(i-1,j), f(i+1,j), f(i,j-1),f(i,j+1))\ \ \ \ if\ (i,j)=1
\\0\ \ \ \ if\ (i,j)=0
\end{cases}
$$
这个是很好理解的，毕竟一个点可以由他上下左右四个点离 0 最近的位置决定。

但是 `for i in range(m) for j in range(n)` 这个循环相当于是从左上开始往右下遍历，也就是说这次遍历只会包含 `f(i-1,j), f(i,j-1)` 两个点的真实值，所以还需要一次从右下角开始的遍历，稍有不同的是右下角遍历的时候还可以同时处理下左上角遍历时的结果。

```python
class Solution:
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        m, n = len(mat), len(mat[0])
        res = [[2e4]*n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                if mat[i][j] == 0:
                    res[i][j] = 0
                else:
                    top, left = 1e4, 1e4
                    if i>0: top = res[i-1][j]
                    if j>0: left = res[i][j-1]
                    res[i][j] = min(top+1, left+1)
        for i in range(m-1, -1, -1):
            for j in range(n-1, -1, -1):
                if mat[i][j] == 0:
                    res[i][j] = 0
                else:
                    bottom, right = 1e4, 1e4
                    if i<m-1: bottom = res[i+1][j]
                    if j<n-1: right = res[i][j+1]
                    res[i][j] = min(res[i][j], bottom+1, right+1)
        return res
```

### 994. 腐烂的橘子

方法一，模拟橘子腐烂的过程，每分钟遍历一次，这里要注意在这分钟腐烂的橘子这分钟不会影响到其他橘子。

```python
class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        fresh = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    fresh += 1
        minute = 0
        while fresh != 0:
            minute += 1
            temp_set = set('')
            for i in range(m):
                for j in range(n):
                    if grid[i][j] > 1 and (i,j) not in temp_set:
                        for ni, nj in ((i-1, j), (i+1, j), (i, j-1), (i, j+1)):
                            if 0<=ni<m and 0<=nj<n and grid[ni][nj] == 1:
                                grid[ni][nj] = 2
                                temp_set.add((ni,nj))
                                fresh -= 1
            if len(temp_set)==0:
                return -1
        return minute
```

方法二，腐烂橘子的广度优先搜索，这里需要注意的是这分钟能感染的橘子下分钟不能再感染了（因为周围被感染过）。

```python
class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        fresh = 0
        queue = collections.deque()
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    fresh += 1
                elif grid[i][j] == 2:
                    queue.append((i,j))
        minute = 0
        while fresh != 0 and queue:
            minute += 1
            for _ in range(len(queue)):
                i, j = queue.popleft()
                for ni, nj in ((i-1,j), (i+1,j), (i,j-1), (i,j+1)):
                    if 0 <= ni < m and 0 <= nj < n and grid[ni][nj] == 1:
                        grid[ni][nj] = 2
                        queue.append((ni, nj))
                        fresh -= 1
        if fresh != 0:
            return -1
        return minute
```

## 递归 / 回溯

### 21. 合并两个有序链表

这道题用递归其实就 4 种情况：

- `list1` 节点为空 - 返回 `list2`
- `list2` 节点为空 - 返回 `list1`
- 都非空 `list1` 节点值小于等于 `list2` - 递归，返回 `list1` 。
- 都非空 `list2` 节点值小于 `list1` - 递归，返回 `list2` 。

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

非递归。

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
        if not list1:
            pre.next = list2
        if not list2:
            pre.next = list1
        return dummy.next
```

### 206. 反转链表

递归。

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

非递归。可以用栈去遍历节点。

```python
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode()
        stack = []
        while head:
            temp = head.next
            head.next = None
            stack.append(head)
            head = temp
        pre = dummy
        while stack:
            node = stack.pop()
            pre.next = node
            pre = pre.next
        return dummy.next
```

非递归，也能用三指针去遍历，因为对一个节点的 `next` 指针逆序与他前后节点和自身都有关系。

```python
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        pre = None
        dummy = None
        while head:
            temp = head.next
            dummy = head
            head.next = pre
            head = temp
            pre = dummy
        return dummy
```

### 77. 组合

Python 里的 `itertools.combinations()` 函数。

```python
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        return list(itertools.combinations(range(1,n+1), k))
```

从 [代码随想录 - 回溯算法](https://programmercarl.com/%E5%9B%9E%E6%BA%AF%E7%AE%97%E6%B3%95%E7%90%86%E8%AE%BA%E5%9F%BA%E7%A1%80.html) 前辈这里学习的回溯算法。

相当于把从 n 个数里找满足条件的 k 个数分解成 `for` 循环（横向遍历 n ）与递归（纵向遍历 `k` ）这样的结构。

```python
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        res = []
        path = []

        def combinations(n, k, i):
            if len(path) == k:
                res.append(path[:])
                return None
            for j in range(i, n+1):
                path.append(j)
                combinations(n, k, j+1)
                path.pop()

        combinations(n, k, 1)
        return res
```

上面的代码并不快，这就引申出了剪枝这个概念。

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202212230903768.png)

有的步骤是多余的，在于取完 x 个数之后，剩下数量必须大于等于 k-x 个，否则没有意义。

这个 x 就是 path 的元素个数。

```python
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        res = []
        path = []

        def combinations(n, k, i):
            if len(path) == k:
                res.append(path[:])
                return None
            for j in range(i, n-(k-len(path))+2):
                path.append(j)
                combinations(n, k, j+1)
                path.pop()

        combinations(n, k, 1)
        return res
```

其实这个这个递归函数不需要传 n，k，我们稍微简化一下：

```python
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        res = []
        path = []

        def combinations(i):
            if len(path) == k:
                res.append(path[:])
                return None
            for j in range(i, n-(k-len(path))+2):
                path.append(j)
                combinations(j+1)
                path.pop()

        combinations(1)
        return res
```

### 46. 全排列

Python 里的 `itertools.permutations()` 函数。

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        return list(itertools.permutations(nums, len(nums)))
```

这道题的树图我们可以画出：

![](https://movis-blog.oss-cn-chengdu.aliyuncs.com/img/202212230951840.png)

我们可以用一个 `used` 数组来记录哪些元素被使用过，但其实，使用过的元素已经在 `path` 里了，因此也不需要记录。

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        res = []
        path = []

        def permutations():
            if len(path) == len(nums):
                res.append(path[:])
                return None
            for i in nums:
                if i not in path:
                    path.append(i)
                    permutations()
                    path.pop()
        
        permutations()
        return res
```

### 784. 字母大小写全排列

这道题其实是求 0（设为小写），1（设为大写）可重复取数共取 k（ k 为字符串种字母的数量）个数的全排列。

我们可以按照上面的老套路，不过既然可以重复取数就没有限制条件了。

```python
class Solution:
    def letterCasePermutation(self, s: str) -> List[str]:
        s_lst = list(s)
        letter_index = []
        for i,v in enumerate(s_lst):
            if not v.isdigit():
                letter_index.append(i)
        res = []
        path = []
        def permutations():
            if len(path) == len(letter_index):
                res.append(path[:])
                return None
            for i in (0, 1):
                path.append(i)
                permutation()
                path.pop()
        permutation()
        for i in range(len(res)):
            for j in zip(res[i], letter_index):
                if j[0]:
                    s_lst[j[1]] = s_lst[j[1]].upper()
                else:
                    s_lst[j[1]] = s_lst[j[1]].lower()
            res[i] = ''.join(s_lst)
        return res
```

Python 的一行代码。

```python
class Solution:
    def letterCasePermutation(self, s: str) -> List[str]:
        return list(map(''.join, itertools.product(*map(lambda x:(x.lower(), x.upper()) if x.isalpha() else x, s))))
```

## 动态规划

### 70. 爬楼梯

状态转移方程：
$$
dp[i]=\begin{cases}dp[i-1]+dp[i-2]\ \ \ \ if\ i>2\\i\ \ \ \ if\ i=1\ or i=2\end{cases}
$$

```python
class Solution:
    def climbStairs(self, n: int) -> int:
        if n == 1:
            return 1
        if n == 2:
            return 2
        dp = [1,2]
        for i in range(2,n):
            dp.append(dp[i-1]+dp[i-2])
        return dp[-1]
```

### 198. 打家劫舍

状态转移方程：
$$
dp[i]=\begin{cases}max(dp[i-2]+nums[i], dp[i-1])\ \ \ \ if\ i>2
\\max(nums[i], nums[i-1])\ \ \ \ if\ i=2
\\nums[i]\ \ \ \ if\ i=1\end{cases}
$$

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return nums[0]
        if n == 2:
            return max(nums[0], nums[1])
        dp = [nums[0], max(nums[0], nums[1])]
        for i in range(2, n):
            dp.append(max(dp[i-2]+nums[i], dp[i-1]))
        return dp[-1]
```

### 120. 三角形最小路径和

状态转移方程
$$
\begin{cases}
min(dp[i-1][j-1],dp[i-1][j])+triangle[i][j]\ \ \ \ if\ i>=1\ and\ i>j>=1\\
dp[i-1][j]+triangle[i][j]\ \ \ \ if\ i>=1\ and\ j=0\\
dp[i-1][j-1]+triangle[i][j]\ \ \ \ if\ i>=1\ and\ i=j\\
triangle[i][j]\ \ \ \ if\ i=0
\end{cases}
$$

```python
class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        n = len(triangle)
        if n == 1:
            return triangle[0][0]
        dp = [[triangle[0][0]]]
        for i in range(1,n):
            temp = []
            for j in range(i+1):
                if j==0:
                    temp.append(dp[i-1][j]+triangle[i][j])
                elif j==i:
                    temp.append(dp[i-1][j-1]+triangle[i][j])
                else:
                    temp.append(min(dp[i-1][j], dp[i-1][j-1])+triangle[i][j])
            dp.append(temp)
        return min(dp[-1])
```

## 位运算

### 231. 2 的幂

2 的幂首先是大于 0 的，其次在二进制表示中只有 1 个 1 ，所以我们可以用 `n&(n-1)` 把最后一个 1 消去看是否结果为 0 。

```python
class Solution:
    def isPowerOfTwo(self, n: int) -> bool:
        return n > 0 and n&(n-1) == 0
```

另外 $2^{31}$ 去取余任何 2 的幂结果应该都为 0 ，所以也可以利用这个性质。

```python
class Solution:
    def isPowerOfTwo(self, n: int) -> bool:
        return (2<<31)%n == 0 if n > 0 else False
```

### 191. 位1的个数

跟上题一样，用 `n&(n-1)` 的方式统计计数。

```python
class Solution:
    def hammingWeight(self, n: int) -> int:
        res = 0
        while n:
            n &= n-1
            res += 1
        return res
```

### 190. 颠倒二进制位

使用字符串去操作。

```python
class Solution:
    def reverseBits(self, n: int) -> int:
        return (int(bin(n)[:1:-1].ljust(32, '0'), 2))
```

逐位颠倒累加。原理有点类似与比如一个字符串 `1234` ，要转成 10 进制数，如果要从前往后遍历的话，每一步都是 `res = res*10 + string[i]` ，相当于经历 0+1， 10+2， 120+3， 1230+4 这个过程。这里可以累加 32 次。

```python
class Solution:
    def reverseBits(self, n: int) -> int:
        res = 0
        for _ in range(32):
            res = (res<<1)|(n&1)
            n >>= 1
        return res
```

### 136. 只出现一次的数字

这道题主要是位运算 异或 这个操作的理解，将所有元素做异或运算，出现两次的元素异或结果为 0 ，最后得到的就是只出现一次的数字。

```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        res = 0
        for i in nums:
            res = res^i
        return res
```

或者用 `reduce` 函数。

```python
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        return reduce(lambda x,y:x^y, nums)
```

# 算法基础

## 二分查找

### 34. 在排序数组中查找元素的第一个和最后一个位置

用二分法模拟过程就行。

```python
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        nums_len = len(nums)
        left = 0
        right = nums_len - 1
        start = -1
        end = -1
        while left <= right:
            mid = (left+right) // 2
            if nums[mid] == target:
                t_s = mid
                t_e = mid
                while t_s>=0 and nums[t_s] == target:
                    t_s -= 1
                while t_e<nums_len and nums[t_e] == target:
                    t_e += 1
                start = t_s + 1
                end = t_e - 1
                return [start, end]
            elif nums[mid] > target:
                right = mid - 1
            else:
                left = mid + 1
        return [start, end]
```

### 33. 搜索旋转排序数组

第一种解法是暴力。

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        res = -1
        for i,v in enumerate(nums):
            if v == target:
                res = i
                break
        return res
```

第二种解法，先用二分法找出 `k` 值，再用二分法找出 `target` 目标值。

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        if len(nums) == 1:
            return 0 if nums[0] == target else -1
        k = 0
        nums_len = len(nums)
        l = 0
        r = nums_len - 1
        lv = nums[l]
        rv = nums[r]
        while r != l+1:
            mid = (l+r) // 2
            if nums[mid] > lv:
                l = mid
            elif nums[mid] < rv:
                r = mid
        k = r if nums[r]<nums[l] else 0
        nums = nums[k:]+nums[:k] if k != 0 else nums
        l = 0
        r = nums_len - 1
        while l<=r:
            mid = (l+r) // 2
            if nums[mid] == target:
                res = mid + k if k != 0 else mid
                if res >= nums_len:
                    res -= nums_len
                return res
            elif nums[mid] > target:
                r = mid - 1
            else:
                l = mid + 1
        return -1
```

### 74. 搜索二维矩阵

矩阵的二分法，重点在于建立 `row, col` 与 `index` 的对应关系。

```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        m = len(matrix)
        n = len(matrix[0])
        l = 0
        r = m*n-1
        while l <= r:
            mid = (l+r) // 2
            row = mid//n
            col = mid%n
            if matrix[row][col] == target:
                return True
            elif matrix[row][col] > target:
                r = mid - 1
            else:
                l = mid + 1
        return False
```

### 153. 寻找旋转排序数组中的最小值

其实相当于 [33. 搜索旋转排序数组](###33. 搜索旋转排序数组) 的寻找 `k` 值。

```python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return nums[0]
        k = 0
        nums_len = len(nums)
        l = 0
        r = nums_len - 1
        lv = nums[l]
        rv = nums[r]
        while r != l+1:
            mid = (l+r) // 2
            if nums[mid] > lv:
                l = mid
            elif nums[mid] < rv:
                r = mid
        k = r if nums[r]<nums[l] else 0
        return nums[k]
```

### 162. 寻找峰值

可以理解为找函数 `y=nums[x]` 的极值点。我们采用二分搜索的策略，为了避免极值点在边界，我们左右两端各添加一个 `float('-inf')` 负无穷。我们可以通过左右指针循环查找，这里的重点是控制指针的变化。

- 当 `mid` 比 `mid-1` , `mid+1` 都大时，返回它（当然由于预处理，应该返回 `mid-1` ）。
- 如果 `mid-1` , `mid` , `mid+1` 呈现单调递增趋势，说明极值点在右边。
- 否则说明极值点在左边。

```python
class Solution:
    def findPeakElement(self, nums: List[int]) -> int:
        nums = [float('-inf')] + nums + [float('-inf')]
        l, r = 0, len(nums)-1
        while l<=r:
            mid = (l+r) // 2
            if nums[mid]>nums[mid-1] and nums[mid]>nums[mid+1]:
                return mid-1
            elif nums[mid+1]>=nums[mid]>=nums[mid-1]:
                l = mid + 1
            else:
                r = mid - 1
        return None
```

## 双指针

### 82. 删除排序链表中的重复元素 II

数据结构里做过。

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

### 15. 三数之和

数据结构里做过。

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums = sorted(nums)
        nums_len = len(nums)
        res = []
        if nums_len < 3 or nums[0]>0 or nums[-1]<0:
            return res
        for i in range(nums_len-2):
            if nums[i]>0:
                break
            if i>0 and nums[i] == nums[i-1]:
                continue
            left = i+1
            right = nums_len-1
            while left<right:
                temp_sum = nums[i]+nums[left]+nums[right]
                if temp_sum > 0:
                    right -= 1
                elif temp_sum < 0:
                    left += 1
                else:
                    res.append([nums[i], nums[left], nums[right]])
                    while left<right and nums[left] == nums[left+1]: left += 1
                    while left<right and nums[right] == nums[right-1]: right -= 1
                    left += 1
                    right -= 1
        return res
```

### 844. 比较含退格的字符串

用栈模拟整个过程，遇到 `#` 时，如果栈非空就出栈，否则不管，如果遇到其它字符则进栈。最后比较两个栈是否一致。

```python
class Solution:
    def backspaceCompare(self, s: str, t: str) -> bool:
        stack_s = []
        for c in s:
            if c == '#':
                if stack_s:
                    stack_s.pop()
            else:
                stack_s.append(c)
        stack_t = []
        for c in t:
            if c == '#':
                if stack_t:
                    stack_t.pop()
            else:
                stack_t.append(c)
        return stack_s == stack_t
```

### 986. 区间列表的交集

用双指针分别控制当前遍历的 A、B 两个列表的区间。如果区间不相交，将靠后区间的指针移到下个区间。如果区间相交，那么相交起始位置就是更大的 `start` ，结束位置是更小的 `end` 。

```python
class Solution:
    def intervalIntersection(self, firstList: List[List[int]], secondList: List[List[int]]) -> List[List[int]]:
        firstList_len = len(firstList)
        secondList_len = len(secondList)
        l, r = 0, 0
        res = []
        while l<firstList_len and r<secondList_len:
            if firstList[l][0] > secondList[r][1]:
                r += 1
                continue
            if secondList[r][0] > firstList[l][1]:
                l += 1
                continue
            start = max(firstList[l][0], secondList[r][0])
            if firstList[l][1] < secondList[r][1]:
                end = firstList[l][1]
                l += 1
            else:
                end = secondList[r][1]
                r += 1
            res.append([start, end])
        return res
```

### 11. 盛最多水的容器

双指针逐渐向中间移动，优先移动高度更低的指针。

```python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        height_len = len(height)
        l, r = 0, height_len-1
        res = 0
        while l<r:
            if height[l] <= height[r]:
                res = max(res, (r-l)*height[l])
                l += 1
            else:
                res = max(res, (r-l)*height[r])
                r -= 1
        return res
```

这道题其实有一个可以优化的地方，当移动到某个程度时就不需要移动了，因为有个理论最大值，面积等于两个线中短线的高度乘两线距离，如果当前面积比理论最大高度乘两线距离还大，就可以直接返回结果了。

```python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        height_len = len(height)
        height_max = max(height)
        l, r = 0, height_len-1
        res = 0
        while l<r:
            if height[l] <= height[r]:
                res = max(res, (r-l)*height[l])
                l += 1
            else:
                res = max(res, (r-l)*height[r])
                r -= 1
            if res >= height_max*(r-l):		# 剪枝
                break
        return res
```

## 滑动窗口

### 438. 找到字符串中所有字母异位词

在编程能力里做过。

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

在编程能力里做过。

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

### 209. 长度最小的子数组

与上题有点相似，总的来说也属于双指针问题。

```python
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        nums_sum = nums[0]
        if nums_sum >= target:
            return 1
        l, r = 0, 0
        res, nums_len = float('inf'), len(nums)
        while r<nums_len:
            if nums_sum<target:
                r += 1
                if r<nums_len:
                    nums_sum += nums[r]
            else:
                res = min(res, r-l+1)
                nums_sum -= nums[l]
                l += 1
        return res if res != float('inf') else 0
```

或者更 Pythonic 一点。

```python
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        res, left, temp_sum = float('inf'), 0, 0
        for right, value in enumerate(nums):
            temp_sum += value
            while temp_sum >= target:
                temp_sum -= nums[left]
                res = min(res, right-left+1)
                left += 1
        return res if res != float('inf') else 0
```

## 广度优先搜索 / 深度优先搜索

### 200. 岛屿数量

这道题这个字符串是真有点坑，实际是个简单的搜索，深度优先搜索或广度优先搜索都可以。

```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        m = len(grid)
        n = len(grid[0])
        searched = [[0]*n for _ in range(m)]
        res = 0
        for row in range(m):
            for col in range(n):
                if grid[row][col]=='1' and not searched[row][col]:
                    res += 1
                    stack = [(row, col)]
                    while stack:
                        r, c = stack.pop()
                        searched[r][c] = 1
                        if r>0 and grid[r-1][c]=='1' and not searched[r-1][c]: stack.append((r-1, c))
                        if r+1<m and grid[r+1][c]=='1' and not searched[r+1][c]: stack.append((r+1, c))
                        if c>0 and grid[r][c-1]=='1' and not searched[r][c-1]: stack.append((r, c-1))
                        if c+1<n and grid[r][c+1]=='1' and not searched[r][c+1]: stack.append((r, c+1))
        return res
```

### 547. 省份数量

这个属于是换了一种图存储方式的搜索，本质还是深度优先搜索或广度优先搜索。

```python
class Solution:
    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        n = len(isConnected)
        searched = [0]*n
        res = 0
        for l in range(n):
            if searched[l]:
                continue
            res += 1
            searched[l] = 1
            stack = [l]
            while stack:
                left = stack.pop()
                for right in range(n):
                    if isConnected[left][right] and not searched[right]:
                        searched[right] = 1
                        stack.append(right)
        return res
```

### 117. 填充每个节点的下一个右侧节点指针 II

二叉树的层次遍历思想，首先想到的是用一个 `list` 来存储每一层的节点，第一个节点指向第二个节点，第二个节点指向第三个节点，依次类推，最后一个节点指向 `None` 不用管。

```python
class Solution:
    def connect(self, root: 'Node') -> 'Node':
        if root is None:
            return root
        queue = [root]
        while queue:
            temp = []
            for _ in range(len(queue)):
                node = queue.pop(0)
                temp.append(node)
                if node.left: queue.append(node.left)
                if node.right: queue.append(node.right)
            for i in range(len(temp)-1):
                temp[i].next = temp[i+1]
        return root
```

上面是很有优化空间的，实际上在出队的时候就可以判断有没有节点的 `next` 指向出队那个节点。

```python
class Solution:
    def connect(self, root: 'Node') -> 'Node':
        if root is None:
            return root
        queue = [root]
        while queue:
            temp = None
            for _ in range(len(queue)):
                node = queue.pop(0)
                if temp:
                    temp.next = node
                temp = node
                if node.left: queue.append(node.left)
                if node.right: queue.append(node.right)
        return root
```

### 572. 另一棵树的子树

一棵树是另一棵树的子树，满足以下三种情况之一：

- 两棵树相同。
- 这棵树是另一棵树左子树的子树。
- 这棵树是另一棵树右子树的子树。

```python
class Solution:
    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        def isSametree(roota, rootb):
            if roota and rootb:
                return roota.val == rootb.val and isSametree(roota.left, rootb.left) and isSametree(roota.right, rootb.right)
            if roota or rootb:
                return False
            return True

        if not (root or subRoot):
            return True
        if not root and subRoot:
            return False
        return isSametree(root, subRoot) or self.isSubtree(root.left, subRoot) or self.isSubtree(root.right, subRoot)
```

其实这个很类似字符串匹配，因为我们显然也可以通过稍加修改的遍历去转换成一个字符串匹配的问题，这个稍加修改就是空节点的值要记为空。

```python
class Solution:
    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        def serialize(root):
            if not root:
                return '()'
            return f'({serialize(root.left)}{root.val}{serialize(root.right)})'
        
        return serialize(subRoot) in serialize(root)
```

### 1091. 二进制矩阵中的最短路径

广度优先搜索，不过这道题让我印象更深刻的不是广搜。而是可变类型与不可变类型，这道题我建立 `visited` 时最开始用的 `[[0]*n]*n` 发现怎么也不对。因为这里内层 `n` 个列表都是同一个列表。

```python
class Solution:
    def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:
        if grid[0][0] or grid[-1][-1]:
            return -1
        queue = collections.deque()
        queue.append((0, 0, 1))
        direction = ((-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1))
        n = len(grid)
        visited = [[0]*n for i in range(n)]
        visited[0][0] = 1
        while queue:
            r, c, count = queue.popleft()
            if r==n-1 and c==n-1:
                return count
            for dx, dy in direction:
                x = r+dx
                y = c+dy
                if 0<=x<n and 0<=y<n and grid[x][y]==0 and visited[x][y]==0:
                    queue.append((x, y, count+1))
                    visited[x][y] = 1
        return -1
```

### 130. 被围绕的区域

从边框上的 `O` 点开始广度优先搜索， 4 个方向上下左右，能搜索到的 `O` 点记录下来，其它点改为 `X` 。

```python
class Solution:
    def solve(self, board: List[List[str]]) -> None:
        hashset = set('')
        m, n = len(board), len(board[0])
        direction = ((-1, 0), (1, 0), (0, -1), (0, 1))
        border = [(i,0) for i in range(m)]+[(i,n-1) for i in range(m)]+[(0,i) for i in range(n)]+[(m-1,i) for i in range(n)]
        for r,c in border:
            if board[r][c] == 'O' and (r,c) not in hashset:
                queue = collections.deque()
                queue.append((r,c))
                hashset.add((r,c))
                while queue:
                    row, col = queue.popleft()
                    for dx, dy in direction:
                        x = row+dx
                        y = col+dy
                        if -1<x<m and -1<y<n and board[x][y]=='O' and (x,y) not in hashset:
                            queue.append((x,y))
                            hashset.add((x,y))
        for r in range(m):
            for c in range(n):
                if (r,c) not in hashset:
                    board[r][c] = 'X'
```

### 797. 所有可能的路径

广度优先搜索并记录。

```python
class Solution:
    def allPathsSourceTarget(self, graph: List[List[int]]) -> List[List[int]]:
        queue = collections.deque()
        queue.append((0, [0]))
        n = len(graph)
        res = []
        while queue:
            node, path = queue.popleft()
            for _next in graph[node]:
                if _next == n-1:
                    res.append(path+[n-1])
                else:
                    queue.append((_next, path+[_next]))
        return res
```

深度优先搜索。

```python
class Solution:
    def allPathsSourceTarget(self, graph: List[List[int]]) -> List[List[int]]:
        n = len(graph)
        def dfs(node):
            if node == n-1:
                return [[node]]
            res = []
            for t in graph[node]:
                for i in dfs(t):
                    res.append([node]+i)
            return res
        return dfs(0) 
```

## 递归 / 回溯

### 78. 子集

求出所有可能长度的组合，因此可以使用 `itertools.combinations` 来根据不同长度求出组合。

```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        res = [[]]
        for i in range(1, len(nums)+1):
            temp = itertools.combinations(nums, i)
            for t in temp:
                res.append(list(t))
        return res
```

回溯。

```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        res = []
        n = len(nums)
        
        def backtracking(start, path):
            res.append(path)
            for i in range(start, n):
                backtrack(i+1, path+[nums[i]])
                
        backtrack(0, [])
        return res
```

### 90. 子集 II

与上题不一样之处在于通过排序达到去重效果。

```python
class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        res = []
        n = len(nums)
        nums = sorted(nums)

        def backtracking(start, path):
            res.append(path)
            for i in range(start, n):
                if i > start and nums[i] == nums[i-1]:
                    continue
                backtracking(i+1, path+[nums[i]])

        backtracking(0, [])
        return res
```

### 47. 全排列 II

调用 `itertools.permutations` 函数，如果不重复就添加到结果里。

```python
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        res = []
        for t in itertools.permutations(nums):
            temp = list(t)
            if temp not in res:
                res.append(temp)
        return res
```

添加一个 `used` 数组记录此次递归元素是否被使用，进而保证没有重复的全排列。

```python
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        res = []
        n = len(nums)
        used = [0]*n
        nums = sorted(nums)

        def backtracking(used, path):
            if len(path) == n:
                res.append(path)
                return None
            for i in range(n):
                if not used[i]:
                    if i>0 and nums[i] == nums[i-1] and not used[i-1]:
                        continue
                    used[i] = 1
                    backtracking(used, path+[nums[i]])
                    used[i] = 0
        
        backtracking(used, [])
        return res
```

### 39. 组合总和

回溯的时候可以重复取数，如果当前和大于目标值就进行剪枝。

```python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        res = []
        n = len(candidates)

        def backtracking(start, path, temp_sum):
            if temp_sum == target:
                res.append(path)
            if temp_sum > target:
                return None
            for i in range(start, n):
                backtracking(i, path+[candidates[i]], temp_sum+candidates[i])

        backtracking(0, [], 0)
        return res
```

### 40. 组合总和 II

与上题不一样的地方在于针对重复情况做特殊处理。

```python
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        res = []
        n = len(candidates)
        candidates = sorted(candidates)

        def bcaktracking(start, path, temp_sum):
            if temp_sum == target:
                res.append(path)
            if temp_sum > target:
                return None
            for i in range(start, n):
                if i>start and candidates[i] == candidates[i-1]:
                    continue
                bcaktracking(i+1, path+[candidates[i]], temp_sum+candidates[i])

        bcaktracking(0, [], 0)
        return res
```

### 17. 电话号码的字母组合

回溯的实际应用。

```python
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        hashmap = {
            '2': ('a', 'b', 'c'),
            '3': ('d', 'e', 'f'),
            '4': ('g', 'h', 'i'),
            '5': ('j', 'k', 'l'),
            '6': ('m', 'n', 'o'),
            '7': ('p', 'q', 'r', 's'),
            '8': ('t', 'u', 'v'),
            '9': ('w', 'x', 'y', 'z')
        }
        res = []
        n = len(digits)
        if n == 0:
            return res

        def backtracking(start, path):
            if len(path) == n:
                res.append(''.join(path))
            for i in range(start, n):
                for t in hashmap[digits[i]]:
                    backtracking(i+1, path+[t])
        
        backtracking(0, [])
        return res
```

### 22. 括号生成

这次递归点在于去生成左括号还是右括号，而不是一对括号。

```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        res = []
        
        def backtracking(l_count, r_count, path):
            if l_count<0 or r_count<0 or r_count<l_count:
                return None
            if l_count == 0 and r_count == 0:
                res.append(path)
                return None
            backtracking(l_count-1, r_count, path+'(')
            backtracking(l_count, r_count-1, path+')')

        backtracking(n, n, '')
        return res
```

### 79. 单词搜索

对每个点递归搜索，对特定情况进行剪枝。

```python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        if not board:
            return False
        m = len(board)
        n = len(board[0])
        word_len = len(word)
        searched = [[0]*n for _ in range(m)]

        def backtracking(r, c, loc):
            if 0<=r<m and 0<=c<n and loc<word_len and not searched[r][c] and board[r][c]==word[loc]:
                if loc == word_len-1:
                    return True
                else:
                    loc += 1
                    searched[r][c] = 1
                    res = backtracking(r-1, c, loc) or backtracking(r+1, c, loc) or backtracking(r, c-1, loc) or backtracking(r, c+1, loc)
                    searched[r][c] = 0
                    return res
            else:
                return False

        for r in range(m):
            for c in range(n):
                if backtracking(r, c, 0):
                    return True
        return False
```

## 动态规划

### 213. 打家劫舍 II

分两种情况讨论。分别是取头不取尾和取尾不取头这两种情况。

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        if len(nums)<3:
            return max(nums)
        
        def rob1(nums: List[int]) -> int:
            n = len(nums)
            if n == 1:
                return nums[0]
            if n == 2:
                return max(nums[0], nums[1])
            dp = [nums[0], max(nums[0], nums[1])]
            for i in range(2, n):
                dp.append(max(dp[i-2]+nums[i], dp[i-1]))
            return dp[-1]
       
        return max(rob1(nums[0:-1]), rob1(nums[1:]))
```

### 55. 跳跃游戏

动态更新当前位置能够到达的最远位置，然后移动当前位置，但是当前位置有一个条件就是要小于上一个当前位置能够到达的最远位置。

```python
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        n = len(nums)
        if n == 1:
            return True
        max_len = nums[0]
        i = 0
        while i <= max_len:
            if i+nums[i] > max_len:
                max_len = i+nums[i]
            if max_len >= n-1:
                return True
            i += 1
        return False
```

### 45. 跳跃游戏 II

将步数和覆盖范围联系起来，求相同覆盖范围覆盖当前位置的最小步数。

```python
class Solution:
    def jump(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return 0
        max_len = 0
        cur_len = 0
        res = 0
        for i in range(n):
            max_len = max(max_len, i+nums[i])
            if i == cur_len:
                if cur_len != n-1:
                    res += 1
                    cur_len = max_len
                    if cur_len >= n-1:
                        break
        return res
```

### 62. 不同路径

最开始尝试了下深度优先搜索，超时了。

很显然这道题不需要搜索，而是一道动态规划，状态转移方程如下：
$$
dp[i][j]=\begin{cases}dp[i-1][j]+dp[i][j-1]\ \ \ \ i>0\&j>0\\1\ \ \ \ i=0|j=0\end{cases}
$$

```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [[1]*n] + [[0]*n for _ in range(m-1)]
        for i in range(m): dp[i][0] = 1
        for r in range(1, m):
            for c in range(1, n):
                dp[r][c] = dp[r-1][c]+dp[r][c-1]
        return dp[-1][-1]
```

### 5. 最长回文子串

数据结构里做过。

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

### 413. 等差数列划分

假设一个数组为等差数列的数组，那么它的等差数列子数组的个数等于 $\sum_{1}^{nums\_len-3}$ 。对于一个不是等差数列的数组，这个数组等差数列子数组则等于其中所有最长等差数列子数组的等差数列子数组个数的和，可能有点难理解，举例，`[1,3,5,10,12,13,14]` 这个数组的等差数列子数组和是等于 `[1,3,5]` 的等差数列子数组 1 加上 `[12,13,14]` 的等差数列子数组 1 等于 2 的。

```python
class Solution:
    def numberOfArithmeticSlices(self, nums: List[int]) -> int:
        nums_len = len(nums)
        if nums_len < 3:
            return 0
        temp = 0
        res = 0
        for i in range(2,nums_len):
            if nums[i]+nums[i-2] == nums[i-1]+nums[i-1]:
                temp += 1
                res += temp
            else:    
                temp = 0
        return res
```

### 91. 解码方法

需要对 5 种情况进行处理：

- 当前字符是 1 或 2 ，此时有点像是斐波那契数列。
- 当前字符是 0 ，前面字符是 1 或 2 ，会导致前面那个字符衍生的结果被影响。
- 当前字符是 0 ，前面字符不是 1 或 2 ，解码不了。
- 当前字符是其他数字，如果这一位和上一位的组合起来比 26 大，那么这一位相当于是独立的，不会有衍生结果。
- 当前字符是其他数字，并且不是上面那种情况，这一位参与斐波那契式计算并添加至结果。

```python
class Solution:
    def numDecodings(self, s: str) -> int:
        s_len = len(s)
        temp = [0, 1]
        res = 1
        for i in range(s_len):
            if s[i] == '1' or s[i] == '2':
                temp.append(temp[-1]+temp[-2])
            elif s[i] == '0':
                if i>0 and s[i-1] in {'1','2'}:
                    res *= temp[-2]
                    temp = [0, 1]
                else:
                    return 0
            else:
                if i>0 and s[i-1] == '2' and s[i] > '6':
                    res *= temp[-1]
                    temp = [0, 1]
                else:
                    res *= temp[-1]+temp[-2]
                    temp = [0, 1]
        if temp[-1] != 1:
            res *= temp[-1]
        return res
```

### 139. 单词拆分

状态转移方程，dp[i] 代表字符串第 i 位及之前能够由字典的单词拼接成：
$$
dp[j]=\begin{cases}True\ \ \ \ if\ dp[i]\&s[i:j]\in wordDict\\False\end{cases}
$$

```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        s_len = len(s)
        dp=[True]+[False]*s_len
        for i in range(s_len):
            for j in range(i+1, s_len+1):
                if dp[i] and s[i:j] in wordDict:
                    dp[j] = True
        return dp[-1]
```

### 300. 最长递增子序列

状态转移方程为，这里 dp[i] 指的是包含第 i 个元素的状态：
$$
dp[i]=\begin{cases}max(dp[x],x\in[0,i-1]\&nums[x]<nums[i])+1\\1\ \ \ \ if\ i=0\end{cases}
$$

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        nums_len = len(nums)
        dp = [1]*nums_len
        for i in range(1,nums_len):
            for j in range(i):
                if nums[j]<nums[i]:
                    dp[i] = max(dp[i], dp[j]+1)
        return max(dp)
```

### 673. 最长递增子序列的个数

相对于上道题，这道题多了一个计数，每次增长序列时，需要重置计数，每次遇到另一组当前最长序列时，需要累加计数。

```python
class Solution:
    def findNumberOfLIS(self, nums: List[int]) -> int:
        nums_len = len(nums)
        dp = [1]*nums_len
        count = [1]*nums_len
        for i in range(1,nums_len):
            for j in range(i):
                if nums[j]<nums[i]:
                    if dp[j]+1>dp[i]:
                        dp[i] = dp[j]+1
                        count[i] = count[j]
                    elif dp[j]+1==dp[i]:
                        count[i] += count[j]
        max_seq = max(dp)
        res = 0
        for i in range(nums_len):
            if max_seq == dp[i]:
                res += count[i]
        return res
```

### 1143. 最长公共子序列

动态规划，二维状态转移方程：
$$
dp[i][j]=\begin{cases}max(dp[i][j-1],dp[i-1][j],dp[i-1][j-1]+1\ \ when\ text_i=text_j)\ \ \ \ if\ i>0\&j>0
\\dp[i-1][j]\ or\ dp[i][j-1]\ \ \ \ if\ i=0|j=0\ \&text_i\ne text_j
\\dp[i-1][j]+1\ or\ dp[i][j-1]+1\ \ \ \ if\ i=0|j=0\ \&text_i=text_j
\\1 or 0\ if\ i=0\&j=0\end{cases}
$$

```python
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        len1 = len(text1)
        len2 = len(text2)
        dp = [[0]*(len2+1) for _ in range(len1+1)]
        for i in range(len1):
            for j in range(len2):
                if text1[i] == text2[j]:
                    dp[i+1][j+1] = dp[i][j]+1
                else:
                    dp[i+1][j+1] = max(dp[i+1][j], dp[i][j+1])  
        return dp[-1][-1]
```

### 583. 两个字符串的删除操作

跟上题基本一样，重点就在于找到最长公共子序列。

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        len1 = len(word1)
        len2 = len(word2)
        dp = [[0]*(len2+1) for _ in range(len1+1)]
        for i in range(len1):
            for j in range(len2):
                if word1[i] == word2[j]:
                    dp[i+1][j+1] = dp[i][j]+1
                else:
                    dp[i+1][j+1] = max(dp[i+1][j], dp[i][j+1])  
        return len1+len2-2*dp[-1][-1]
```

### 72. 编辑距离

状态转移方程：
$$
dp[i][j]=\begin{cases}dp[i-1][j-1]，此时两个字母相同所以不用操作\\min(dp[i][j-1],dp[i-1][j],dp[i-1][j-1])+1，此时两个字母不同，选择增删改里最优方式执行\end{cases}
$$

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        len1 = len(word1)
        len2 = len(word2)
        dp = [[0]*(len2+1) for _ in range(len1+1)]
        for i in range(len1+1): dp[i][0] = i
        for j in range(len2+1): dp[0][j] = j
        for i in range(len1):
            for j in range(len2):
                if word1[i] == word2[j]:
                    dp[i+1][j+1] = dp[i][j]
                else:
                    dp[i+1][j+1] = min(dp[i][j+1], dp[i+1][j], dp[i][j])+1
        return dp[-1][-1]
```

### 322. 零钱兑换

做过。

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        if amount == 0:
            return 0
        dp = [-1]*(amount+1)
        for i in range(1, amount+1):
            if i in coins:
                dp[i] = 1
            else:
                temp = []
                for coin in coins:
                    if i-coin>0 and dp[i-coin] != -1:
                        temp.append(dp[i-coin])
                if len(temp) != 0:
                    dp[i] = min(temp)+1
        return dp[-1]
```

### 343. 整数拆分

我感觉这道题是在做数学题，找规律，找到 10 左右应该能发现规律。

```python
class Solution:
    def integerBreak(self, n: int) -> int:
        dp=[1,2,4,6,9]
        for i in range(n-6):
            dp.append(3*dp[-3])
        return dp[n-2]
```

##  位运算

### 201. 数字范围按位与

问题的本质在于找最长公共前缀。所以假设两个数位长度不一样显然公共前缀都为 0 。

```python
class Solution:
    def rangeBitwiseAnd(self, left: int, right: int) -> int:
        if len(bin(left)) != len(bin(right)):
            return 0
        res = left
        for i in range(left+1, right+1):
            res &= i
        return res
```

正常找最长前缀当然也可以。

```python
class Solution:
    def rangeBitwiseAnd(self, left: int, right: int) -> int:
        shift = 0
        while left != right:
            left >>= 1
            right >>= 1
            shift += 1
        return right<<shift
```

或者利用 Brian Kernighan 算法，一直消去 `right` 的最后一个 1 使 `right` 小于等于 `left` ，此时 `right` 就是答案。

> Brian Kernighan's Algorithm
>
> 通过 n&(n-1) 可以使 n 最后一位上的 1 变成 0 。

```python
class Solution:
    def rangeBitwiseAnd(self, left: int, right: int) -> int:
        while left<right:
            right = right&(right-1)
        return right
```

## 其他

### 384. 打乱数组

`random` 类里 `sample` 和 `shuffle` 方法的使用。

```python
class Solution:

    def __init__(self, nums: List[int]):
        import random
        self.nums = nums

    def reset(self) -> List[int]:
        return self.nums

    def shuffle(self) -> List[int]:
        return random.sample(self.nums, len(self.nums))
```

或是

```python
class Solution:

    def __init__(self, nums: List[int]):
        import random
        self.nums = nums

    def reset(self) -> List[int]:
        return self.nums

    def shuffle(self) -> List[int]:
        res = self.nums.copy()
        random.shuffle(res)
        return res
```

### 202. 快乐数

找规律题。

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

### 149. 直线上最多的点数

暴力。

```python
class Solution:
    def maxPoints(self, points: List[List[int]]) -> int:
        points_len = len(points)
        if points_len <= 2:
            return points_len
        res = 0
        for i in range(points_len):
            for j in range(i+1, points_len):
                x1,y1,x2,y2 = points[i][0],points[i][1],points[j][0],points[j][1]
                count = 2
                for k in range(j+1, points_len):
                    x,y = points[k][0], points[k][1]
                    if (y-y1)*(x2-x1) == (y2-y1)*(x-x1): count+=1
                res = max(res, count)
        return res
```

