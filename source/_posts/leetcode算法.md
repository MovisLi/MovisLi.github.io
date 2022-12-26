---
title: 「算法」 - 学习计划
date: 2022-12-26 16:02:31
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
        def permutation():
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

