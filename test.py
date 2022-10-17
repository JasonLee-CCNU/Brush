import random
from collections import defaultdict, deque
from sortedcontainers import SortedList
from heapq import heappop, heappush
from functools import lru_cache
from random import random
import re

from typing import Counter, List, Optional


def canFormArray(arr: List[int], pieces: List[List[int]]) -> bool:
    ans = True

    def contain(arr: List[int], pieces: List[int]) -> bool:
        i, j = 0, 0
        m, n = len(arr), len(pieces)
        while i < m and j < n:
            if arr[i] != pieces[j]:
                if 0 < j < n:
                    j = 0
                i += 1
            else:
                i += 1
                j += 1
        if j == n:
            return True
        else:
            return False

    for piece in pieces:
        if not contain(arr, piece):
            ans = False

    return ans

# print(canFormArray([1,2,3], [[2],[1,3]]))


def largestArea(grid: List[str]) -> int:
    m, n = len(grid), len(grid[0])
    begin = set()
    vis = set()
    for i in range(m):
        for j in range(n):
            if (i == 0 or j == 0) and grid[i][j] != '0' and grid[i][j] not in begin:
                begin.add((i, j))
            if grid[i][j] == '0':
                for x, y in [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]:
                    if 0 <= x < m and 0 <= y < n and grid[x][y] != '0' and grid[x][y] not in begin:
                        begin.add((x, y))
    ans = 0
    q = deque(begin)
    while q:
        i, j = q.popleft()
        if (i, j) not in vis:
            vis.add((i, j))
            v = grid[i][j]
            for x, y in [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]:
                if 0 <= x < m and 0 <= y < n and v != '0' and grid[x][y] == v and (x, y) not in begin and (x, y) not in vis:
                    begin.add((x, y))
                    vis.add((x, y))
                    q.append((x, y))

    def bfs(i: int, j: int):
        q = deque()
        q.append((i, j))
        res = 0
        while q:
            l = len(q)
            res += l
            for _ in range(l):
                i, j = q.popleft()
                v = grid[i][j]
                for x, y in [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]:
                    if 0 <= x < m and 0 <= y < n and grid[x][y] == v and grid[x][y] != '0' and (x, y) not in vis:
                        vis.add((x, y))
                        q.append((x, y))
        return res

    for i in range(m):
        for j in range(n):
            if (i, j) not in vis and grid[i][j] != '0':
                vis.add((i, j))
                res = bfs(i, j)
                ans = max(ans, res)
    return ans

# print(largestArea(["111","222","333"]))


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def distributeCoins(root: Optional[TreeNode]) -> int:
    res = 0

    def dfs(root: Optional[TreeNode]) -> int:
        nonlocal res
        if not root:
            return 0

        left = dfs(root.left)
        right = dfs(root.right)
        res += abs(left)
        res += abs(right)
        return root.val - 1 + left + right

    dfs(root)

    return res


class Node:
    def __init__(self, val: int) -> None:
        self.val = val
        self.next = None
        self.pre = None


class MyLinkedList:

    def __init__(self):
        self.array = []
        self.head = Node(-1)
        self.tail = Node(-1)
        self.head.next = self.tail
        self.tail.pre = self.head

    def get(self, index: int) -> int:
        if index < 0 or index >= len(self.array):
            return -1
        return self.array[index].val

    def addAtHead(self, val: int) -> None:
        node = Node(val)
        nxt = self.head.next
        self.head.next = node
        node.next = nxt
        node.pre = self.head
        self.array = [node] + self.array
        self.tail.pre = self.array[-1]

    def addAtTail(self, val: int) -> None:
        node = Node(val)
        pre = self.tail.pre
        node.pre = pre
        pre.next = node
        self.tail.pre = node
        self.array = self.array + [node]

    def addAtIndex(self, index: int, val: int) -> None:
        if index < 0:
            self.addAtHead(val)
        if index == len(self.array):
            self.addAtTail(val)

    def deleteAtIndex(self, index: int) -> None:
        pass


def zeroFilledSubarray(nums: List[int]) -> int:
    ans = 0
    n = len(nums)
    i = j = 0
    while i <= j and j < n:
        if nums[j] == 0:
            ans += (j - i) + 1
            j += 1
        else:
            j += 1
            i = j
    return ans

# print(zeroFilledSubarray([2,10,2019]))


def decrypt(code: List[int], k: int) -> List[int]:
    n = len(code)
    preSum = [0] * (2 * n + 1)
    for i in range(1, len(preSum)):
        if i <= n:
            preSum[i] = preSum[i - 1] + code[i - 1]
        else:
            preSum[i] = preSum[i - 1] + code[i - n - 1]
    res = [0] * n
    if k < 0:
        for i in range(n):
            res[i] = preSum[i + n] - preSum[i + n + k]
    if k > 0:
        for i in range(n):
            res[i] = preSum[i + k + 1] - preSum[i + 1]
    return res

# print(decrypt([2,4,9,3], -2))


def numberOf2sInRange(n: int) -> int:
    s = str(n)

    # @cache
    def f(i: int, cnt: int, is_limit: bool, is_num: bool) -> int:
        if i == len(s):
            return cnt
        res = 0
        if not is_num:  # 可以跳过当前数位
            res = f(i + 1, cnt, False, False)
        up = int(s[i]) if is_limit else 9
        for d in range(0 if is_num else 1, up + 1):  # 枚举要填入的数字 d
            res += f(i + 1, cnt + 1 if d == 2 else cnt,
                     is_limit and d == up, True)
        return res
    return f(0, 0, True, False)


def profitableSchemes(n: int, minProfit: int, group: List[int], profit: List[int]) -> int:
    m = len(group)
    mode = 10 ** 9 + 7
    dp = [[[0] * (minProfit + 1) for _ in range(n + 1)] for _ in range(m + 1)]
    for j in range(n + 1):
        dp[0][j][0] = 1

    for k in range(1, m + 1):
        nums, value = group[k - 1], profit[k - 1]
        for j in range(0, n + 1):
            for i in range(0, minProfit + 1):
                f = dp[k - 1][j][i]
                if j >= nums:
                    f += dp[k - 1][j - nums][max(i - value, 0)]
                dp[k][j][i] = f % mode
    return dp[m][n][minProfit]

# print(profitableSchemes(5,3,[2,2],[2,3]))


def matchPlayersAndTrainers(players: List[int], trainers: List[int]) -> int:
    players.sort()
    trainers.sort()
    ans = 0
    p, t = len(players), len(trainers)
    i, j = 0, 0
    while i < p and j < t:
        if players[i] <= trainers[j]:
            ans += 1
            i += 1
            j += 1
        else:
            j += 1
    return ans

# print(matchPlayersAndTrainers([1,1,1], [10]))


def goodIndices(nums: List[int], k: int) -> List[int]:
    n = len(nums)
    f = [0] * n
    b = [0] * n
    ans = []
    f[0] = 1
    b[-1] = 1
    for i in range(1, n):
        num = nums[i]
        if num <= nums[i - 1]:
            f[i] = f[i - 1] + 1
        else:
            f[i] = 1

    for i in range(n - 2, -1, -1):
        num = nums[i]
        if num <= nums[i + 1]:
            b[i] = b[i + 1] + 1
        else:
            b[i] = 1
    for i in range(k, n - k):
        if f[i - 1] >= k and b[i + 1] >= k:
            ans.append(i)
    return ans

# print(goodIndices([878724,201541,179099,98437,35765,327555,475851,598885,849470,943442], 4))


def kSmallestPairs(nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
    m, n = len(nums1), len(nums2)
    q = [(nums1[i] + nums2[0], i, 0) for i in range(m)]
    ans = []
    while len(ans) < k:
        if q:
            top = heappop(q)
            i, j = top[1], top[2]
            ans.append([nums1[i], nums2[j]])
            if j < n - 1:
                heappush(q, (nums1[i] + nums2[j + 1], i, j + 1))
        else:
            break

    return ans

# print(kSmallestPairs([1,2], [3], 3))


def btreeGameWinningMove(root: Optional[TreeNode], n: int, x: int) -> bool:
    lcnt, rcnt = 0, 0

    def dfs(root, val: int):
        nonlocal lcnt, rcnt
        if not root:
            return 0
        l = dfs(root.left, val)
        r = dfs(root.right, val)
        if root.val == val:
            lcnt, rcnt = l, r
            return 0
        return l + r + 1

    dfs(root, x)
    if lcnt + rcnt + 1 < n - (lcnt + rcnt + 1):
        return True
    if lcnt > n - lcnt or rcnt > n - rcnt:
        return True
    return False


def maxValue(root: TreeNode, k: int) -> int:

    def dfs(root):
        res = [0] * (k + 1)

        if not root:
            return res
        l = dfs(root.left)
        r = dfs(root.right)
        res[0] = max(l) + max(r)

        for i in range(k):
            for j in range(k - i):
                if l[i] + r[j] + root.val > res[i + j + 1]:
                    res[i + j + 1] = l[i] + r[j] + root.val
        return res

    return max(dfs(root))


def canTransform(start: str, end: str) -> bool:
    m, n = len(start), len(end)

    i, j = 0, 0
    while i < m and j < n:
        s, e = start[i], end[j]
        if s == 'L':
            if e == 'R':
                return False
            if e == 'X':
                j += 1
            if e == 'L':
                if i < j:
                    return False
                i += 1
                j += 1
        if s == 'R':
            if e == 'R':
                if i > j:
                    return False
                i += 1
                j += 1
            if e == 'X':
                j += 1
            if e == 'L':
                return False
        if s == 'X':
            if e == 'R':
                i += 1
            if e == 'L':
                i += 1
            if e == 'X':
                i += 1
                j += 1
    if i == m:
        for a in range(j, n):
            if end[a] != 'X':
                return False
        return True
    if j == n:
        for b in range(i, m):
            if start[b] != 'X':
                return False
        return True

# print(canTransform("RXXLRXRXL", "XRLXXRRLX"))


def subdomainVisits(cpdomains: List[str]) -> List[str]:
    hashmap = dict()
    for s in cpdomains:
        a, b = s.split(' ')
        a = int(a)
        tmp = b.split('.')
        sb = ""
        for i in range(len(tmp) - 1, -1, -1):
            if i < len(tmp) - 1:
                sb = tmp[i] + '.' + sb
            else:
                sb = tmp[i] + sb

            hashmap[sb] = hashmap.get(sb, 0) + a

    ans = []
    for k, v in hashmap.items():
        ans.append(str(v) + " " + k)
    return ans

# print(subdomainVisits(["900 google.mail.com", "50 yahoo3.com", "1 intel.mail.com", "5 wiki.org"]))


def isFlipedString(s1: str, s2: str) -> bool:
    if len(s1) <= 1:
        return s1 == s2
    for i, char in enumerate(s1):
        if s1[i + 1:] + s1[:i + 1] == s2:
            return True
    return False


class Cashier:

    def __init__(self, n: int, discount: int, products: List[int], prices: List[int]):
        self.n, self.discount = n, discount
        self.cnt = 0
        self.products = dict()
        for i, p in enumerate(products):
            self.products[p] = prices[i]

    def getBill(self, product: List[int], amount: List[int]) -> float:
        nums = 0
        for i, p in enumerate(product):
            nums += self.products[p] * amount[i] * 1.0
        if self.cnt == self.n:
            nums = nums - (nums * self.discount) / 100
            self.cnt = 0
        else:
            self.cnt += 1
        return nums

# [[192,34,[77],[302]],[[77],[343]],[[77],[990]],[[77],[101]]]


def reformatNumber(number: str) -> str:
    subNumber = re.sub("\D", "", number)
    if len(subNumber) <= 3:
        return subNumber
    else:
        tmpStr = []
        i, n = 0, len(subNumber)
        while i < n:
            if n - i == 4:
                tmpStr.append(subNumber[i: i + 2])
                tmpStr.append(subNumber[i + 2:])
                break
            if n - i == 2:
                tmpStr.append(subNumber[i:])
                break
            else:
                tmpStr.append(subNumber[i: i + 3])
                i += 3
        return '-'.join(tmpStr)


def pileBox(box: List[List[int]]) -> int:
    box.sort()
    n = len(box)
    dp = [box[i][2] for i in range(n)]
    ans = dp[0]
    for i in range(1, n):
        tmp = 0
        for j in range(i):
            if box[j][0] < box[i][0] and box[j][1] < box[i][1] and box[j][2] < box[i][2]:
                # print("in", i, j)
                tmp = max(tmp, dp[j])
        dp[i] += tmp
        ans = max(ans, dp[i])
    return ans


def checkOnesSegment(s: str) -> bool:
    n = len(s)
    i = j = 0
    res = 0
    while i < n and j <= i:
        if res > 1:
            return False
        if s[i] == '1':
            j = i + 1
            while j < n and s[j] == '1':
                j += 1
            res += 1
            i = j
        else:
            i += 1
    return True if res <= 1 else False


def scoreOfParentheses(s: str) -> int:
    score = 0
    stack = []
    for i, char in enumerate(s):
        if not stack:
            if char == '(':
                stack.append([i, 0])

        else:
            if char == ')':
                if i - stack[-1][0] == 1:
                    stack[-1][1] = 1
                else:
                    stack[-1][1] *= 2
                node = stack.pop()
                if not stack:
                    score += node[1]
                else:
                    stack[-1][1] += node[1]
            else:
                stack.append([i, 0])

    return score

# print(scoreOfParentheses("(()(()))"))


class Solution:

    def __init__(self, nums: List[int]):
        self.nums = nums

    def reset(self) -> List[int]:
        return self.nums

    def shuffle(self) -> List[int]:
        copy = self.nums.copy()
        random.shuffle(copy)
        return copy


def maxAscendingSum(nums: List[int]) -> int:
    res = nums[0]
    tmp = nums[0]
    for j in range(1, len(nums)):
        if nums[j] > nums[j - 1]:
            tmp += nums[j]
        else:
            tmp = nums[j]
        res = max(res, tmp)
    return res


def minSwap(nums1: List[int], nums2: List[int]) -> int:
    a, b = 0, 1
    n = len(nums1)
    for i in range(1, n):
        a1, b1 = a, b
        if nums1[i] > nums1[i - 1] and nums2[i] > nums2[i - 1] and nums1[i] > nums2[i - 1] and nums2[i] > nums1[i - 1]:
            a = min(a1, b1)
            b = min(a1, b1) + 1
        elif nums1[i] > nums1[i - 1] and nums2[i] > nums2[i - 1]:
            a = a1
            b = b1 + 1
        elif nums1[i] > nums2[i - 1] and nums2[i] > nums1[i - 1]:
            a = b1
            b = a1 + 1
    return min(a, b)

# print(minSwap([1,3,5,4], [1,2,3,7]))


def areAlmostEqual(s1: str, s2: str) -> bool:

    ans = []
    for i in range(len(s1)):
        if s1[i] != s2[i]:
            ans.append([s1[i], s2[i]])

    if len(ans) == 0:
        return True
    if len(ans) == 2 and ans[0] == ans[1][::-1]:
        return True
    return False

# Definition for singly-linked list.


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def nextLargerNodes(self, head: Optional[ListNode]) -> List[int]:
        stack = list()
        ans = [0] * (10 ** 4)
        i = 0
        cur = head
        while cur:
            while stack and stack[-1][0] < cur.val:
                node = stack.pop()
                ans[node[1]] = cur.val
            stack.append((cur.val, i))
            i += 1
            cur = cur.next
        return ans[:i]


def hIndex(citations: List[int]) -> int:
    mi, ma = 1001, -1
    for cita in citations:
        if cita < mi:
            mi = cita
        if cita > ma:
            ma = cita
    count = [[0, 0] for _ in range(ma - mi + 1)]
    for cita in citations:
        if not count[cita - mi]:
            count[cita - mi] = [0, 0]
        count[cita - mi][0] += 1
        count[cita - mi][1] = cita
    count = [c for c in count if c[0]]
    hindex = 0
    preSum = [0] * (len(count) + 1)
    for i in range(1, len(count) + 1):
        preSum[i] = preSum[i - 1] + count[i - 1][0]
    for i in range(len(count)):
        for j in range(preSum[i], preSum[i + 1]):
            if count[i][0] != 0 and preSum[-1] - j <= count[i][1]:
                if i == 0:
                    if j == preSum[i] and 0 <= preSum[-1] - j:
                        hindex = max(hindex, preSum[-1] - j)
                    if j > preSum[i] and count[i][1] <= preSum[-1] - j:
                        hindex = max(hindex, preSum[-1] - j)
                else:
                    if j == preSum[i] and count[i - 1][1] <= preSum[-1] - preSum[i]:
                        hindex = max(hindex, preSum[-1] - preSum[i])
                    if j > preSum[i] and count[i][1] <= preSum[-1] - j:
                        hindex = max(hindex, preSum[-1] - j)
    return hindex

# print(hIndex([1,1]))


def numComponents(head: Optional[ListNode], nums: List[int]) -> int:
    ans = 0
    nums = set(nums)
    pre = ListNode(-1)
    cur = head
    pre.next = cur
    while cur:
        if cur.val not in nums and pre.val in nums:
            ans += 1
        pre = cur
        cur = cur.next
    return ans + 1 if pre.val in nums else ans

# [3,4,0,2,1]
# [4]


def camelMatch(queries: List[str], pattern: str) -> List[bool]:
    patterns = []
    up0 = []
    i = 0
    m = len(pattern)
    for j in range(m):
        if pattern[j].isupper():
            patterns.append(pattern[i: j])
            up0.append(pattern[j])
            i = j + 1
    if i < m:
        patterns.append(pattern[i:])
    else:
        patterns.append("")

    qs = []
    up1 = []
    n = len(queries)
    for q in queries:
        query = []
        up = []
        n1 = len(q)
        i = 0
        for j in range(n1):
            if q[j].isupper():
                query.append(q[i: j])
                up.append(q[j])
                i = j + 1
        if i < n1:
            query.append(q[i:])
        else:
            query.append("")
        qs.append(query)
        up1.append(up)

    def contains(s1, s2):
        if s2 == "":
            return True
        if len(s2) > len(s1):
            return False
        c1 = Counter(s1)
        c2 = Counter(s2)
        for k, v in c2.items():
            if k not in c1 or v > c1[k]:
                return False
        return True
    ans = [False] * n
    for index, q in enumerate(qs):
        if len(q) == len(patterns) and up1[index] == up0:
            flag = True
            for i in range(len(patterns)):
                if not contains(q[i], patterns[i]):
                    flag = False
                    break
            if flag:
                ans[index] = True
    return ans

# print(camelMatch(["CompetitiveProgramming","CounterPick","ControlPanel"], "CooP"))


def longestSubsequence(arr: List[int], difference: int) -> int:
    n = len(arr)
    hashmap = {}
    dp = [1] * n
    res = 1
    hashmap[arr[0]] = 1
    for i in range(1, n):
        pre = arr[i] - difference
        if pre in hashmap:
            dp[i] = max(hashmap[pre] + 1, dp[i])
        res = max(res, dp[i])
        if arr[i] not in hashmap:
            hashmap[arr[i]] = dp[i]
        else:
            hashmap[arr[i]] = max(hashmap[arr[i]], dp[i])
    return res

# print(longestSubsequence([1,5,7,8,5,3,4,2,1], -2))


def combinationSum4(nums: List[int], target: int) -> int:
    hashmap = {}

    def dfs(target):
        if target == 0:
            return 1
        res = 0
        for num in nums:
            if target - num < 0:
                continue
            if target - num in hashmap:
                res += hashmap[target - num]
            else:
                res += dfs(target - num)
        hashmap[target] = res
        return res

    res = dfs(target)
    return res


def lastStoneWeightII(stones: List[int]) -> int:
    n = len(stones)
    s = sum(stones)
    t = s // 2
    dp = [0] * (t + 1)
    for i in range(1, n + 1):
        v = stones[i - 1]
        for j in range(t, v - 1, -1):
            dp[j] = max(dp[j], dp[j - v] + v)
    return abs(s - dp[t] - dp[t])


def distinctSubseqII(s: str) -> int:
    mod = 10 ** 9 + 7
    n = len(s)
    dp = [[0, 0] for _ in range(n)]
    dp[0] = [0, 1]
    hashmap = dict()
    hashmap[s[0]] = 0
    for i in range(1, n):
        dp[i][0] += dp[i - 1][0] + dp[i - 1][1]
        dp[i][0] %= mod
        if s[i] not in hashmap:
            dp[i][1] += 1
        dp[i][1] += dp[i - 1][0] + dp[i - 1][1]
        dp[i][1] %= mod
        if s[i] in hashmap and hashmap[s[i]] != i:
            dp[i][1] -= dp[hashmap[s[i]]][0]
        hashmap[s[i]] = i

    return (dp[-1][0] + dp[-1][1]) % mod

# print(distinctSubseqII("bcbbca"))


def buildArray(target: List[int], n: int) -> List[str]:
    ans = []
    index, m = 0, len(target)
    for i in range(1, n + 1):
        if index == m:
            break
        if i == target[index]:
            ans.append("Push")
            index += 1
        else:
            ans.append("Push")
            ans.append("Pop")
    return ans


def medianSlidingWindow(nums: List[int], k: int) -> List[float]:
    n = len(nums)
    q = SortedList([])
    i = j = 0
    ans = []
    while j < n:
        l = len(q)
        if l == k:
            if l & 1:
                ans.append(q[l // 2])
            else:
                ans.append((q[l // 2] + q[l // 2 - 1]) / 2)
        while j - i >= k and i <= j:
            q.remove(nums[i])
            i += 1
        q.add(nums[j])
        j += 1
    if k & 1:
        ans.append(q[k // 2])
    else:
        ans.append((q[k // 2] + q[k // 2 - 1]) / 2)
    return ans

# print(medianSlidingWindow([1,3,-1,-3,5,3,6,7], 3))


def tupleSameProduct(nums: List[int]) -> int:
    hashmap = defaultdict(list)
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            tmp  =nums[i] * nums[j]
            hashmap[tmp].append((nums[i], nums[j]))
            hashmap[tmp].append((nums[j], nums[i]))
    res = 0
    for _, v in hashmap.items():
        res += len(v) * (len(v) - 2)
    return res

# print(tupleSameProduct([1,2,4,5,10]))

def minChanges(nums: List[int], k: int) -> int:
    maxValue = 0x3f3f3f3f
    n = len(nums)
    dp = [[maxValue] * 1024 for _ in range(k)]
    g = [maxValue] * k
    for i in range(k):
        hashmap = {}
        cnt = 0
        for j in range(i, n, k):
            hashmap[nums[j]] = hashmap.get(nums[j], 0) + 1
            cnt += 1
        if i == 0:
            for p in range(1024):
                dp[i][p] = min(dp[i][p], cnt - hashmap.get(p, 0))
                g[i] = min(g[i], dp[i][p])
        else:
            for p in range(1024):
                dp[i][p] = g[i - 1] + cnt
                for cur in hashmap.keys():
                    dp[i][p] = min(dp[i][p], dp[i-1][p ^ cur] + cnt - hashmap.get(cur))
                g[i] = min(g[i], dp[i][p])

    return dp[k-1][0]

# print(minChanges([1,2,0,3,0], 1))


def atMostNGivenDigitSet(digits: List[str], n: int) -> int:
    s = str(n)
    k = len(s)
    
    @lru_cache
    def f(i: int, is_limit: bool, is_num: bool):
        if i == k:
            if is_num:
                return 1
            else:
                return 0
        res = 0
        if not is_num:
            res = f(i + 1, False, False)
        up = s[i] if is_limit else '9'
        for d in digits:
            if d < up:
                res += f(i + 1, False, True)
            elif d == up:
                res += f(i + 1, is_limit, True)
            else:
                break
        return res
    
    res = f(0, True, False)
    return res

print(atMostNGivenDigitSet( ["1","4","9"], 100))