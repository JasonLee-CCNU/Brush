from fractions import Fraction
from heapq import heapify, heappop, heappush, heapreplace
from inspect import iscoroutine
import random
from unittest import registerResult
from sortedcontainers import SortedDict, SortedList, SortedKeyList
from functools import lru_cache
from bisect import bisect_left
from cmath import log, sqrt
from collections import Counter, defaultdict
from genericpath import sameopenfile
import math
from textwrap import wrap
from this import s
from tkinter.tix import Tree
from typing import Deque, List, Type
import re
from collections import deque
import collections
#from curses import window
from itertools import count
from math import degrees, floor
import numbers
from operator import le, ne
import queue
from secrets import choice
from sys import maxsize
from turtle import left, position, right


def longestSubarray(nums) -> int:
    invalid = 1
    left = right = 0
    ans = 0
    while right < len(nums):
        if nums[right] != 1:
            invalid -= 1
        if invalid < 0:
            ans = max(ans, right - left - 1)
            while left <= right and invalid < 0:
                if nums[left] != 1:
                    invalid += 1
                left += 1
        right += 1
    if invalid >= 0:
        ans = max(ans, right - left - 1)
    return ans


def longestSubstring(s: str, k: int) -> int:
    n = len(s)
    res = 0
    for t in range(1, 27):
        map = [0]*26
        T = less = 0
        left = right = 0
        while right < n:
            index = ord(s[right]) - ord('a')
            map[index] += 1
            if map[index] == 1:
                T += 1
                less += 1
            if map[index] == k:
                less -= 1
            while T > t:
                if left > right:
                    break
                index_1 = ord(s[left]) - ord('a')
                if map[index_1] == 1:
                    T -= 1
                    less -= 1
                if map[index_1] == k:
                    less += 1
                map[index_1] -= 1
                left += 1
            if less == 0:
                res = max(res, right - left + 1)
            right += 1

    return res


def findMinHeightTrees(n: int, edges: List[List[int]]) -> List[int]:
    res = list()
    if n == 1:
        res.append(0)
        return res
    degrees = [0 for _ in range(n)]
    map = [[] for _ in range(n)]
    for edge in edges:
        degrees[edge[0]] += 1
        degrees[edge[1]] += 1
        map[edge[0]].append(edge[1])
        map[edge[1]].append(edge[0])
    q = deque()
    for index in range(n):
        if degrees[index] == 1:
            q.append(index)
    while q:
        res = list()
        for i in range(len(q)):
            node = q.popleft()
            res.append(node)
            for neighbor in map[node]:
                degrees[neighbor] -= 1
                if degrees[neighbor] == 1:
                    q.append(neighbor)
    return res

#print('result', findMinHeightTrees(4, [[1,0],[1,2],[1,3]]))


def findAnagrams(s: str, p: str) -> List[int]:
    ans = []
    n = len(s)
    n_p = len(p)
    left = right = 0
    valid = 0
    map_p = dict(collections.Counter(p))
    map_n = {}
    while right < n:
        char = s[right]
        map_n[char] = map_n.get(char, 0) + 1
        if map_p.get(char, 0) == map_n[char]:
            valid += 1
        while right - left + 1 >= n_p:
            if valid == len(map_p):
                ans.append(left)
            if map_n[s[left]] == map_p.get(s[left], 0):
                valid -= 1
            map_n[s[left]] -= 1
            left += 1
        right += 1
    return ans
#print(findAnagrams("cbaebabacd", "abc"))


# 'abcde' 'bcdea'

# "abcde" "abced"

def rotateString(s: str, goal: str) -> bool:
    first = s
    n_s, n_g = len(s), len(goal)
    if n_s != n_g:
        return False
    while True:
        s = s[1: n_s] + s[0]
        if s == goal:
            return True
        if s == first:
            return False

# print(rotateString('bbbacddceeb','ceebbbbacdd'))


def numSubarrayProductLessThanK(nums: List[int], k: int) -> int:
    n = len(nums)
    ans = left = right = 0
    product = 1
    while right < n:
        product *= nums[right]

        while product >= k and left <= right:
            product //= nums[left]
            left += 1

        if product < k and left <= right:
            ans += 1
            if right - left > 0:
                ans += (right - left)
        right += 1

    return ans

#print(numSubarrayProductLessThanK([57,44,92,28,66,60,37,33,52,38,29,76,8,75,22], 18))


class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = children


def levelOrder(root: 'Node') -> List[List[int]]:
    if not root:
        return None
    q = deque()
    res = []
    q.append(root)
    while q:
        for _ in range(len(q)):
            node = q.popleft()
            res.append(node.val)
            for child in node.children:
                q.append(child)
    return res


def numSubarraysWithSum(nums: List[int], goal: int) -> int:
    n = len(nums)
    res = 0
    if sum(nums) < goal:
        return res
    left1 = left2 = right = 0
    sum1 = sum2 = 0
    while right < n:
        sum1 += nums[right]
        while left1 <= right and sum1 > goal:
            sum1 -= nums[left1]
            left1 += 1
        sum2 += nums[right]
        while left2 <= right and sum2 >= goal:
            sum2 -= nums[left2]
            left2 += 1
        res += left2 - left1
        right += 1
    return res

#print(numSubarraysWithSum( [1,0,1,0,1], goal = 2))


def totalFruit(fruits: List[int]) -> int:
    n = len(fruits)

    ans = 0
    left = right = 0
    map = dict()
    while right < n:

        map[fruits[right]] = map.get(fruits[right], 0) + 1

        while len(map) > 2 and left <= right:
            map[fruits[left]] -= 1
            if map[fruits[left]] == 0:
                del map[fruits[left]]
            left += 1
        if len(map) <= 2:
            ans = max(ans, right - left + 1)
        right += 1
    return ans
# print(totalFruit([0]))


def countNumbersWithUniqueDigits(n: int) -> int:
    res, cur = 10, 9
    if n == 0:
        return 1
    if n == 1:
        return 9
    for i in range(n - 1):
        '''
        排列+DP
        '''
        cur *= 9 - i
        res += cur
    return res


def maxSumTwoNoOverlap(nums: List[int], firstLen: int, secondLen: int) -> int:
    n = len(nums)
    if n < firstLen + secondLen:
        return
    res = 0
    for i, j in [(firstLen, secondLen), (secondLen, firstLen)]:
        sum_1 = sum(nums[:i])
        sum_2 = sum(nums[i:i + j])
        res_1 = sum_1 + sum_2
        index = i
        firstMaxValue = sum_1
        while index < n - j:
            sum_1 += (nums[index] - nums[index - i])
            firstMaxValue = max(firstMaxValue, sum_1)
            sum_2 += nums[index + j] - nums[index]
            res_1 = max(res_1, firstMaxValue + sum_2)
            index += 1
        res = max(res, res_1)
    return res
#print(maxSumTwoNoOverlap([8,20,6,2,20,17,6,3,20,8,12], 5, 4))

# 输入：text = "ababa"
# 输出：3


def maxRepOpt1(text: str) -> int:
    window = [0]*26
    for char in text:
        window[ord(char) - ord('a')] += 1
    left = right = 0
    n = len(text)
    res = 0
    while right < n:
        leng = 0
        i = right
        while i + leng < n and text[i + leng] == text[i]:
            leng += 1
        j = leng + i + 1
        w = 0
        while j + w < n and text[j + w] == text[i]:
            w += 1
        # 只允许跳一个
        res = max(res, min(w + leng + 1, window[ord(text[i]) - ord('a')]))
        right += leng
    return res


def numberOfLines(widths: List[int], s: str) -> List[int]:

    raw = 1
    left = 100
    for char in s:
        index = ord(char) - ord('a')
        left -= widths[index]
        if left < 0:
            raw += 1
            left = 100 - widths[index]
    return (raw, 100 - left)

# print(numberOfLines(
# [4,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10]
# ,"bbbcccdddaaa"))


def longestOnes(nums: List[int], k: int) -> int:
    zeroCount = left = right = 0
    res, n = 0, len(nums)
    while right < n:
        if nums[right] == 0:
            zeroCount += 1

        while zeroCount > k and left <= right:
            if nums[left] == 0:
                zeroCount -= 1
            left += 1

        if zeroCount <= k:
            res = max(res, right - left + 1)

        right += 1
    return res
# print(longestOnes([0,0,0,1], 4))


class RandomizedSet:

    def __init__(self):
        self.map = dict()
        self.numbers = list()

    def insert(self, val: int) -> bool:
        if val in self.map:
            return False
        self.numbers.append(val)
        self.map[val] = len(self.numbers) - 1
        return True

    def remove(self, val: int) -> bool:
        if val not in self.map:
            return False
        id = self.map[val]
        if id != len(self.numbers) - 1:
            self.numbers[id] = self.numbers[-1]
            self.map[self.numbers[-1]] = id
        self.numbers.pop()
        del self.map[val]
        return True

    def getRandom(self) -> int:
        return choice(self.numbers)

# dp+滑动窗口


def new21Game(n: int, k: int, maxPts: int) -> float:

    dp = [0.0] * (k + maxPts)

    s = min(n - k + 1, maxPts)

    for index in range(k, min(n + 1, k + maxPts)):
        dp[index] = 1.0

    for i in range(k - 1, -1, -1):
        dp[i] = s / maxPts
        s += dp[i] - dp[i + maxPts]

    return dp[0]

#print(new21Game(0, 0, 1))


# 输入：customers = [1,0,1,2,1,1,7,5], grumpy = [0,1,0,1,0,1,0,1], minutes = 3
# 输出：16
# 解释：书店老板在最后 3 分钟保持冷静。
# 感到满意的最大客户数量 = 1 + 1 + 1 + 1 + 7 + 5 = 16.


def maxSatisfied(customers: List[int], grumpy: List[int], minutes: int) -> int:
    n = len(customers)
    total = sum(c for c, g in zip(customers, grumpy) if g == 0)
    maxIncrease = increase = sum(
        c * g for c, g in zip(customers[:minutes], grumpy[:minutes]))
    for i in range(minutes, n):
        increase += customers[i] * grumpy[i] - \
            customers[i - minutes] * grumpy[i - minutes]
        maxIncrease = max(maxIncrease, increase)
    return total + maxIncrease


#print(maxSatisfied([1,0,1,2,1,1,7,5], [0,1,0,1,0,1,0,1], 3))

def numberOfSubarrays(nums: List[int], k: int) -> int:
    res = count = 0
    left, right, n = 0, 0, len(nums)
    temp = 0
    while right < n:
        if nums[right] & 1:
            count += 1
        if count > k and left <= right:
            if nums[left] & 1:
                count -= 1
            left += 1
        x = left
        while count == k and left <= right:
            if nums[left] & 1:
                break
            left += 1

        if count == k:
            if nums[right] & 1:
                temp = left - x + 1
            res += temp

        right += 1
    return res

#print(numberOfSubarrays([2,2,2,1,2,2,1,2,2,2], 2))


def equalSubstring(s: str, t: str, maxCost: int) -> int:
    n = len(s)
    left = res = cost = 0
    for right in range(n):
        cost += abs(ord(s[right]) - ord(t[right]))
        while cost > maxCost and left <= right:
            cost -= abs(ord(s[left]) - ord(t[left]))
            left += 1
        if cost <= maxCost:
            res = max(res, right - left + 1)
    return res

#print(equalSubstring("abcd", "cdef", 3))


def maxFreq(s: str, maxLetters: int, minSize: int, maxSize: int) -> int:

    ans = 0
    hashmap = dict()

    n = len(s)
    for left in range(n - minSize + 1):

        subString = s[left: left + minSize - 1]
        c = Counter(subString)

        right = left + minSize - 1
        while right < n and minSize <= right - left + 1 <= maxSize:
            key = s[right]
            subString += key
            c.update(key)
            if len(c) <= maxLetters:

                hashmap[subString] = hashmap.get(subString, 0) + 1
            right += 1
    if len(hashmap) > 0:
        ans = hashmap[max(hashmap, key=hashmap.get)]
    return ans


#print(maxFreq("badeeadeaecabacdbeba", 5, 7, 13))


def mostCommonWord(paragraph: str, banned: List[str]) -> str:
    sentece = Counter([word.lower() for word in " ".join(re.findall(
        r'\b\w+\b', paragraph)).split() if word.lower() not in set(banned)])
    return max(sentece, key=sentece.get)


# print(mostCommonWord("Bob. hIt, baLl", ["bob", "hit"]))


def balancedString(s: str) -> int:
    n = len(s)
    k = n // 4
    hashmap = Counter(s)
    left = 0
    res = n
    for right in range(n):
        c = s[right]
        hashmap[c] -= 1

        while left < n and hashmap['Q'] <= k and hashmap['W'] <= k and hashmap['E'] <= k and hashmap['R'] <= k:
            c_left = s[left]
            hashmap[c_left] += 1
            res = min(res, right - left + 1)
            left += 1
    return res

# print(balancedString("QWER"))


def numOfSubarrays(arr: List[int], k: int, threshold: int) -> int:
    n = len(arr)
    avg = 0
    left = 0
    res = 0
    compare = k * threshold
    for right in range(n):
        avg += arr[right]
        if right - left + 1 > k:
            avg -= arr[left]
            left += 1
        if right - left + 1 == k:
            if avg >= compare:
                res += 1
    return res

# print(numOfSubarrays([2,2,2,2,5,5,5,8], 3, 4))


def numberOfSubstrings(s: str) -> int:
    count = [0] * 3
    n = len(s)
    left = res = valid = 0
    for right in range(n):
        char = s[right]
        index = ord(char) - ord('a')
        if count[index] == 0:

            valid += 1
        count[index] += 1

        while valid == 3:

            res += n - right
            char_l = s[left]
            index_l = ord(s[left]) - ord('a')
            if count[index_l] == 1:
                valid -= 1
            count[index_l] -= 1
            left += 1

    return res


# print(numberOfSubstrings("abc"))


def minOperations(nums: List[int], x: int) -> int:
    res = -1
    num = sum(nums) - x
    left, n = 0, len(nums)
    cnt = 0
    for right in range(n):
        cnt += nums[right]
        while left <= right and cnt > num:
            cnt -= nums[left]
            left += 1
        if cnt == num:
            res = max(res, right - left + 1)
    return n - res if res > -1 else -1

# print(minOperations([3,2,20,1,1,3], 10))


def maxScore(cardPoints: List[int], k: int) -> int:
    n = len(cardPoints)
    right = n - 1
    left = n - 1 - k + 1
    res = sum(cardPoints[left:])
    count = res
    right += 1
    while right < n + k:

        count += cardPoints[right % n]
        right += 1

        count -= cardPoints[left % n]
        left += 1

        res = max(res, count)

    return res

# print(maxScore([11,49,100,20,86,29,72], 4))


def lengthLongestPath(input: str) -> int:
    stack = []
    res, i, n = 0, 0, len(input)

    while i < n:
        depth = 1  # 初始为1
        while i < n and input[i] == '\t':
            depth += 1
            i += 1
        length, isFlie = 0, False
        while i < n and input[i] != '\n':
            if input[i] == '.':
                isFlie = True
            length += 1
            i += 1
        i += 1

        while len(stack) >= depth:
            stack.pop()
        if stack:
            length += stack[-1] + 1
        if isFlie:
            res = max(res, length)
        else:
            stack.append(length)

    return res


def maxRotateFunction(nums: List[int]) -> int:
    s, n = sum(nums), len(nums)
    dp = sum([x*y for x, y in enumerate(nums)])
    res = dp
    for i in range(1, n):
        dp += s - n * nums[n - i]
        res = max(res, dp)
    return res

# print(maxRotateFunction([4]))


def binaryGap(n: int) -> int:
    div = n
    res = 0
    prev = -1
    cnt = 0
    while div:
        div, mod = divmod(div, 2)
        if mod == 1:
            if prev == -1:
                prev = cnt
            res = max(res, cnt - prev)
            prev = cnt
        cnt += 1
    return res

# print(binaryGap(22))


def reachingPoints(sx: int, sy: int, tx: int, ty: int) -> bool:
    while sx < tx and ty > sy:
        if tx > ty:
            tx %= ty
        else:
            ty %= tx
    if tx == sx and ty == sy:
        return True
    elif tx == sx:
        return ty > sy and (ty - sy) % tx == 0
    elif ty == sy:
        return tx > sx and (tx - sx) % ty == 0
    else:
        return False

# print(reachingPoints(1, 1, 3, 5))


def exist(board: List[List[str]], word: str) -> bool:

    m, n = len(board), len(board[0])
    flag = False

    def dfs(vis, i, x, y):
        nonlocal flag
        if i < len(word) and not vis[x][y] and board[x][y] == word[i]:
            if i == len(word) - 1:
                flag = True
                return
            vis[x][y] = True
            for nx, ny in [(x + 1, y), (x - 1, y), (x, y - 1), (x, y + 1)]:
                if 0 <= nx < m and 0 <= ny < n:
                    dfs(vis, i + 1, nx, ny)
            vis[x][y] = False
        else:
            return
    for x in range(m):
        for y in range(n):
            vis = [[False] * n for _ in range(m)]
            if board[x][y] == word[0]:
                dfs(vis, 0, x, y)
    return flag


#print(exist([["C", "A", "A"], ["A", "A", "A"], ["B", "C", "D"]], "AAB"))


def hIndex(citations: List[int]) -> int:
    n = len(citations)
    i, j = 0, n - 1

    while i <= j:
        mid = i + (j - i) // 2
        if n - mid > citations[mid]:

            i = mid + 1
        else:
            j = mid - 1

    return n - i

# print(hIndex([0,0,0]))


def minEatingSpeed(piles: List[int], h: int) -> int:
    maxNumber = max(piles)
    minNumber = 1

    while minNumber < maxNumber:
        mid = minNumber + (maxNumber - minNumber) // 2
        waster = 0
        for p in piles:
            div, mod = divmod(p, mid)
            if mod:
                waster += 1
            waster += div
        if waster > h:
            minNumber = mid + 1
        else:
            maxNumber = mid

    return maxNumber


# print(minEatingSpeed([5], 4))


def shipWithinDays(weights: List[int], days: int) -> int:
    minValue = max(weights)
    maxValue = sum(weights)
    n = len(weights)
    while minValue < maxValue:
        midValue = minValue + (maxValue - minValue) // 2
        d = i = count = 0
        while i < n:
            count += weights[i]
            if count > midValue:
                d += 1
                count = weights[i]
            if i != n - 1 and count == midValue:
                d += 1
                count = 0
            if i == n - 1 and count <= midValue:
                d += 1
            i += 1
        if d > days:
            minValue = midValue + 1
        else:
            maxValue = midValue
    return maxValue


# print(shipWithinDays([3,3,3,3,3,3], 2))


def splitArray(nums: List[int], m: int) -> int:
    n = len(nums)

    dp = []
    if m == 1:
        return sum(nums)
    if n < m:
        return 0
    else:
        dp = nums[:m]
    for j in range(m, n):
        v = dp[: m - 1] + [dp[m - 1] + nums[j]]
        i = 0
        s1 = v
        while i < m - 1:
            s = dp[:i] + [dp[i] + dp[i+1]] + dp[i+2:] + [nums[j]]
            if max(s) < max(s1):
                s1 = s
            i += 1
        dp = s1
    return max(dp)

# print(splitArray([3,3,3,3,3,3],3))


def minDays(bloomDay: List[int], m: int, k: int) -> int:
    minValue = min(bloomDay)
    maxValue = max(bloomDay)
    n = len(bloomDay)
    if m * k > n:
        return -1

    def check(day, k):
        i = 0
        cnt = 0
        while i < n - k + 1:
            if bloomDay[i] <= day:
                j = 0
                flag = False
                for x in range(0, k):
                    if bloomDay[i + x] > day:
                        j = i + x
                        flag = True
                        break

                if not flag:
                    cnt += 1
                    i = i + k
                else:
                    i = j
            else:
                i += 1
        return cnt

    while minValue < maxValue:
        midValue = minValue + (maxValue - minValue) // 2

        cnt = check(midValue, k)

        if cnt >= m:
            maxValue = midValue
        else:
            minValue = midValue + 1

    return minValue


# print(minDays([1,10,2,9,3,8,4,7,5,6], 4, 2))


def maxDistance(position: List[int], m: int) -> int:

    def check(x: int) -> bool:
        i = 1
        cnt = 0
        prev = position[0]
        while i < len(position):
            if position[i] >= prev:
                prev = position[i]
                cnt += 1
            i += 1
        return cnt

    position.sort()
    left, right, ans = 1, position[-1] - position[0], -1
    while left <= right:
        mid = (left + right) // 2
        if check(mid) >= m - 1:
            ans = mid
            left = mid + 1
        else:
            right = mid - 1

    return ans


#print(maxDistance( [5,4,3,2,1,1000000000], 2))


class Node:
    def __init__(self, val, isLeaf, topLeft, topRight, bottomLeft, bottomRight):
        self.val = val
        self.isLeaf = isLeaf
        self.topLeft = topLeft
        self.topRight = topRight
        self.bottomLeft = bottomLeft
        self.bottomRight = bottomRight


def construct(grid: List[List[int]]) -> 'Node':

    def check(left, right, top, bot):
        zero = 0
        one = 1
        for i in range(left, right + 1):
            for j in range(top, bot + 1):
                if grid[i][j] != zero:
                    zero = 1
        for i in range(left, right + 1):
            for j in range(top, bot + 1):
                if grid[i][j] != one:
                    one = 0
        if zero == 0 and one == 0:
            return 0
        if one == 1 and zero == 1:
            return 1
        return 2

    def buildTree(left, right, top, bot):
        if left == right and top == bot:
            return Node(grid[left][top], 1, None, None, None, None)
        if check(left, right, top, bot) == 0:
            print(0, left, right, top, bot)
            return Node(0, 1, None, None, None, None)
        if check(left, right, top, bot) == 1:
            print(1, left, right, top, bot)
            return Node(1, 1, None, None, None, None)
        node = Node(1, 0, None, None, None, None)
        node.topLeft = buildTree(
            left, (left + right) // 2, top, (top + bot) // 2)
        node.topRight = buildTree(
            (left + right) // 2 + 1, right, top, (top + bot) // 2)
        node.bottomLeft = buildTree(
            left, (left + right) // 2, (top + bot) // 2 + 1, bot)
        node.bottomRight = buildTree(
            (left + right) // 2 + 1, right, (top + bot) // 2 + 1, bot)
        return node

    print(check(4, 7, 0, 3))
    # return buildTree(0, len(grid) - 1, 0, len(grid[0]) - 1)
    return None


# construct([[1,1,1,1,0,0,0,0],[1,1,1,1,0,0,0,0],[1,1,1,1,1,1,1,1],[1,1,1,1,1,1,1,1],[1,1,1,1,0,0,0,0],[1,1,1,1,0,0,0,0],[1,1,1,1,0,0,0,0],[1,1,1,1,0,0,0,0]])

def minTime(time: List[int], m: int) -> int:
    minValue = 0
    maxValue = sum(time) - time[-1]
    n = len(time)

    def check(midValue):
        cnt = 0
        max_value = time[0]
        i = k = 1
        while i < n:
            if cnt + min(time[i], max_value) > midValue:
                cnt = 0
                max_value = time[i]
                k += 1
            else:
                cnt += min(max_value, time[i])
                max_value = max(time[i], max_value)
            i += 1
        return k

    while minValue < maxValue:
        midValue = minValue + (maxValue - minValue) // 2
        if check(midValue) <= m:
            maxValue = midValue
        else:
            minValue = midValue + 1

    return minValue

# print(minTime([1, 2, 3, 3], 2))


def my_generator():
    for i in range(5):
        if i == 2:
            return '我被迫中断了'
        else:
            yield i


def wrap_my_generator(generator):  # 定义一个包装“生成器”的生成器，它的本质还是生成器
    # 自动触发StopIteration异常，并且将return的返回值赋值给yield from表达式的结果，即result
    result = yield from generator
    print("result", result)


def main(generator):
    for j in generator:
        print(j)


g = my_generator()
wrap_g = wrap_my_generator(g)
# main(wrap_g)


def numTeams(rating: List[int]) -> int:

    def lowbit(x):          # lowbit函数：二进制最低位1所表示的数字
        return x & (-x)

    def add(i, delta):      # 单点更新：执行+delta
        while i < len(tree):
            tree[i] += delta
            i += lowbit(i)

    def query(i):           # 前缀和查询
        presum = 0
        while i > 0:
            presum += tree[i]
            i -= lowbit(i)
        return presum

    '''主程序'''
    # 离散化：绝对数值转秩次rank
    uniques = sorted(rating)    # rating没有重复值
    rank_map = {v: i+1 for i, v in enumerate(uniques)}  # 【rank从1开始】

    # 构建树状数组
    tree = [0] * (len(rank_map) + 1)    # 树状数组下标从1开始

    # 枚举中间点
    ans = 0
    n = len(rating)
    add(rank_map[rating[0]], 1)     # 先将第一个元素入列
    for i in range(1, n-1):         # 从第二个元素开始遍历，直至倒数第二个元素
        rank = rank_map[rating[i]]  # 当前元素的排名/秩次

        small1 = query(rank-1)      # 查询前序元素中排名<rank的元素数目
        large1 = i - small1         # small1 + large1 = i

        # 共有rank-1个元素排名<rank：small1 + small2 = rank-1
        small2 = (rank-1) - small1
        large2 = n-1 - i - small2   # small2 + large2 = n-1-i

        add(rank, 1)                # 当前元素入列

        ans += small1*large2 + large1*small2

    return ans


# print(numTeams([2,5,3,4,1]))

def minMutation(start: str, end: str, bank: List[str]) -> int:
    if end not in bank:
        return -1
    if start == end:
        return 0
    q = deque()
    q.append((start, 0))
    b = set(bank)
    vis = set()
    while q:
        cur, step = q.popleft()
        for i in range(8):
            c = cur[i]
            for char in ['A', 'C', 'G', 'T']:
                if c != char:
                    s = cur[:i] + char + cur[i + 1:]
                    if s in b and s not in vis:
                        if s == end:
                            step += 1
                            return step
                        q.append((s, step + 1))
                        vis.add(s)

    return -1


# print(minMutation("AACCGGTT", "AAACGGTA", ["AACCGGTA","AACCGCTA","AAACGGTA"]))


def diStringMatch(s: str) -> List[int]:
    n = len(s)
    lo, hi = 0, n
    res = []
    for i in range(n):
        if s[i] == 'I':
            res.append(lo)
            lo += 1
        else:
            res.append(hi)
            hi -= 1
    res.append(hi)
    return res

# print(diStringMatch("III"))


def findDisappearedNumbers(nums: List[int]) -> List[int]:
    # 原地hash或者抽屉原理
    res = []
    n = len(nums)
    for i in range(n):
        while nums[i] != nums[nums[i] - 1]:
            tmp = nums[i]
            nums[i] = nums[tmp - 1]
            nums[tmp - 1] = tmp
    for j in range(n):
        if nums[j] != j + 1:
            res.append(j + 1)
    return res

# print(findDisappearedNumbers([1, 1]))


def findNumberOfLIS(nums: List[int]) -> int:
    n = len(nums)
    dp = [[1, 1] for _ in range(n)]
    for i in range(1, n):
        for j in range(i):
            if nums[i] > nums[j]:
                pre = dp[i][0]
                dp[i][0] = max(dp[i][0], dp[j][0] + 1)
                if dp[i][0] > pre:
                    dp[i][1] = dp[j][1]
                if dp[i][0] == pre and pre > 1 and dp[j][0] == pre - 1:
                    dp[i][1] += dp[j][1]

    print(dp)
    m = 0
    res = 0
    for d in dp:
        if d[0] > m:
            m = d[0]
            res = d[1]
        elif d[0] == m:
            res += d[1]
    return res

# print(findNumberOfLIS([1,2,3,1,2,3,1,2,3]))


def isInterleave(s1: str, s2: str, s3: str) -> bool:
    m, n = len(s1), len(s2)
    if m + n != len(s3):
        return False
    dp = [False for i in range(n + 1)]
    dp[0] = True
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 and j > 0:
                if dp[j - 1] and s2[j - 1] == s3[j - 1]:
                    dp[j] = True
            if i > 0 and j == 0:
                if dp[0] and s1[i - 1] == s3[i - 1]:
                    dp[0] = True
                else:
                    dp[0] = False
            if i > 0 and j > 0:
                if dp[j] and s1[i - 1] == s3[i + j - 1]:
                    dp[j] = True
                elif dp[j - 1] and s2[j - 1] == s3[i + j - 1]:
                    dp[j] = True
                else:
                    dp[j] = False
    return dp[n]


def networkDelayTime(times: List[List[int]], n: int, k: int) -> int:
    g = [[float('inf')] * n for _ in range(n)]
    for x, y, time in times:
        g[x - 1][y - 1] = time

    dist = [float('inf')] * n
    dist[k - 1] = 0
    used = [False] * n
    for _ in range(n):
        x = -1
        for y, u in enumerate(used):
            if not u and (x == -1 or dist[y] < dist[x]):
                x = y
        used[x] = True
        for y, time in enumerate(g[x]):
            dist[y] = min(dist[y], dist[x] + time)

    ans = max(dist)
    return ans if ans < float('inf') else -1


def isAlienSorted(words: List[str], order: str) -> bool:
    hashmap = {}
    for i, char in enumerate(order):
        if char not in hashmap:
            hashmap[char] = i
    for i in range(len(words) - 1):
        m = n = 0
        while m < len(words[i]) and n < len(words[i + 1]):
            if m == n:
                if hashmap[words[i][m]] > hashmap[words[i + 1][n]]:
                    return False
                elif hashmap[words[i][m]] < hashmap[words[i + 1][n]]:
                    break
                else:
                    m += 1
                    n += 1
            else:
                break
        if m < len(words[i]) and n >= len(words[i + 1]):
            return False
    return True

# print(isAlienSorted(["fxasxpc","dfbdrifhp","nwzgs","cmwqriv","ebulyfyve","miracx","sxckdwzv","dtijzluhts","wwbmnge","qmjwymmyox"]
# ,"zkgwaverfimqxbnctdplsjyohu"))


def maxEnvelopes(envelopes: List[List[int]]) -> int:
    envelopes = sorted(envelopes, key=lambda x: (x[0], -x[1]))
    n = len(envelopes)
    heights = [envelop[1] for envelop in envelopes]
    dp = [heights[0]]
    for i in range(1, n):
        if heights[i] > dp[-1]:
            dp.append(heights[i])
        else:
            left = bisect_left(dp, heights[i])
            dp[left] = heights[i]

    return len(dp)

# print(maxEnvelopes([[5,4],[6,4],[6,7],[2,3]]))


def findKthNumber(m: int, n: int, k: int) -> int:
    def check(mid: int) -> bool:
        count = mid // n * n
        for i in range((mid // n)+1, m + 1):
            count += (mid // i)
        return count >= k
    lo, hi = 0, m * n
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if check(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo

# print(findKthNumber(2, 3, 6))


def smallestDistancePair(nums: List[int], k: int) -> int:
    nums.sort()
    n = len(nums)

    def check(mid: int) -> bool:
        left, right = 0, 1
        count = 0
        while right < n and left <= right:
            if nums[right] - nums[left] <= mid:
                count += (right - left)
            else:
                left += 1
                continue
            right += 1
        return count >= k
    lo, hi = 0, nums[-1] - nums[0]
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if check(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo

# print(smallestDistancePair([1, 3, 1], 1))


def isMatch(s: str, p: str) -> bool:
    m, n = len(p), len(s)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True
    for i in range(1, m + 1):
        for j in range(n + 1):
            if p[i - 1] == '*':
                if j == 0:
                    dp[i][j] = dp[i - 1][j]
                else:
                    dp[i][j] = dp[i - 1][j - 1] or dp[i][j - 1] or dp[i - 1][j]
            elif p[i - 1] == '?':
                if j > 0:
                    dp[i][j] = dp[i - 1][j - 1]
            else:
                if j > 0 and p[i - 1] == s[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
    return dp[m][n]

# print(isMatch("acdcb", "a*******b"))


def minMoves2(nums: List[int]) -> int:
    nums.sort()
    n = len(nums)
    res = 0
    if n & 1 == 1:
        mid1 = mid2 = n // 2
        res1 = res2 = 0
        pre1 = pre2 = 0
        while mid1 >= 1 and mid2 <= n - 2:
            pre1 += nums[mid1] - nums[mid1 - 1]
            pre2 += nums[mid2 + 1] - nums[mid2]
            res1 += pre1
            res2 += pre2
            mid1 -= 1
            mid2 += 1
        res = res1 + res2
    else:
        mid1, mid2 = n // 2 - 1, n // 2
        diff = nums[mid2] - nums[mid1]
        res1 = res2 = 0
        pre1 = pre2 = 0
        while mid1 >= 1 and mid2 <= n - 2:
            pre1 += nums[mid1] - nums[mid1 - 1]
            pre2 += nums[mid2 + 1] - nums[mid2]
            res1 += pre1
            res2 += pre2
            mid1 -= 1
            mid2 += 1
        res = res1 + res2 + diff * (n // 2)
    return res

# print(minMoves2([1,2,3]))


def minMoves(nums: List[int]) -> int:
    n = len(nums)
    nums = set(nums)
    m = max(nums)
    return sum(m - num for num in nums) + n - len(nums)

# print(minMoves([5,6,8,8,5]))


def findRightInterval(intervals: List[List[int]]) -> List[int]:
    interlist = sorted(intervals, key=lambda x: x[0])
    hashmap = {}
    hashmap_1 = {}
    for i, x in enumerate(interlist):
        hashmap[x[0]] = i
    for i, x in enumerate(intervals):
        hashmap_1[x[0]] = i
    res = []

    def binarySeach(lo: int, hi: int, v: int) -> int:
        while lo < hi:
            mid = lo + (hi - lo) // 2
            if interlist[mid][0] >= v:
                hi = mid
            else:
                lo = mid + 1
        return lo
    n = len(intervals)
    for x in intervals:
        idx = hashmap[x[0]]
        if idx == n - 1:
            res.append(-1)
            continue
        lo = binarySeach(idx, n - 1, x[1])
        if interlist[lo][0] >= x[1]:
            res.append(hashmap_1[interlist[lo][0]])
        else:
            res.append(-1)
    return res


# print(findRightInterval([[1,1],[3,4]]))

def minFallingPathSum(matrix: List[List[int]]) -> int:
    n = len(matrix)
    dp = matrix[0]
    for i in range(1, n):
        new = [0] * n
        for j in range(n):
            val = dp[j]
            if 0 <= j - 1:
                val = min(val, dp[j - 1])
            if j + 1 < n:
                val = min(val, dp[j + 1])
            new[j] = val + matrix[i][j]
        dp = new
    res = min(dp)
    return res
# print(minFallingPathSum([[17,82],[1,-44]]))


def cutOffTree(forest: List[List[int]]) -> int:
    trees = sorted([(forest[i][j], i, j) for i, row in enumerate(forest)
                   for j, col in enumerate(row) if forest[i][j] > 1])
    m, n = len(forest), len(forest[0])

    def bfs(sx: int, sy: int, ix: int, iy: int) -> int:
        q = deque([(0, sx, sy)])
        vis = {(sx, sy)}
        while q:
            h, x, y = q.popleft()
            if x == ix and y == iy:
                return h
            for i, j in [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]:
                if 0 <= i < m and 0 <= j < n and (i, j) not in vis and forest[i][j] > 0:
                    q.append((h+1, i, j))
                    vis.add((i, j))
        return -1

    ans = prei = prej = 0
    for h, x, y in trees:
        res = bfs(prei, prej, x, y)
        if res == -1:
            return -1
        else:
            ans += res
        prei, prej = x, y
    return ans

# print(cutOffTree([[4,2,3],[0,0,1],[7,6,5]]))


def diffWaysToCompute(expression: str) -> List[int]:
    hashmap = dict()

    def childWaysToCompute(expression: str) -> List[int]:
        res = []
        if expression in hashmap:
            res = hashmap[expression]
            return res
        for i, char in enumerate(expression):
            if char in ['-', '+', '*']:
                l = childWaysToCompute(expression[:i])
                r = childWaysToCompute(expression[i+1:])
                for i in l:
                    for j in r:
                        if char == '-':
                            res.append(i - j)
                        if char == '+':
                            res.append(i + j)
                        if char == '*':
                            res.append(i * j)
        if not res and expression:
            res.append(int(expression))
        hashmap[expression] = res
        return res
    return childWaysToCompute(expression)
# print(diffWaysToCompute("2*3-4*5"))


def longestCommonSubsequence(text1: str, text2: str) -> int:
    m, n = len(text1), len(text2)
    dp = [0 for _ in range(n + 1)]
    for i in range(1, m + 1):
        tmp = dp.copy()
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                tmp[j] = max(dp[j-1] + 1, dp[j], tmp[j-1])
            else:
                tmp[j] = max(dp[j], tmp[j-1])
        dp = tmp
    return dp[n]

# print(longestCommonSubsequence("abcba", "abcbcba"))


def findSubstringInWraproundString(p: str) -> int:
    dp = defaultdict(int)
    k = 1
    dp[p[0]] = 1
    for i in range(1, len(p)):
        if ord(p[i]) - ord(p[i-1]) == 1 or ord(p[i-1]) - ord(p[i]) == 25:
            k += 1
        else:
            k = 1
        dp[p[i]] = max(dp[p[i]], k)
    return sum(dp.values())

# print(findSubstringInWraproundString("zab"))


def maximalSquare(matrix: List[List[str]]) -> int:
    m, n = len(matrix), len(matrix[0])
    dp = [[0 for _ in range(n)] for _ in range(m)]
    res = 0
    if matrix[0][0] == '1':
        dp[0][0] = 1
    for i in range(m):
        for j in range(n):
            if i == 0 and j > 0:
                if matrix[i][j] == '1':
                    dp[i][j] = 1
                    res = max(res, dp[i][j])
                continue
            if j == 0 and i > 0:
                if matrix[i][j] == '1':
                    dp[i][j] = 1
                    res = max(res, dp[i][j])
                continue

            if i > 0 and j > 0 and matrix[i][j] == '1':
                if matrix[i-1][j-1] == '1' and matrix[i-1][j] == '1' and matrix[i][j-1] == '1':
                    if dp[i-1][j-1] == dp[i-1][j] == dp[i][j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = min(dp[i-1][j-1], dp[i-1]
                                       [j], dp[i][j-1]) + 1
                else:
                    dp[i][j] = 1
            res = max(res, dp[i][j])
    return res * res
# print(maximalSquare([["1", "0"]]))


def minDistance(word1: str, word2: str) -> int:
    if word1 == word2:
        return 0
    m, n = len(word1), len(word2)
    dp = [0 for _ in range(n + 1)]
    for i in range(1, m + 1):
        tmp = dp.copy()
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                tmp[j] = max(dp[j-1] + 1, dp[j], tmp[j-1])
            else:
                tmp[j] = max(dp[j], tmp[j-1])
        dp = tmp
    return m + n - 2 * dp[n]

# print(minDistance("leetcode", "etco"))


def minInsertions(s: str) -> int:
    n = len(s)
    dp = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        dp[i][i] = 1
    for i in range(n-1, -1, -1):
        for j in range(i+1, n):
            if s[i] == s[j]:
                dp[i][j] = max(dp[i+1][j-1] + 2, dp[i+1][j], dp[i][j-1])
            else:
                dp[i][j] = max(dp[i+1][j], dp[i][j-1])
    return n - dp[0][n-1]
# print(minInsertions("mbadm"))


def findClosest(words: List[str], word1: str, word2: str) -> int:
    hashmap = defaultdict(list)
    for i, word in enumerate(words):
        hashmap[word].append(i)
    l1, l2 = hashmap[word1], hashmap[word2]
    m, n = len(l1), len(l2)
    i = j = 0
    res = float("inf")
    while i < m and j < n:
        if l1[i] == l2[j]:
            return 0
        if l1[i] > l2[j]:
            res = min(res, l1[i] - l2[j])
            j += 1
        elif l1[i] < l2[j]:
            res = min(res, l2[j] - l1[i])
            i += 1
    return res

# print(findClosest(["I","am","a","student","from","a","university","in","a","city"], "a", "student"))


def optimalDivision(nums: List[int]):
    n = len(nums)
    hashmap = {}

    def findDivision(i, j):
        res = 0
        res1 = float("inf")
        if j == i:
            res = res1 = nums[i]
            hashmap[(i, j)] = (res, res1, str(nums[i]), str(nums[i]))
            return (res, res1, str(nums[i]), str(nums[i]))
        if j - i == 1:
            res = res1 = nums[i] / nums[j]
            hashmap[(i, j)] = (res, res1, str(nums[i]) + '/' +
                               str(nums[j]), str(nums[i]) + '/' + str(nums[j]))
            return (res, res1, str(nums[i]) + '/' + str(nums[j]), str(nums[i]) + '/' + str(nums[j]))
        if (i, j) in hashmap:
            return hashmap[(i, j)]
        s = ""
        s1 = ""
        for k in range(i+1, j+1):
            l = findDivision(i, k-1)
            r = findDivision(k, j)
            ms = l[0] / r[1]
            mi = l[1] / r[0]
            if ms > res:
                if j - k >= 1:
                    s = l[2] + '/' + '(' + r[3] + ')'
                else:
                    s = l[2] + '/' + r[3]
                res = ms
            if mi < res:
                if j - k >= 1:
                    s1 = l[3] + '/' + '(' + r[2] + ')'
                else:
                    s1 = l[3] + '/' + r[2]
                res1 = mi
        if (i, j) not in hashmap:
            hashmap[(i, j)] = (res, res1, s, s1)
        return (res, res1, s, s1)
    findDivision(0, n-1)
    return hashmap[(0, n-1)][2]
    # '1000/(1000)/(1000/100/(1000/100))'

# print(optimalDivision([1000,100,10,2]))


def alienOrder(words: List[str]) -> str:
    res = ""
    bot = ord('a')
    degrees = [0] * 26
    n = len(words)
    graph = defaultdict(list)
    for word in words:
        for char in word:
            if char not in graph:
                graph[char]
    i, flag = 0, True
    while i < n - 1 and flag:
        word1, word2 = words[i], words[i + 1]
        l1, l2 = len(word1), len(word2)
        l = min(l1, l2)
        index = 0
        while index < l:
            if word1[index] != word2[index]:
                graph[word1[index]].append(word2[index])
                degrees[ord(word2[index]) - ord('a')] += 1
                break
            index += 1
        if index == l and l1 > l2:
            flag = False
        i += 1
    if not flag:
        return ""

    q = deque()
    for i in range(26):
        char = chr(i + bot)
        if degrees[i] == 0 and char in graph:
            q.append(char)

    while q:
        r = q.popleft()
        res += r
        for v in graph[r]:
            index = ord(v) - bot
            degrees[index] -= 1
            if degrees[index] == 0:
                q.append(v)
    if len(res) != len(graph):
        res = ""
    return res

# "wertf"
# print(alienOrder(["z","x","z"]))


def canPartition(nums: List[int]) -> bool:
    m = len(nums)
    s = sum(nums)
    if s & 1:
        return False
    n = s // 2
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = True
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if j - nums[i - 1] < 0:
                dp[i][j] = dp[i - 1][j]
            else:
                dp[i][j] = dp[i - 1][j] or dp[i - 1][j - nums[i - 1]]
    return dp[m][n]

# print(canPartition([1,5,11]))


def makesquare(matchsticks: List[int]) -> bool:

    s = sum(matchsticks)
    if s & 1:
        return False
    length = s // 4
    n = len(matchsticks)
    dp = [-1] * (1 << n)
    dp[0] = 0

    for i in range(1 << n):
        for j in range(n):
            if i & (1 << j):
                j1 = i & ~(1 << j)
                if dp[j1] >= 0 and dp[j1] + matchsticks[j] <= length:
                    dp[i] = (dp[j1] + matchsticks[j]) % length
                    break

    return dp[-1] == 0
# print(makesquare([1,1,2,2,2]))


def maxProduct(s: str) -> int:
    n = len(s)
    dp = [0 for _ in range(1 << n)]

    def check(s: str) -> bool:
        n = len(s)
        if n <= 0:
            return False
        l, r = 0, n - 1
        while l < r:
            if s[l] != s[r]:
                return False
            else:
                l += 1
                r -= 1
        return True

    for i in range(n):
        dp[1 << i] = 1

    for idx in range(1, 1 << n):
        pos = 0
        r = 0
        s1 = ""
        for i in range(n):
            if idx & (1 << i):
                pos = i
                s1 += s[i]
        if check(s1):
            r = max(r, len(s1))
        r = max(r, dp[idx & ~(1 << pos)])
        dp[idx] = r

    res = 0
    for i in range(1, 1 << n):
        split = i >> 1
        j = (i - 1) & i
        while j > split:
            res = max(res, dp[j] * dp[i ^ j])
            j = (j - 1) & i
    return res

# print(maxProduct("leetcodecom"))


def calculateMinimumHP(dungeon: List[List[int]]) -> int:
    m, n = len(dungeon), len(dungeon[0])
    dp = [[float("inf")] * n for _ in range(m)]
    if dungeon[m-1][n-1] >= 0:
        dp[m-1][n-1] = 1
    else:
        dp[m-1][n-1] = -dungeon[m-1][n-1] + 1

    for i in range(m-1, -1, -1):
        for j in range(n-1, -1, -1):
            if i == m - 1 and j == n - 1:
                continue
            r = float("inf")
            if i + 1 <= m - 1:
                r = min(r, dp[i+1][j])
            if j + 1 <= n - 1:
                r = min(r, dp[i][j+1])
            dp[i][j] = r - dungeon[i][j] if r - dungeon[i][j] > 0 else 1

    return dp[0][0]

# print(calculateMinimumHP([[-2,-3,3],[-5,-10,1],[10,30,-5]]))


def minEatingSpeed(piles: List[int], h: int) -> int:
    return bisect_left(range(max(piles)), -h, 1, key=lambda k: -sum((pile + k - 1) // k for pile in piles))


def findCheapestPrice(n: int, flights: List[List[int]], src: int, dst: int, k: int) -> int:
    dp = [float("inf")] * n
    dp[src] = 0
    res = float("inf")
    for _ in range(1, k + 2):
        d = dp.copy()
        for f, t, p in flights:
            d[t] = min(d[t], dp[f] + p)
        dp = d
        res = min(res, dp[dst])

    return res

# print(findCheapestPrice(3,[[0,1,100],[1,2,100],[0,2,500]],0,2,1))


def isBoomerang(points: List[List[int]]) -> bool:
    x, y, z = points[0], points[1], points[2]
    vector_xy = [y[0] - x[0], y[1] - x[1]]
    vector_xz = [z[0] - x[0], z[1] - x[1]]
    if vector_xy == vector_xz or vector_xy == [0, 0] or vector_xz == [0, 0] or vector_xy[0] * vector_xz[1] == vector_xy[1] * vector_xz[0]:
        return False
    return True

# print(isBoomerang([[1,1],[2,2],[3,3]]))


def isMatch(s: str, p: str) -> bool:
    m, n = len(s), len(p)

    def matches(i: int, j: int) -> bool:
        if i == 0:
            return False
        if p[j - 1] == '.':
            return True
        return s[i - 1] == p[j - 1]

    f = [[False] * (n + 1) for _ in range(m + 1)]
    f[0][0] = True
    for i in range(m + 1):
        for j in range(1, n + 1):
            if p[j - 1] == '*':
                f[i][j] |= f[i][j - 2]
                if matches(i, j - 1):
                    f[i][j] |= f[i - 1][j]
            else:
                if matches(i, j):
                    f[i][j] |= f[i - 1][j - 1]
    return f[m][n]


def countPalindromicSubsequences(s: str) -> int:
    n = len(s)
    mod = 1000000007
    if n == 0:
        return 0
    if n == 1:
        return 1
    if n == 2:
        return 2
    dp = [[0] * n for _ in range(n)]

    for i in range(n):
        dp[i][i] = 1
        if i < n-1:
            dp[i][i+1] = 2

    for i in range(n-3, -1, -1):
        for j in range(i+2, n):
            if s[i] == s[j]:
                left, right = i+1, j-1
                while left <= right and s[left] != s[i]:
                    left += 1
                while right >= left and s[right] != s[j]:
                    right -= 1
                if left > right:
                    dp[i][j] = 2 * dp[i+1][j-1] + 2
                elif left == right:
                    dp[i][j] = 2 * dp[i+1][j-1] + 1
                else:
                    dp[i][j] = 2 * dp[i+1][j-1] - dp[left+1][right-1]
            else:
                dp[i][j] = dp[i][j-1] + dp[i+1][j] - dp[i+1][j-1]
            dp[i][j] = dp[i][j] % mod
    return dp[0][n-1]
# 3104860382,104860361
# print(countPalindromicSubsequences('abcdabcdabcdabcdabcdabcdabcdabcddcbadcbadcbadcbadcbadcbadcbadcba'))


def candy(ratings: List[int]) -> int:
    n = len(ratings)
    left = [0] * n
    for i in range(n):
        if i > 0 and ratings[i] > ratings[i - 1]:
            left[i] = left[i - 1] + 1
        else:
            left[i] = 1

    right = ret = 0
    for i in range(n - 1, -1, -1):
        if i < n - 1 and ratings[i] > ratings[i + 1]:
            right += 1
        else:
            right = 1
        ret += max(left[i], right)

    return ret
# print(candy([1,2,87,87,87,2,1]))


def numDistinct(s: str, t: str) -> int:
    m, n = len(s), len(t)
    dp = [[0] * n for _ in range(m)]
    if s[m - 1] == t[n - 1]:
        dp[m - 1][n - 1] = 1

    for i in range(m - 2, -1, -1):
        if s[i] == t[n - 1]:
            dp[i][n - 1] = dp[i + 1][n - 1] + 1
        else:
            dp[i][n - 1] = dp[i + 1][n - 1]
    for i in range(m - 2, -1, -1):
        for j in range(n - 2, -1, -1):
            if m - i >= n - j:
                if s[i] == t[j]:
                    dp[i][j] = dp[i + 1][j] + dp[i + 1][j + 1]
                else:
                    dp[i][j] = dp[i + 1][j]

    return dp[0][0]

# print(numDistinct("babgbag","bag"))


def findMatrix(matrix: List[List]) -> List[int]:
    m, n = len(matrix), len(matrix[0])
    i = 0
    flag = True
    ans = []
    while i <= m + n - 2:
        pm = m if flag else n
        pn = n if flag else m
        x = pm - 1 if i >= pm else i
        y = i - x
        while x >= 0 and y < pn:
            if flag:
                ans.append(matrix[x][y])
            else:
                ans.append(matrix[y][x])
            x -= 1
            y += 1
        flag = not flag
        i += 1

    return ans

# print(findMatrix([[1,2,3],[4,5,6],[7,8,9]]))


def maxPoints(points: List[List[int]]) -> int:
    def gcd(x: int, y: int) -> int:
        return x if y == 0 else gcd(y, x % y)

    ans = 1
    n = len(points)

    for i in range(n):
        x = points[i]
        hashmap = {}
        mv = 0
        for j in range(i + 1, n):
            y = points[j]
            a, b = y[0] - x[0], y[1] - x[1]
            k = gcd(a, b)
            key = str(a // k) + " " + str(b // k)
            hashmap[key] = hashmap.get(key, 0) + 1
            mv = max(mv, hashmap[key])
        ans = max(ans, mv + 1)
    return ans


def minCut(s: str) -> int:
    n = len(s)
    dp = [[True] * n for _ in range(n)]
    for i in range(n - 2, -1, -1):
        for j in range(i + 1, n):
            dp[i][j] = (s[i] == s[j] and dp[i + 1][j - 1])

    f = [-1] + [i for i in range(n)]
    for i in range(1, n):
        mi = f[i] + 1
        for k in range(i):
            if dp[k][i]:
                mi = min(mi, f[k] + 1)
        f[i + 1] = min(f[i + 1], mi)
    return f[n]

# print(minCut("ab"))


def countDigitOne(n: int) -> int:
    r = 0
    for i in range(n + 1):
        while i > 0:
            d = i % 10
            if d == 1:
                r += 1
            i = i // 10
    return r

# print(countDigitOne(999))


def findPairs(nums: List[int], k: int) -> int:
    nums.sort()
    n = len(nums)
    l, r = 0, n - 1
    if nums[n - 1] - nums[l] < k:
        return 0

    while l < r and r < n:
        mid = l + (r - l) // 2
        if nums[mid] - nums[0] >= k:
            r = mid
        else:
            l = mid + 1

    if r == 0:
        l, r = 0, 1
    else:
        l, r = 0, l
    ans = 0
    while l <= r and r < n:
        if nums[r] - nums[l] > k:
            l += 1
        else:
            if r > l:
                if r - l == 1 or (r - l >= 2 and nums[r] != nums[r - 1]):
                    if nums[r] - nums[l] == k:
                        ans += 1
            r += 1

    return ans

# print(findPairs([1,2,4,4,3,3,0,9,2,3],3))


def duplicateZeros(arr: List[int]) -> None:
    n = len(arr)
    top = 0
    i = -1

    while top < n:
        i += 1
        top += 1 if arr[i] else 2
    j = n - 1
    if top == n + 1:
        arr[j] = 0
        j -= 1
        i -= 1
    while j >= 0:
        arr[j] = arr[i]
        j -= 1
        if arr[i] == 0:
            arr[j] = arr[i]
            j -= 1
        i -= 1

# print(duplicateZeros([1,0,2,3,0,4,5,0]))


def constructArray(n: int, k: int) -> List[int]:
    ans = [0] * n
    for i in range(n - k - 1):
        ans[i] = i + 1

    j = 0
    left, right = n - k, n

    for i in range(n - k - 1, n):
        if j == 0:
            ans[i] = left
            left += 1
            j = 1
        else:
            ans[i] = right
            right -= 1
            j = 0

    return ans

# print(constructArray(6, 1))


def checkRecord(n: int) -> int:
    mod = 10 ** 9 + 7
    dp = [[[0, 0, 0], [0, 0, 0]] for _ in range(n + 1)]
    dp[0][0][0] = 1

    for i in range(1, n + 1):

        for j in range(2):  # P
            for k in range(3):
                dp[i][j][0] = (dp[i - 1][j][k] + dp[i][j][0]) % mod

        for k in range(3):  # A
            dp[i][1][0] = (dp[i][1][0] + dp[i - 1][0][k]) % mod

        for j in range(2):  # L
            for k in range(1, 3):
                dp[i][j][k] = (dp[i][j][k] + dp[i - 1][j][k - 1]) % mod

    ans = 0
    for j in range(2):
        for k in range(3):
            ans += dp[n][j][k]

    return ans % mod

# print(checkRecord(10101))


def findSubstring(s: str, words: List[str]) -> List[int]:
    ans = []
    m, n, l = len(words), len(words[0]), len(s)
    m *= n
    i = 0
    while i < l - m + 1:
        cnt = Counter(words)
        left = i
        flag = True
        num = 0
        while left < i + m - n + 1:
            word = s[left: left + n]
            if word in cnt:
                cnt[word] -= 1
                if cnt[word] < 0:
                    break
                num += n
            else:
                flag = False
                break
            left += n
        if flag:
            if num == m:
                ans.append(i)
        i += 1
    return ans

# print(findSubstring("lingmindraboofooowingdingbarrwingmonkeypoundcake",
# ["fooo","barr","wing","ding","wing"]))


def findLUSlength(strs: List[str]) -> int:
    def is_subseq(s: str, t: str) -> bool:
        pt_s = pt_t = 0
        while pt_s < len(s) and pt_t < len(t):
            if s[pt_s] == t[pt_t]:
                pt_s += 1
            pt_t += 1
        return pt_s == len(s)

    ans = -1
    for i, s in enumerate(strs):
        check = True
        for j, t in enumerate(strs):
            if i != j and is_subseq(s, t):
                check = False
                break
        if check:
            ans = max(ans, len(s))

    return ans

# print(findLUSlength(["aabbcc", "aabbcc","cb"]))


def wiggleSort(nums: List[int]) -> None:
    array = sorted(nums)
    n = len(nums)
    mid = (n + 1) // 2 - 1
    right = n - 1
    for i in range(0, n, 2):
        nums[i] = array[mid]
        if i + 1 < n:
            nums[i + 1] = array[right]
        mid -= 1
        right -= 1

# print(wiggleSort([1,3,2,2,3,1]))


def maxSlidingWindow(nums: List[int], k: int) -> List[int]:
    q = deque()
    n = len(nums)
    res = []

    for j in range(0, k):
        if not q:
            q.append(nums[j])
        else:
            while q and q[-1] < nums[j]:
                q.pop()
            q.append(nums[j])
    if q:
        res.append(q[0])
        if q[0] == nums[0]:
            q.popleft()
    for i in range(1, n - k + 1):
        if not q:
            q.append(nums[i + k - 1])
        else:
            while q and q[-1] < nums[i + k - 1]:
                q.pop()
            q.append(nums[i + k - 1])
        res.append(q[0])
        if q[0] == nums[i]:
            q.popleft()
    return res

# print(maxSlidingWindow([1,3,-1,-3,5,3,6,7],3))


def diffWaysToCompute(expression: str) -> List[int]:

    ops = []
    i, j, n = 0, 0, len(expression)
    while j < n and i <= j:
        if expression[j].isdigit():
            j += 1
            if j >= n or (j < n and not expression[j].isdigit()):
                ops.append(expression[i: j])
        else:
            ops.append(expression[j])
            j += 1
            i = j

    @lru_cache(maxsize=None)
    def childWaysToCompute(l: int, r: int) -> List[int]:
        if l == r:
            return [int(ops[l])]
        res = []
        for i in range(l, r, 2):
            lchild = childWaysToCompute(l, i)
            rchild = childWaysToCompute(i + 2, r)
            for lc in lchild:
                for rc in rchild:
                    if ops[i + 1] == '-':
                        res.append(lc - rc)
                    if ops[i + 1] == '+':
                        res.append(lc + rc)
                    if ops[i + 1] == '*':
                        res.append(lc * rc)
        return res
    return childWaysToCompute(0, len(ops) - 1)

# print(diffWaysToCompute("11-1-1"))


def wiggleMaxLength(nums: List[int]) -> int:
    n = len(nums)
    m = 1
    dp = [[0] * n for _ in range(n)]
    for i in range(n):
        dp[i][i] = 1
        if i + 1 < n and nums[i] != nums[i + 1]:
            dp[i][i + 1] = 2
            m = 2

    for gap in range(2, n):
        for i in range(n):
            if i + gap < n:
                for j in range(i, i + gap):
                    if nums[j - 1] > nums[j] and nums[j] < nums[i + gap]:
                        dp[i][i + gap] = max(dp[i][i + gap], dp[i][j] + 1)
                    if nums[j - 1] < nums[j] and nums[j] > nums[i + gap]:
                        dp[i][i + gap] = max(dp[i][i + gap], dp[i][j] + 1)
                    m = max(m, dp[i][i + gap])

    return m

# print(wiggleMaxLength([3,3,3,2,5]))\


def slidingPuzzle(board: List[List[int]]) -> int:
    m, n = 2, 3
    target = "123450"
    s = ""
    for i in range(m):
        for j in range(n):
            s += str(board[i][j])

    step = 0
    hashmap = {0: [1, 3], 1: [0, 4, 2], 2: [
        1, 5], 3: [0, 4], 4: [3, 1, 5], 5: [4, 2]}
    q = deque()
    visited = set()
    q.append(s)
    visited.add(s)
    while q:
        size = len(q)
        for i in range(size):
            tail = q.pop()
            if tail == target:
                return step
            idx = tail.index("0")
            round = hashmap[idx]
            for j in round:
                string = list(tail)
                string[j], string[idx] = string[idx], string[j]
                cur = "".join(string)
                if cur not in visited:
                    q.appendleft(cur)
                    visited.add(cur)
        step += 1

    return -1

# print(slidingPuzzle([[4,1,2],[5,0,3]]))


def minAddToMakeValid(s: str) -> int:
    res = need = 0
    for char in s:
        if char == '(':
            need += 1
        if char == ')':
            need -= 1
            if need == -1:
                res += 1
                need = 0

    return res + need


def minInsertions(s: str) -> int:
    res = need = 0

    for char in s:
        if char == '(':
            need += 2
            if need & 1:
                res += 1
                need -= 1

        if char == ')':
            need -= 1
            if need == -1:
                res += 1
                need = 1

    return res + need


def removeCoveredIntervals(intervals: List[List[int]]) -> int:
    intervals.sort(key=lambda x: (x[0], -x[1]))
    ans = 0
    l, r = intervals[0][0], intervals[0][1]
    for interval in intervals[1:]:
        if l <= interval[0] <= r and l <= interval[1] <= r:
            ans += 1
        if l <= interval[0] <= r and r <= interval[1]:
            r = interval[1]
        if r < interval[0]:
            l, r = interval[0], interval[1]

    return len(intervals) - ans

# print(removeCoveredIntervals([[1,4],[3,6],[2,8]]))


def intervalIntersection(firstList: List[List[int]], secondList: List[List[int]]) -> List[List[int]]:
    l, r = 0, 0
    ans = []
    m, n = len(firstList), len(secondList)
    while l < m and r < n:
        x, y = firstList[l], secondList[r]
        if x[0] < y[0]:
            ans.append(x)
            l += 1
        elif x[0] == y[0]:
            if x[1] >= y[1]:
                ans.append(x)
                l += 1
            else:
                ans.append(y)
                r += 1
        else:
            ans.append(y)
            r += 1
    if l < m:
        ans += firstList[l:]
    if r < n:
        ans += secondList[r:]

    p, q = ans[0][0], ans[0][1]
    res = []
    for a in ans[1:]:
        if p <= a[0] <= q and p <= a[1] <= q:
            res.append([a[0], a[1]])
            continue
        if p <= a[0] <= q and q <= a[1]:
            res.append([a[0], q])
            q = a[1]
            continue
        if q < a[0]:
            p, q = a[0], a[1]

    return res


# print(intervalIntersection([[8, 15]], [[2, 6], [8, 10], [12, 20]]))


def oddCells(m: int, n: int, indices: List[List[int]]) -> int:
    row = [0] * m
    col = [0] * n
    for x, y in indices:
        row[x] += 1
        col[y] += 1

    ans = 0
    xl = sum([1 for x in row if x & 1])
    yr = sum([1 for y in col if y & 1])
    ans += xl * (n - yr) + yr * (m - xl)
    return ans


def asteroidCollision(asteroids: List[int]) -> List[int]:
    ans = []

    for asteroid in asteroids:
        alive = True
        while ans and ans[-1] * asteroid < 0:
            if ans[-1] < 0 and asteroid > 0:
                break
            if abs(ans[-1]) < abs(asteroid):
                ans.pop()
            elif abs(ans[-1]) > abs(asteroid):
                alive = False
                break
            elif abs(ans[-1]) == abs(asteroid):
                alive = False
                ans.pop()
                break
        if alive:
            ans.append(asteroid)

    return ans

# print(asteroidCollision([10,2,-5]))


class Trie:
    def __init__(self) -> None:
        self.hashmap = dict()
        self.ids = list()


class WordFilter:

    def add(self, word: str, idx: int, isEnd: bool) -> None:
        trie = self.ltrie if isEnd else self.rtrie
        trie.ids.append(idx)
        for w in word:
            if w not in trie.hashmap:
                trie.hashmap[w] = Trie()
            trie = trie.hashmap[w]
            trie.ids.append(idx)

    def query(self, string: str, isEnd: bool) -> List[int]:
        trie = self.ltrie if isEnd else self.rtrie
        for char in string:
            if char not in trie.hashmap:
                return None
            trie = trie.hashmap[char]
        return trie.ids

    def __init__(self, words: List[str]):
        self.ltrie = Trie()
        self.rtrie = Trie()
        for idx, word in enumerate(words):
            self.add(word, idx, True)
            self.add(word[::-1], idx, False)

    def f(self, pref: str, suff: str) -> int:
        l = self.query(pref, True)
        r = self.query(suff[::-1], False)

        if l and r:
            i, j = len(l), len(r)
            while i > 0 and j > 0:
                if l[i - 1] > r[j - 1]:
                    i -= 1
                elif l[i - 1] < r[j - 1]:
                    j -= 1
                elif l[i - 1] == r[j - 1]:
                    return l[i - 1]
        return -1


# Definition for a QuadTree node.
class Node:
    def __init__(self, val, isLeaf, topLeft, topRight, bottomLeft, bottomRight):
        self.val = val
        self.isLeaf = isLeaf
        self.topLeft = topLeft
        self.topRight = topRight
        self.bottomLeft = bottomLeft
        self.bottomRight = bottomRight


class Solution:
    def intersect(self, quadTree1: 'Node', quadTree2: 'Node') -> 'Node':

        node = Node(0, 0, None, None, None, None)
        allLeaf = False

        if quadTree1.isLeaf or quadTree2.isLeaf:

            if quadTree1.isLeaf and quadTree2.isLeaf:
                node.isLeaf = 1
                node.val = quadTree1.val | quadTree2.val
            else:
                if quadTree1.isLeaf and quadTree1.val == 1:
                    node.isLeaf = 1
                    node.val = 1

                if quadTree2.isLeaf and quadTree2.val == 1:
                    node.isLeaf = 1
                    node.val = 1

                if quadTree1.isLeaf and quadTree1.val == 0:
                    node = quadTree2

                if quadTree2.isLeaf and quadTree2.val == 0:
                    node = quadTree1
        else:

            allLeaf = True
            node.topLeft = self.intersect(quadTree1.topLeft, quadTree2.topLeft)
            allLeaf &= node.topLeft.isLeaf
            node.topRight = self.intersect(
                quadTree1.topRight, quadTree2.topRight)
            allLeaf &= node.topRight.isLeaf
            node.bottomLeft = self.intersect(
                quadTree1.bottomLeft, quadTree2.bottomLeft)
            allLeaf &= node.bottomLeft.isLeaf
            node.bottomRight = self.intersect(
                quadTree1.bottomRight, quadTree2.bottomRight)
            allLeaf &= node.bottomRight.isLeaf

        if allLeaf:
            if node.topLeft.val == node.topRight.val == node.bottomLeft.val == node.bottomRight.val:
                node.isLeaf = 1
                node.val = node.topLeft.val
                node.topLeft = node.topRight = node.bottomLeft = node.bottomRight = None
        return node


def sequenceReconstruction(nums: List[int], sequences: List[List[int]]) -> bool:

    adj = defaultdict(set)
    adj1 = defaultdict(set)
    for seq in sequences:
        for i in range(len(seq) - 1):
            j = i + 1
            adj[seq[j]].add(seq[i])
            adj1[seq[i]].add(seq[j])

    indegree = [0] * (len(nums) + 1)
    q = deque()
    for i in range(1, len(indegree)):
        indegree[i] = len(adj[i])
        if indegree[i] == 0:
            q.append(i)
    adj.clear()
    ans = []
    while q:
        size = len(q)
        if size > 1:
            return False
        pre = q.popleft()
        ans.append(pre)
        for edge in adj1[pre]:
            indegree[edge] -= 1
            if indegree[edge] == 0:
                q.append(edge)

    return ans == nums

# print(sequenceReconstruction([4,1,5,2,6,3], [[5,2,6,3],[4,1,5,2]]))


def loudAndRich(richer: List[List[int]], quiet: List[int]) -> List[int]:
    ans = []

    n = len(quiet)
    adj = defaultdict(set)
    indegree = [0] * n

    ans = [i for i in range(n)]
    q = deque()
    for i, j in richer:
        adj[i].add(j)
        indegree[j] = indegree[j] + 1
    for i, x in enumerate(indegree):
        if x == 0:
            q.append(i)

    while q:
        size = len(q)

        for _ in range(size):
            pre = q.popleft()
            for edge in adj[pre]:
                idx = ans[edge]
                preIdx = ans[pre]
                if quiet[preIdx] < quiet[idx]:
                    ans[edge] = preIdx
                indegree[edge] -= 1
                if indegree[edge] == 0:
                    q.append(edge)

    return ans

# print(loudAndRich([], [0]))


def productExceptSelf(nums: List[int]) -> List[int]:
    n = len(nums)
    pre = [1] * n
    suf = [1] * n
    ans = [0] * n
    pre[0] = nums[0]
    suf[-1] = nums[-1]
    for i in range(1, n):
        pre[i] = pre[i - 1] * nums[i]
    for j in range(n - 2, -1, -1):
        suf[j] = suf[j + 1] * nums[j]

    for index in range(n):
        if index == 0:
            ans[index] = suf[1]
        elif index == n - 1:
            ans[index] = pre[n - 2]
        else:
            ans[index] = pre[index - 1] * suf[index + 1]

    return ans


# print(productExceptSelf([1,2,3,4]))


class MyCalendarTwo:

    def __init__(self):
        self.d = SortedDict()

    def book(self, start: int, end: int) -> bool:
        self.d[start] = self.d.setdefault(start, 0) + 1
        self.d[end] = self.d.setdefault(end, 0) - 1

        ans = maxscore = 0
        for v in self.d.values():
            maxscore += v
            ans = max(ans, maxscore)
            if ans >= 3:
                if self.d[start] - 1 == 0:
                    del self.d[start]
                else:
                    self.d[start] -= 1
                if self.d[end] + 1 == 0:
                    del self.d[end]
                else:
                    self.d[end] += 1
                return False
        return True


def findMin(nums: List[int]) -> int:
    l, r = 0, len(nums) - 1

    while l < r:
        mid = l + (r - l) // 2
        if nums[l] == nums[mid] == nums[r]:
            l += 1
            r -= 1
        elif nums[l] <= nums[mid] <= nums[r]:
            r = mid - 1
        elif nums[l] >= nums[mid] and nums[mid] <= nums[r]:
            r = mid
        elif nums[l] <= nums[mid] and nums[mid] >= nums[r]:
            l = mid + 1

    return nums[l]

# print(findMin([1,10,10,10]))


def isValid(s: str) -> bool:
    need = 0
    for char in s:
        if need < 0:
            return False
        if char == '(':
            need += 1
        if char == ')':
            need -= 1
    return need == 0


def removeInvalidParentheses(s: str) -> List[str]:
    lneed = rneed = 0
    for char in s:
        if char == '(':
            lneed += 1
        if char == ')':
            if lneed:
                lneed -= 1
            else:
                rneed += 1

    ans = []

    def dfs(s: str, index: int, lneed: int, rneed: int) -> None:
        nonlocal ans
        if lneed == rneed == 0:
            if isValid(s):
                ans.append(s)
            return
        if lneed + rneed > len(s):
            return
        for i in range(index, len(s)):
            char = s[i]
            # 防止回头
            if i > index and s[i] == s[i - 1]:
                continue
            if lneed and char == '(':
                dfs(s[:i] + s[i + 1:], i, lneed - 1, rneed)
            if rneed and char == ')':
                dfs(s[:i] + s[i + 1:], i, lneed, rneed - 1)

    dfs(s, 0, lneed, rneed)
    return ans

# print(removeInvalidParentheses("))("))


def shiftGrid(grid: List[List[int]], k: int) -> List[List[int]]:
    m, n = len(grid), len(grid[0])
    tmp = [0] * m
    for _ in range(k):
        for i in range(m):
            tmp[i] = grid[i][0]
        for j in range(1, n):
            tmp1 = [0] * m
            for i in range(m):
                tmp1[i] = grid[i][j]
                if j == 1:
                    grid[i][j] = grid[i][j - 1]
                else:
                    grid[i][j] = tmp[i]
            tmp = tmp1

        for i in range(m):
            grid[(i + 1) % m][0] = tmp[i]

    return grid

# print(shiftGrid([[1,2,3]], 100))


def maxSubarraySumCircular(A: List[int]) -> int:
    n = len(A)
    p = [0]
    for _ in range(2):
        for a in A:
            p.append(p[-1] + a)
    ans = A[0]
    q = deque([0])
    for i in range(1, len(p)):
        if i - q[0] > n:
            q.popleft()
        ans = max(ans, p[i] - p[q[0]])
        while q and p[q[-1]] >= p[i]:
            q.pop()
        q.append(i)
    return ans

# print(maxSubarraySumCircular([5,-3,5]))


def longestSubarray(nums: List[int], limit: int) -> int:
    s = SortedList()
    s.add(nums[0])
    pre = 0
    ans = 1
    n = len(nums)
    for i in range(1, n):
        s.add(nums[i])
        mx, mn = s[-1], s[0]
        while mx - mn > limit and pre < i:
            s.remove(nums[pre])
            mx = s[-1]
            mn = s[0]

            pre += 1
        if mx - mn <= limit:
            ans = max(ans, i - pre + 1)
    return ans

# print(longestSubarray([10,1,2,4,7,2], 5))


def intersectionSizeTwo(intervals: List[List[int]]) -> int:
    # 区间调度问题，贪心方法，可用射箭思路来解，集合S为射箭的位置集合
    intervals.sort(key=lambda x: (x[1], -x[0]))  # 从最内部区间开始计算
    first, second = intervals[0][1] - 1, intervals[0][1]
    ans = 2
    for x, y in intervals[1:]:
        if x <= first:
            continue
        if first < x <= second:
            ans += 1
            first = second
            second = y
        else:
            first = y - 1
            second = y
            ans += 2

    return ans

# print(intersectionSizeTwo([[1, 3], [1, 4], [2, 5], [3, 5]]))


def constrainedSubsetSum(nums: List[int], k: int) -> int:
    n = len(nums)
    q = deque()
    ans = float("-inf")
    for i in range(k + 1):
        insert = nums[i]
        if q:
            insert += q[0][0]
            if insert < nums[i]:
                insert = nums[i]
        while q and insert > q[-1][0]:
            q.pop()
        q.append((insert, i))
        ans = max(ans, q[0][0])

    for j in range(k + 1, n):
        node = q[0]
        if node[1] < j - k:
            q.popleft()
        insert = nums[j]
        if q:
            insert += q[0][0]
            if insert < nums[j]:
                insert = nums[j]
        while q and insert > q[-1][0]:
            q.pop()
        q.append((insert, j))
        ans = max(ans, q[0][0])

    return ans
# 11355
# print(constrainedSubsetSum([-5266,4019,7336,-3681,-5767], 2))

# [1,-1,-2,4,-7,3], 2
# 你可以选择子序列 [1,-1,4,3] （上面加粗的数字），和为 7


def maxResult(nums: List[int], k: int) -> int:
    n = len(nums)
    q = deque()

    for i in range(1, min(k + 1, n)):
        insert = nums[i]
        if q:
            insert += q[0][0]
            if insert < nums[i] + nums[0]:
                insert = nums[i] + nums[0]
        while q and insert > q[-1][0]:
            q.pop()
        if i == 1:
            insert += nums[0]
        q.append((insert, i))

    for j in range(k + 1, n - 1):
        node = q[0]
        if node[1] < j - k:
            q.popleft()
        insert = nums[j]
        if q:
            insert += q[0][0]

        while q and insert > q[-1][0]:
            q.pop()
        q.append((insert, j))

    if n - 1 - k > 0 and q[0][1] < n - 1 - k:
        q.popleft()
    return nums[-1] + q[0][0] if q else nums[-1]

# print(maxResult([-123], 10))

# Definition for a binary tree node.


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class CBTInserter:

    def __init__(self, root: TreeNode):
        self.q = deque()
        self.root = root
        tmp = deque([root])
        while tmp:
            node = tmp.popleft()
            if node.left:
                tmp.append(node.left)
            if node.right:
                tmp.append(node.right)
            if not(node.left and node.right):
                self.q.append(node)

    def insert(self, v: int) -> int:
        q = self.q
        tmp = TreeNode(v)

        node = q[0]
        res = node.val
        if not node.left:
            node.left = tmp
        elif not node.right:
            node.right = tmp
            q.popleft()
        q.append(tmp)

        return res

    def get_root(self) -> TreeNode:
        return self.root


MAX_LEVEL = 32
P_FACTOR = 0.25


def random_level() -> int:
    lv = 1
    while lv < MAX_LEVEL and random.random() < P_FACTOR:
        lv += 1
    return lv


class SkiplistNode:
    __slots__ = 'val', 'forward'

    def __init__(self, val: int, max_level=MAX_LEVEL):
        self.val = val
        self.forward = [None] * max_level


class Skiplist:
    def __init__(self):
        self.head = SkiplistNode(-1)
        self.level = 0

    def search(self, target: int) -> bool:
        curr = self.head
        for i in range(self.level - 1, -1, -1):
            # 找到第 i 层小于且最接近 target 的元素
            while curr.forward[i] and curr.forward[i].val < target:
                curr = curr.forward[i]
        curr = curr.forward[0]
        # 检测当前元素的值是否等于 target
        return curr is not None and curr.val == target

    def add(self, num: int) -> None:
        update = [self.head] * MAX_LEVEL
        curr = self.head
        for i in range(self.level - 1, -1, -1):
            # 找到第 i 层小于且最接近 num 的元素
            while curr.forward[i] and curr.forward[i].val < num:
                curr = curr.forward[i]
            update[i] = curr
        lv = random_level()
        self.level = max(self.level, lv)
        new_node = SkiplistNode(num, lv)
        for i in range(lv):
            # 对第 i 层的状态进行更新，将当前元素的 forward 指向新的节点
            new_node.forward[i] = update[i].forward[i]
            update[i].forward[i] = new_node

    def erase(self, num: int) -> bool:
        update = [None] * MAX_LEVEL
        curr = self.head
        for i in range(self.level - 1, -1, -1):
            # 找到第 i 层小于且最接近 num 的元素
            while curr.forward[i] and curr.forward[i].val < num:
                curr = curr.forward[i]
            update[i] = curr
        curr = curr.forward[0]
        if curr is None or curr.val != num:  # 值不存在
            return False
        for i in range(self.level):
            if update[i].forward[i] != curr:
                break
            # 对第 i 层的状态进行更新，将 forward 指向被删除节点的下一跳
            update[i].forward[i] = curr.forward[i]
        # 更新当前的 level
        while self.level > 1 and self.head.forward[self.level - 1] is None:
            self.level -= 1
        return True


def poorPigs(buckets: int, minutesToDie: int, minutesToTest: int) -> int:
    pigs, k = 0, minutesToTest // minutesToDie + 1
    while math.pow(k, pigs) < buckets:
        pigs += 1

    return pigs

# print(poorPigs(4, 15, 15))


def fractionAddition(expression: str) -> str:
    def gcd(a: int, b: int) -> int:
        c = a % b
        if c == 0:
            return b
        else:
            return gcd(b, c)

    def strToint(s: str) -> tuple:
        tmp = s.split("/")
        numerator, denominator = int(tmp[0]), int(tmp[1])
        return numerator, denominator

    def add(a: str, b: str) -> str:
        al, ar = strToint(a)
        bl, br = strToint(b)

        if ar == 1 and br == 1:
            return str(al + bl) + "/1"

        if al == 0 or bl == 0:
            if al == 0 and bl == 0:
                return "0"
            if al == 0:
                return b[1:] if bl > 0 else b
            else:
                return a[1:] if al > 0 else a

        g = gcd(ar, br)
        x = al * (br // g) + bl * (ar // g)
        y = ar * (br // g)

        if x == 0:
            return "0/1"
        g1 = gcd(abs(x), y)
        return str(x // g1) + "/" + str(y // g1)

    a, b = "", ""
    index = 0
    n = len(expression)
    for i in range(n):
        if (i > 0 and (expression[i] == '+' or expression[i] == '-')) or i == n - 1:
            if b == "":
                b = expression[index: i] if i < n - 1 else expression[index: n]
                index = i
            elif a == "":
                a = expression[index: i] if i < n - 1 else expression[index: n]
                index = i
            if a == "":
                continue
            a = add(a, b)
            b = ""
    return a if a != "" else b


# print(fractionAddition("-1/2+1/2+1/3"))

def arrayRankTransform(arr: List[int]) -> List[int]:
    hashmap = dict()
    atharr = sorted(arr)
    hashmap[atharr[0]] = 1
    i = 1
    while i < len(arr):
        if atharr[i] != atharr[i - 1]:
            hashmap[atharr[i]] = hashmap[atharr[i - 1]] + 1
        i += 1

    for i, a in enumerate(arr):
        arr[i] = hashmap[a]

    return arr

# print(arrayRankTransform([37,12,28,9,100,56,80,5,12]))


def shortestSubarray(nums: List[int], k: int) -> int:
    n = len(nums)
    sumNums = [0] * (n + 1)
    for i, num in enumerate(nums):
        sumNums[i + 1] = sumNums[i] + num
    q = deque([(0, 0)])
    ans = n + 1
    for i in range(1, n + 1):
        while q and sumNums[i] - q[0][0] >= k:
            ans = min(ans, i - q[0][1])
            q.popleft()
        while q and sumNums[i] <= q[-1][0]:
            q.pop()
        q.append((sumNums[i], i))
    return ans if ans != n + 1 else -1

# print(shortestSubarray([2, -1, 2, 1], 3))


def validSquare(p1: List[int], p2: List[int], p3: List[int], p4: List[int]) -> bool:
    p = [p1, p2, p3, p4]
    edges = list()
    for i in range(3):
        for j in range(i + 1, 4):
            edges.append(
                sqrt(pow(p[i][0] - p[j][0], 2) + pow(p[i][1] - p[j][1], 2)))

    edges.sort()
    if (edges[0] == edges[1] == edges[2] == edges[3]) and (edges[4] == edges[5]) and (edges[3] < edges[4]):
        return True
    else:
        return False


class UF:
    def __init__(self, n: int):
        self.n = n
        self.cnt = n
        self.parent = [0] * self.n
        for i in range(self.n):
            self.parent[i] = i

    def union(self, p: int, q: int):
        rootp = self.find(p)
        rootq = self.find(q)
        if rootp == rootq:
            return
        self.parent[rootp] = rootq
        self.cnt -= 1

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def connected(self, p: int, q: int) -> bool:
        rootp = self.find(p)
        rootq = self.find(q)
        return rootp == rootq

    def count(self) -> int:
        return self.cnt


def largestComponentSize(nums: List[int]) -> int:
    n = len(nums)
    uf = UF(max(nums) + 1)
    for i in range(0, n):
        j = 2
        while j * j <= nums[i]:
            if nums[i] % j == 0:
                uf.union(nums[i], j)
                uf.union(nums[i], nums[i] // j)
            j += 1

    return max(Counter((uf.find(num) for num in nums)).values())

# print(largestComponentSize([20,50,9,63]))


base = 10 ** 9 + 7


def cuttingRope(n: int) -> int:
    dp = [0] * (n + 1)
    dp[1] = 1
    dp[2] = 1

    for i in range(3, n + 1):
        for j in range(1, i):
            dp[i] = max(dp[i], max(j * (i - j), dp[j] * (i - j)))

    return dp[n] % base


# print(cuttingRope(58))

class MyCircularQueue:

    def __init__(self, k: int):
        self.array = [-1] * k
        self.volume = 0
        self.num = 0

    def enQueue(self, value: int) -> bool:
        if self.volume >= len(self.array):
            return False
        self.num %= len(self.array)
        self.array[self.num] = value
        self.num += 1
        self.volume += 1
        return True

    def deQueue(self) -> bool:
        if self.volume == 0:
            return False
        self.array[(self.num - self.volume + len(self.array)) %
                   len(self.array)] = -1
        self.volume -= 1
        return True

    def Front(self) -> int:
        if self.volume == 0:
            return -1
        res = self.array[(self.num - self.volume +
                          len(self.array)) % len(self.array)]

        return res

    def Rear(self) -> int:
        if self.volume == 0:
            return -1
        res = self.array[(self.num - 1) % len(self.array)]

        return res

    def isEmpty(self) -> bool:
        return self.volume == 0

    def isFull(self) -> bool:
        return self.volume >= len(self.array)


def orderlyQueue(s: str, k: int) -> str:
    s = list(s)
    ans = s
    if k > 1:
        ans.sort()
        return "".join(ans)
    for i in range(1, len(s)):
        tmp = s[i:] + s[: i]
        m = n = 0
        while m < len(s) and n < len(s):
            if ans[m] == tmp[n]:
                m += 1
                n += 1
            elif ans[m] > tmp[n]:
                m += 1
                break
            else:
                n += 1
                break
        if m > n:
            ans = tmp
    return "".join(ans)


# print(orderlyQueue("nhtq", 1))


def minWindow(s: str, t: str) -> str:
    def cmp(s: dict, t: dict) -> bool:
        for k, v in t.items():
            if s[k] < v:
                return False
        return True
    l = r = 0
    n = len(s)
    cnnt = Counter(t)
    cnns = dict()
    length = n
    ans = ""
    while l <= r and r < n:
        if s[r] in cnnt:
            cnns[s[r]] = cnns.get(s[r], 0) + 1
            while l <= r and len(cnns) == len(cnnt) and cmp(cnns, cnnt):
                if s[l] in cnnt:
                    if r - l + 1 <= length:
                        length = r - l + 1
                        ans = s[l: r + 1]
                    cnns[s[l]] -= 1
                    if cnns[s[l]] == 0:
                        del cnns[s[l]]
                l += 1
        r += 1
    return ans


def calculate(s: str) -> int:
    hashmap = {'+': 0, '-': 1, '*': 2, '/': 2}
    stack = []
    array = []
    tmp = ""
    for char in s:
        if char != ' ':
            tmp += char
    l = 0
    for i, char in enumerate(tmp):
        if char.isdigit():
            if i == len(tmp) - 1:
                array.append(int(tmp[l:]))
            continue
        else:
            array.append(int(tmp[l: i]))

            while stack and hashmap[char] <= hashmap[stack[-1]]:
                sign = stack.pop()
                a = array.pop()
                b = array.pop()
                if sign == '*':
                    array.append(a * b)
                if sign == '/':
                    array.append(b // a)
                if sign == '-':
                    array.append(b - a)
                if sign == '+':
                    array.append(a + b)
            stack.append(char)
            l = i + 1
    while stack:
        sign = stack.pop()
        a = array.pop()
        b = array.pop()
        if sign == '+':
            array.append(a + b)
        if sign == '-':
            array.append(b - a)
        if sign == '*':
            array.append(a * b)
        if sign == '/':
            array.append(b // a)
    return array[-1]

# print(calculate("1+2*5/3+6/4*2"))


def missingTwo(nums: List[int]) -> List[int]:
    if len(nums) == 1:
        if nums[0] == 1:
            return [2, 3]
        else:
            return [1, 2] if nums[0] > 2 else [1, 3]
    nums.sort()
    ans = []
    for i in range(len(nums) - 1):
        if nums[i] + 1 < nums[i + 1]:
            ans.append(nums[i])
    if len(ans) == 2:
        return ans
    elif len(ans) == 1:
        if nums[0] == 1:
            ans.append(nums[-1] + 1)
        else:
            ans.append(1)

    return ans

# print(missingTwo([1]))


def stringMatching(words: List[str]) -> List[str]:
    res = []
    for i, word in enumerate(words):
        for j in range(len(words)):
            if i == j:
                continue
            if words[j].find(word) >= 0:
                res.append(word)
                break
    return res
# print(stringMatching(["leetcoder","leetcode","od","hamlet","am"]))


def getKthMagicNumber(k: int) -> int:
    q = [1]
    heapify(q)
    q3 = [3]
    q5 = [5]
    q7 = [7]
    heapify(q3)
    heapify(q5)
    heapify(q7)

    if k == 1:
        return 1
    for _ in range(1, k):
        node3 = q3[0]
        node5 = q5[0]
        node7 = q7[0]
        mi = min(node3, node5, node7)
        if mi == node3:
            heappush(q, node3)
            heappop(q3)
            if node3 % 5 == 0 and node3 % 7 != 0:
                heappush(q3, node3 * 5)
                heappush(q3, node3 * 7)
            if node3 % 7 == 0:
                heappush(q3, node3 * 7)
            if node3 % 5 != 0 and node3 % 7 != 0:
                heappush(q3, node3 * 3)
                heappush(q3, node3 * 5)
                heappush(q3, node3 * 7)
        if mi == node5:
            heappush(q, node5)
            heappop(q5)
            if node5 % 7:
                heappush(q5, node5 * 5)
                heappush(q5, node5 * 7)
            else:
                heappush(q5, node5 * 7)
        if mi == node7:
            heappush(q, node7)
            heappop(q7)
            heappush(q7, node7 * 7)
        heappop(q)
    return q[0]

# print(getKthMagicNumber(251))


def minStartValue(nums: List[int]) -> int:
    minV = float("inf")
    prev = 0
    for num in nums:
        prev += num
        minV = min(minV, prev)
    return 1 if minV > 0 else 1 - minV


def numSimilarGroups(strs: List[str]) -> int:
    n = len(strs)
    fa = list(range(n))

    def find(x: int) -> int:
        if fa[x] != x:
            fa[x] = find(fa[x])
        return fa[x]

    def check(str1: str, str2: str) -> bool:
        num = 0
        for s1, s2 in zip(str1, str2):
            if s1 != s2:
                num += 1
            if num > 2:
                return False
        return True

    for i in range(n - 1):
        for j in range(i + 1, n):
            if check(strs[i], strs[j]):
                fi, fj = find(i), find(j)
                if fi == fj:
                    continue
                else:
                    fa[fj] = fi

    return sum(1 for i in range(n) if fa[i] == i)


def solveEquation(equation: str) -> str:
    left, right = equation.split("=")
    if left[0] != '-':
        left = "+" + left
    if right[0] != '-':
        right = "+" + right
    xstr = 0
    intstr = 0
    m, n = len(left), len(right)

    def find(left: list, m: int, flag: int) -> None:
        nonlocal xstr, intstr
        i = 0
        for j in range(1, m):
            if j == m - 1:
                if left[j] == 'x':
                    if j - i == 1:
                        xstr += int(left[i: j] + "1") * flag
                    else:
                        xstr += int(left[i: j]) * flag
                else:
                    intstr += int(left[i:]) * (-flag)
            if left[j] == '+' or left[j] == '-':
                if left[j - 1] == 'x':
                    if j - 1 - i > 1:
                        xstr += int(left[i: j - 1]) * flag
                    else:
                        xstr += int(left[i: j - 1] + "1") * flag
                else:
                    intstr += int(left[i: j]) * (-flag)
                i = j
    find(left, m, 1)
    find(right, n, -1)
    if xstr == intstr == 0:
        return "Infinite solutions"
    if xstr == 0 and intstr != 0:
        return "No solution"
    return "x=" + str(intstr // xstr)


# print(solveEquation("1-x+x-x+x=-99-x+x-x+x"))


def reformat(s: str) -> str:
    res = ""
    number, chars = [], []
    for char in s:
        if char.isdigit():
            number.append(char)
        else:
            chars.append(char)
    if abs(len(number) - len(chars)) > 1:
        return ""
    i = 0
    j = 0
    while i < len(number) and j < len(chars):
        if len(number) > len(chars):
            res += number[i]
            res += chars[j]
        else:
            res += chars[j]
            res += number[i]
        i += 1
        j += 1
    if j < len(chars):
        res += chars[-1]
    if i < len(number):
        res += number[-1]
    return res


def groupThePeople(groupSizes: List[int]) -> List[List[int]]:
    hashmap = defaultdict(list)
    for i, group in enumerate(groupSizes):
        hashmap[group].append(i)

    res = []
    for k, v in hashmap.items():
        if len(v) == k:
            res.append(v)
        else:
            i = len(v) // k
            for j in range(1, i + 1):
                res.append(v[k * (j - 1): k * j])
    return res


def maxChunksToSorted(arr: List[int]) -> int:
    stack = []
    for a in arr:
        if stack:
            if a >= stack[-1]:
                stack.append(a)
            else:
                top = stack.pop()
                while stack and stack[-1] > a:
                    stack.pop()
                stack.append(top)
        else:
            stack.append(a)

    return len(stack)


def canPartitionKSubsets(nums: List[int], k: int) -> bool:
    n = len(nums)
    num = sum(nums)
    if num % k:
        return False
    hashmap = dict()
    var = num // k

    def dfs(l: int, r: int, tmp: int, used: int) -> bool:
        if l == k - 1:
            return True

        if tmp == var:
            rs = dfs(l + 1, 0, 0, used)
            hashmap[used] = rs
            return rs

        if used in hashmap:
            return hashmap[used]
        for i in range(r, n):
            if (used >> i) & 1 == 0:
                tmp += nums[i]
                if tmp > var:
                    tmp -= nums[i]
                    continue
                used |= 1 << i

                if dfs(l, i + 1, tmp, used):
                    return True

                used ^= 1 << i
                tmp -= nums[i]
        return False
    return dfs(0, 0, 0, 0)

# print(canPartitionKSubsets([1,1,1,1,2,2,2,2],4))


def largestRectangleArea(heights: List[int]) -> int:
    stack = []
    ans = heights.copy()
    idx = 0
    for i, height in enumerate(heights):
        if stack:
            tmp = 0

            while stack and heights[stack[-1]] > height:
                j = stack.pop()
                ans[j] += (i - j - 1) * heights[j]
            if stack:
                tmp = max(tmp, (i - stack[-1] - 1) * height)
                stack.append(i)
            else:
                tmp = max(tmp, (i - idx) * height)
                stack.append(i)
            ans[i] += tmp
        else:
            stack.append(i)

    for i in stack[:-1]:
        ans[i] += (stack[-1] - i) * heights[i]
    return max(ans)

# print(largestRectangleArea([999,999,999,999]))


def maxEqualFreq(nums: List[int]) -> int:
    n = len(nums)
    end = [0] * n
    start = [0] * n

    def findNear(nums: List[int], end: List[int]):
        need = 0
        cur = None
        i = j = 0
        zeroSum = 0
        begin = 0
        while j < n and i <= j:
            if j == 0:
                cur = nums[j]
                need += 1
            else:
                if cur == nums[j]:
                    need += 1
                else:
                    need -= 1

            if need >= 1:
                if nums[j - 1] == cur:
                    end[j] = j - i
                    begin = j
                else:
                    end[j] = j - begin
            elif need == -1:
                end[j] = zeroSum
                i = j
                need = 1
                cur = nums[j]
            elif need == 0:
                if nums[j - 1] == cur:
                    end[j] = j - i

                    i = j
                    need = 1
                else:
                    end[j] = j - begin
                zeroSum += j - i + 1
                cur = nums[j]
            j += 1
    findNear(nums, end)
    findNear(nums[::-1], start)
    start = start[::-1]
    res = 0
    for i in range(n):
        if i == 0 or i == n - 1 or (not end[i] % 2 and not start[i] % 2):
            res = max(res, end[i] + start[i])
        res = max(res, 2 * min(end[i], start[i]))
    return res + 1

# print(maxEqualFreq([1,1,1,2,2,2]))


def busyStudent(self, startTime: List[int], endTime: List[int], queryTime: int) -> int:
    return sum(1 if start <= queryTime <= end else 0 for start, end in zip(startTime, endTime))


def compareVersion(version1: str, version2: str) -> int:
    res = 0
    v1, v2 = version1.split('.'), version2.split('.')
    m, n = len(v1), len(v2)
    i = j = 0
    while i <= m or j <= n:
        f1 = f2 = 0
        if i < m:
            for x in v1[i]:
                f1 = f1 * 10 + int(x)
        if j < n:
            for x in v2[j]:
                f2 = f2 * 10 + int(x)
        if f1 > f2:
            res = 1
            break
        if f1 < f2:
            res = -1
            break
        i += 1
        j += 1
        if i == m and j == n:
            break
    return res
# print(compareVersion("2.1", "1.1"))


def computeArea(ax1: int = 0, ay1: int = 0, ax2: int = 0, ay2: int = 0, bx1: int = 0, by1: int = 0, bx2: int = 0, by2: int = 0) -> int:
    res = abs(ax1 - ax2) * abs(ay1 - ay2) + abs(bx1 - bx2) * abs(by1 - by2)
    if ax1 >= bx2 or bx1 >= ax2:
        return res
    if ay2 <= by1 or by2 <= ay1:
        return res
    x = [ax1, ax2, bx1, bx2]
    y = [ay1, ay2, by1, by2]
    x.sort()
    y.sort()
    repeated = abs(x[1] - x[2]) * abs(y[1] - y[2])
    return res - repeated

# print(computeArea(ax1 = -3, ay1 = 0, ax2 = 3, ay2 = 4, bx1 = 0, by1 = -1, bx2 = 9, by2 = 2))


def mirrorReflection(p: int, q: int) -> int:
    def lineY(k, x, x1, y1):
        y = k * (x - x1) + y1
        return y

    def lineX(k, y, x1, y1):
        x = (y - y1) / k + x1
        return x

    points = [[p, 0], [p, p], [0, p]]
    ans = -1
    k = Fraction(q, p)
    x1, y1 = Fraction(p), Fraction(q)
    while ans == -1:

        for i, point in enumerate(points):
            x, y = lineX(k, point[1], x1, y1), lineY(k, point[0], x1, y1)
            if x == Fraction(point[0]) and y == Fraction(point[1]):
                ans = i
                break

        if ans != -1:
            break
        k = -k
        for x, y in [[p, p], [0, 0]]:
            tmpx = lineX(k, y, x1, y1)
            x, y = Fraction(x), Fraction(y)
            if Fraction(0) <= tmpx <= Fraction(p) and (tmpx, y) != (x1, y1):
                x1, y1 = tmpx, y
                break
            tmpy = lineY(k, x, x1, y1)
            if Fraction(0) <= tmpy <= Fraction(p) and (x, tmpy) != (x1, y1):
                x1, y1 = x, tmpy
                break
    return ans

# print(mirrorReflection(69, 58))


def findClosestElements(arr: List[int], k: int, x: int) -> List[int]:
    q = []
    for a in arr:
        q.append((abs(a - x), a))

    heapify(q)

    res = SortedList()

    if len(q) < k:
        return None

    for _ in range(k):
        num = heappop(q)
        res.add(num[1])
    return list(res)


def getSkyline(buildings: List[List[int]]) -> List[List[int]]:
    building = []
    for x, y, h in buildings:
        building.append((x, -h))
        building.append((y, h))
    building.sort()
    q = []
    res = []
    pre = 0
    for point, heigth in building:
        if heigth < 0:
            heappush(q, heigth)
        else:
            q.remove(-heigth)

        heapify(q)
        cur = q[0] if q else 0
        if cur != pre:
            temp = []
            temp.append(point)
            temp.append(-cur)
            res.append(temp)
            pre = cur
    return res

# print(getSkyline([[2,9,10],[3,7,15],[5,12,12],[15,20,10],[19,24,8]]))


def fractionToDecimal(numerator: int, denominator: int) -> str:
    hashmap = dict()
    res = ""
    if numerator % denominator == 0:
        return str(numerator // denominator)

    if numerator * denominator < 0:
        res += "-"
    numerator, denominator = abs(numerator), abs(denominator)
    res += str(numerator // denominator) + "."
    numerator %= denominator
    while numerator:
        hashmap[numerator] = len(res)
        numerator *= 10
        res += str(numerator // denominator)
        numerator %= denominator
        if numerator in hashmap:
            return res[:]

    return res


def numMatchingSubseq(s: str, words: List[str]) -> int:
    count = [[] for _ in range(256)]
    for i, char in enumerate(s):
        count[ord(char) - ord('a')].append(i)

    res = 0
    for word in words:
        i = 0
        j = 0
        n = len(word)
        while i < n:
            char = word[i]
            l = count[ord(char) - ord('a')]
            if len(l) == 0:
                break
            pos = bisect_left(l, j)
            if pos == -1 or pos == len(l):
                break
            j = l[pos] + 1
            i += 1
        if i == n:
            res += 1
    return res

# print(numMatchingSubseq("abcde", ["a","bb","acd","ace"]))


def reverseParentheses(s: str) -> str:
    stack = []

    for char in s:
        tmp = []
        if char == ')':
            while stack and stack[-1] != '(':
                letter = stack.pop()
                tmp.append(letter)
            if stack and stack[-1] == '(':
                stack.pop()
            stack += tmp
        else:
            stack.append(char)

    return "".join(stack)

# print(reverseParentheses("(abcd)"))


def finalPrices(prices: List[int]) -> List[int]:
    stack = []
    n = len(prices)
    for i, price in enumerate(prices[::-1]):
        if stack:
            while stack and stack[-1] > price:
                stack.pop()
            if stack:
                prices[n - i - 1] -= stack[-1]
            stack.append(price)
        else:
            stack.append(price)

    return prices

# print(finalPrices([8,4,6,2,3]))


def numberToWords(num: int) -> str:
    s = str(num)[::-1]
    a1 = ["Zero", "One", "Two", "Three", "Four",
          "Five", "Six", "Seven", "Eight", "Nine"]
    a2 = ["Ten", "Eleven", "Twelve", "Thirteen", "Fourteen",
          "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"]
    a3 = ["Twenty", "Thirty", "Forty", "Fifty",
          "Sixty", "Seventy", "Eighty", "Ninety"]
    a4 = ["Hundred", "Thousand", "Million", "Billion"]

    def threeNumberTostr(s: str, i: int) -> str:
        res = ""
        if int(s) == 0:
            return ""
        if i == 1:
            res += '*'
            res += a4[1]
        if i == 2:
            res += '*'
            res += a4[2]
        if i == 3:
            res += '*'
            res += a4[3]
        if len(s) == 1 or s[1] == '0':
            res += '*'
            if s[0] != '0':
                res += a1[int(s[0])]
        if len(s) >= 2 and s[1] == '1':
            res += '*'
            res += a2[int(s[1::-1]) - 10]
        if len(s) >= 2 and int(s[1]) > 1:
            if s[0] == '0':
                res += '*'
                res += a3[int(s[1]) - 2]
            else:
                res += '*'
                res += a1[int(s[0])]
                res += '*'
                res += a3[int(s[1]) - 2]
        if len(s) >= 3:
            res += '*'

            if s[2] != '0':
                res += a4[0]
                res += '*'
                res += a1[int(s[2])]
        return res

    res = ""
    bit = 0
    for i in range(0, len(s), 3):
        j = i + 3 if i + 3 <= len(s) else len(s)
        res += threeNumberTostr(s[i: j], bit)
        bit += 1

    tmp = []
    for e in res.split('*')[::-1]:
        if e:
            tmp.append(e)
    res1 = " ".join(tmp)
    return res1

# print(numberToWords(1001))


class Node:
    def __init__(self) -> None:
        self.children = dict()
        self.isend = False


class StreamChecker:

    def __init__(self, words: List[str]):
        self.root = Node()
        self.sb = ""

        for word in words:
            tmp = self.root
            self.add(word[::-1], tmp)

    def add(self, s, root):
        for char in s:
            if char not in root.children:
                root.children[char] = Node()
            root = root.children[char]
        root.isend = True

    def find(self, l, r, root):
        sb = self.sb
        while r >= l:
            if sb[r] not in root.children:
                return False
            root = root.children[sb[r]]
            if root.isend:
                return True
            r -= 1
        return root.isend

    def query(self, letter: str) -> bool:
        self.sb += letter
        n = len(self.sb)
        l = max(0, n - 200)
        r = n - 1
        root = self.root
        return self.find(l, r, root)

# streamChecker = StreamChecker(["cd", "f", "kl"])
# streamChecker.query("c")
# print(streamChecker.query("d"))


def minAbsoluteSumDiff(nums1: List[int], nums2: List[int]) -> int:
    tmp = nums1.copy()
    mod = 10 ** 9 + 7
    n = len(nums1)
    nums = 0
    for x, y in zip(nums1, nums2):
        nums += abs(x - y)
    nums1.sort()
    res = nums
    for i, num in enumerate(nums2):
        idx = bisect_left(nums1, num)
        if idx == -1:
            res = min(res, nums - abs(num - tmp[i]) + abs(num - nums1[0]))
            continue
        elif idx == n:
            res = min(res, nums - abs(num - tmp[i]) + abs(num - nums1[n - 1]))
            continue

        li = -1
        if idx - 1 >= 0:
            li = idx - 1
        if li != -1:
            if num - nums1[li] < nums1[idx] - num:
                idx = li
        res = min(res, nums - abs(num - tmp[i]) + abs(num - nums1[idx]))

    return res % mod

# print(minAbsoluteSumDiff([38,48,73,55,25,47,45,62,15,34,51,20,76,78,38,91,69,69,73,38,74,75,86,63,73,12,100,59,29,28,94,43,100,2,53,31,73,82,70,94,2,38,50,67,8,40,88,87,62,90,86,33,86,26,84,52,63,80,56,56,56,47,12,50,12,59,52,7,40,16,53,61,76,22,87,75,14,63,96,56,65,16,70,83,51,44,13,14,80,28,82,2,5,57,77,64,58,85,33,24],
# [90,62,8,56,33,22,9,58,29,88,10,66,48,79,44,50,71,2,3,100,88,16,24,28,50,41,65,59,83,79,80,91,1,62,13,37,86,53,43,49,17,82,27,17,10,89,40,82,41,2,48,98,16,43,62,33,72,35,10,24,80,29,49,5,14,38,30,48,93,86,62,23,17,39,40,96,10,75,6,38,1,5,54,91,29,36,62,73,51,92,89,88,74,91,87,34,49,56,33,67]))


class trie:
    def __init__(self, words: List[str]):
        self.root = Node()

        for word in words:
            tmp = self.root
            self.add(word, tmp)

    def add(self, s, root):
        for char in s:
            if char not in root.children:
                root.children[char] = Node()
            root = root.children[char]
        root.isend = True

    def find(self, s, root):
        tmp = None
        for char in s:
            if char in root.children:
                root = root.children[char]
                tmp = root
            else:
                tmp = None
                break
        return tmp


def maxRectangle(words: List[str]) -> List[str]:
    array = [[] for _ in range(101)]
    maxLen = 0
    for word in words:
        maxLen = max(maxLen, len(word))
        array[len(word)].append(word)
    tree = trie(words)

    def dfs(array, tmp, tree, roots, maxLen):

        nonlocal res, v
        if len(array[0]) * maxLen < v:
            return

        for a in array:
            t = []
            isres = 0
            iscontinue = False
            for i, char in enumerate(a):
                root = tree.find(char, roots[i])
                if not root:
                    iscontinue = True
                    break
                t.append(root)
                if root.isend:
                    isres += 1
            if not iscontinue:
                tmp.append(a)
                if len(tmp) >= maxLen:
                    return
                if isres == len(roots):
                    if len(tmp) * len(tmp[0]) >= v:
                        v = len(tmp) * len(tmp[0])
                        res = tmp[:]
                dfs(array, tmp, tree, t, maxLen)
                tmp.pop()
    res = None
    v = 0
    for i in range(101):

        if len(array[i]) > 0:
            dfs(array[i], [], tree, [tree.root] * len(array[i][0]), maxLen)

    return res
# print(maxRectangle(["aa"]))


def levelOrder(root: TreeNode) -> List[int]:
    # 动态数组
    res = []

    def dfs(root, n, res):
        if not root:
            return
        if len(res) == n:
            res.append([root.val])
        else:
            res[n].append(root.val)
        dfs(root.left, n + 1, res)
        dfs(root.right, n + 1, res)
    dfs(root, 0, res)
    return res


def longestNiceSubarray(nums: List[int]) -> int:
    res = 1
    left = 0
    right = 1
    n = len(nums)
    while left < right and right < n:
        tmp = left
        while tmp < right and nums[tmp] & nums[right] == 0:
            tmp += 1
        if tmp == right:
            res = max(res, right - left + 1)
            right += 1
        else:
            if tmp == right - 1:
                left = right
                right += 1
            else:
                left = tmp + 1
    return res

# print(longestNiceSubarray([3,1,5,11,13]))


def kthSmallestPrimeFraction(arr: List[int], k: int):
    q = []
    n = len(arr)
    for i in range(n - 1):
        for j in range(i+1, n):
            if len(q) < k:
                heappush(q, (-arr[i] / arr[j], arr[i], arr[j]))
            elif len(q) == k:
                if arr[i] / arr[j] >= -q[0][0]:
                    continue
                else:
                    heappop(q)
                    heappush(q, (-arr[i] / arr[j], arr[i], arr[j]))
    return (q[0][1], q[0][2])

# print(kthSmallestPrimeFraction([1,7], 1))


def numRollsToTarget(n: int, k: int, target: int) -> int:
    dp = [[0] * (target + 1) for _ in range(n + 1)]
    dp[0][0] = 1
    mod = 10 ** 9 + 7
    for i in range(1, n + 1):
        for j in range(1, target + 1):
            for p in range(1, k + 1):
                if j >= p:
                    dp[i][j] += dp[i-1][j-p]
                    dp[i][j] %= mod

    return dp[n][target]

# print(numRollsToTarget(1,6,3))


class FoodRatings:

    def __init__(self, foods: List[str], cuisines: List[str], ratings: List[int]):
        n = len(foods)
        self.hashmap = defaultdict(SortedList)
        self.foodToCuisine = dict()
        self.foodToRating = dict()
        for i in range(n):
            self.foodToRating[foods[i]] = ratings[i]
            self.foodToCuisine[foods[i]] = cuisines[i]
            self.hashmap[cuisines[i]].add((-ratings[i], foods[i]))

    def changeRating(self, food: str, newRating: int) -> None:
        cuisine = self.foodToCuisine[food]
        rating = self.foodToRating[food]
        l = self.hashmap[cuisine]
        l.discard((-rating, food))
        l.add((-newRating, food))
        self.foodToRating[food] = newRating

    def highestRated(self, cuisine: str) -> str:
        return self.hashmap[cuisine][0][1]


def shortestAlternatingPaths(n: int, redEdges: List[List[int]], blueEdges: List[List[int]]) -> List[int]:
    res = [-1] * n
    res[0] = 0
    graph = defaultdict(list)
    graph1 = defaultdict(list)
    for x, y in redEdges:
        graph[x].append((y, 0))
    for x, y in blueEdges:
        graph1[x].append((y, 1))
    q = deque()
    s = set()
    q.append((0, -1, 0))
    s.add((0, -1))
    while q:
        p, color, cnt = q.popleft()
        cnt += 1
        if color == -1 or color == 1:
            for number, c in graph[p]:
                if (number, c) in s:
                    continue
                if res[number] == -1 or res[number] > cnt:
                    res[number] = cnt
                q.append((number, c, cnt))
                s.add((number, c))

        if color == -1 or color == 0:
            for number, c in graph1[p]:
                if (number, c) in s:
                    continue
                if res[number] == -1 or res[number] > cnt:
                    res[number] = cnt
                q.append((number, c, cnt))
                s.add((number, c))

    return res

# print(shortestAlternatingPaths(5,[[2,2],[0,1],[0,3],[0,0],[0,4],[2,1],[2,0],[1,4],[3,4]],[[1,3],[0,0],[0,3],[4,2],[1,0]]
# ))


def openLock(deadends: List[str], target: str) -> int:
    res = 0
    q = deque()
    s = set()
    q.append(("0000", 0))
    s.add("0000")
    deadends = set(deadends)
    flag = False
    while q:
        if flag:
            break
        p, cnt = q.popleft()
        if p in deadends:
            continue
        if p == target:
            return cnt
        cnt += 1
        for i in range(4):
            l = (int(p[i]) - 1) % 10
            r = (int(p[i]) + 1) % 10
            p1 = p[:i] + str(l) + p[i+1:]
            p2 = p[:i] + str(r) + p[i+1:]
            if p1 == target:
                res = cnt
                flag = True
                break
            if p2 == target:
                flag = True
                res = cnt
                break
            if p1 not in s and p1 not in deadends:
                q.append((p1, cnt))
                s.add(p1)
            if p2 not in s and p2 not in deadends:
                q.append((p2, cnt))
                s.add(p2)

    return res if res != 0 else -1

# print(openLock(["0000"], "8888"))

# print(len("txrvxjnwksqhxuxt"))


def shortestCommonSupersequence(str1: str, str2: str) -> str:
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    ans = []
    i, j = m, n
    while i > 0 and j > 0:
        if dp[i][j] == dp[i-1][j-1] + 1 and str1[i-1] == str2[j-1]:
            ans.append(str1[i-1])
            i -= 1
            j -= 1
        elif dp[i][j] == dp[i-1][j] and str1[i-1] != str2[j-1]:
            ans.append(str1[i-1])
            i -= 1
        elif dp[i][j] == dp[i][j-1] and str1[i-1] != str2[j-1]:
            ans.append(str2[j-1])
            j -= 1
    ans = "".join(ans[::-1])
    if i == 0:
        ans = str2[:j] + ans
    else:
        ans = str1[:i] + ans

    return ans

# "bbabacacaaabc", "abbbaccacaaab"
# print(shortestCommonSupersequence("bcacaaab", "bbabaccc"))


# print(shortestCommonSupersequence("bcacaaab", "bbabaccc") == "bbabaccacaaab")

def maxDiff(num: int) -> int:
    nums = []
    while num > 0:
        nums.append(num % 10)
        num = num // 10
    nums.reverse()
    hashmap = defaultdict(list)
    for i, number in enumerate(nums):
        hashmap[number].append(i)
    big = nums[:]
    low = nums[:]
    for i, number in enumerate(big):
        if number < 9:
            n = big[i]
            for j in hashmap[n]:
                big[j] = 9
            break
    changed = False
    for i, number in enumerate(low):
        if i == 0:
            if number > 1:
                n = low[i]
                for j in hashmap[n]:
                    low[j] = 1
                break
        else:
            if number > 0:
                n = low[i]
                if 0 not in hashmap[n]:
                    changed = True
                    for j in hashmap[n]:
                        low[j] = 0
                if changed:
                    break
    bigNum = 0
    for number in big:
        bigNum *= 10
        bigNum += number
    lowNum = 0
    for number in low:
        lowNum *= 10
        lowNum += number
    return bigNum - lowNum

# 8808050
# print(maxDiff(1101057))



    