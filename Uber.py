# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 17:05:12 2021

@author: prchandr
"""

from collections import Counter,defaultdict
import collections
from heapq import *

"""
49. Group Anagrams

Given an array of strings strs, group the anagrams together. You can return the answer in any order.

An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.

Example 1:

Input: strs = ["eat","tea","tan","ate","nat","bat"]
Output: [["bat"],["nat","tan"],["ate","eat","tea"]]
"""
def groupAnagrams(strs):
    ans = defaultdict(list)
    for s in strs:
        ans[tuple(sorted(s))].append(s)
    return ans.values()


"""
34. Find First and Last Position of Element in Sorted Array

Given an array of integers nums sorted in ascending order, find the starting and ending position of a given target value.

If target is not found in the array, return [-1, -1].

Follow up: Could you write an algorithm with O(log n) runtime complexity?

Example 1:

Input: nums = [5,7,7,8,8,10], target = 8
Output: [3,4]
"""  
def searchRange(nums, target):
    leftidx = searchFinding(nums,target,True)

    if leftidx == len(nums) or nums[leftidx] != target:
        return [-1,-1]

    return (leftidx, searchFinding(nums, target, False)-1)


def searchFinding(nums,target,key):
    lo = 0
    hi = len(nums)
    while lo < hi:
        mid = (lo + hi) // 2
        if nums[mid] > target or (key and nums[mid] == target):
            hi = mid
        else:
            lo = mid + 1
    return lo

"""
23. Merge k Sorted Lists

You are given an array of k linked-lists lists, each linked-list is sorted in ascending order.

Merge all the linked-lists into one sorted linked-list and return it.

Example 1:

Input: lists = [[1,4,5],[1,3,4],[2,6]]
Output: [1,1,2,3,4,4,5,6]
Explanation: The linked-lists are:
[
  1->4->5,
  1->3->4,
  2->6
]
merging them into one sorted list:
1->1->2->3->4->4->5->6
"""
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def mergeKLists(lists):
        nodes = []
        head = point = ListNode(0)
        
        for l in lists:
            while l:
                nodes.append(l.val)
                l = l.next
        
        for x in sorted(nodes):
            point.next = ListNode(x)
            point = point.next
        
        return head.next
    

"""
Basic Calculator
"""
def calculator(s):
    stack = []
    now = ""
    op = "+"
    
    s += "#"
    
    for ch in s:
        
        if ch == " ":
            continue
        elif ch.isdigit():
            now += ch
            continue
        
        if op == "+":
            stack.append(int(now))
        
        elif op == "-":
            stack.append(-int(now))
        
        elif op == "*":
            top = stack.pop()
            stack.append(int(top) * int(now))
            
        elif op == "/":
            top = stack.pop()
            res = 0
            if int(top) < 0 and int(top) % int(now) != 0:
                res = int(top) // int(now) + 1
            else:
                res = int(top) // int(now)
            
            stack.append(res)
        
        now = ""
        op =ch
        
    return sum(stack)

"""
767. Reorganize String

Given a string S, check if the letters can be rearranged so that two characters that are adjacent to each other are not the same.

If possible, output any possible result.  If not possible, return the empty string.

Example 1:

Input: S = "aab"
Output: "aba"
Example 2:

Input: S = "aaab"
Output: ""
"""

def reorganizeString(S):
    # Use counter to count the frequencies. 
    counter = Counter(S)
    maxheap = []
    result = []
    last = None
    # Push counted chars into maxheap. Use negative values to use the heapq as a maxheap. 
    for k, v in  counter.items():    
        heappush(maxheap, [-v, k])
    while maxheap:
        # Pop the most frequent letter.
        item = heappop(maxheap)
        v, k = -item[0], item[1]
        result += k
        # Decrease frequency now that we used it 
        v -= 1
        # This line is the key. If we push it back right after using, it's going to be popped again 
        # in the next iteration resulting in the same letter being adjacent.            
        if last: heappush(maxheap, last)
        # If the count is more than 0, we can pop it back into the heap - in the next iteration
        if v > 0: last = [-v, k] 
        else: last = None
    # If the final string is the same length as input, return it. If now, then we failed.        
    return ''.join(result) if len(result) == len(S) else ""
"""
99. Recover Binary Search Tree

You are given the root of a binary search tree (BST), where exactly two nodes of the tree were swapped by mistake. Recover the tree without changing its structure.

Follow up: A solution using O(n) space is pretty straight forward. Could you devise a constant space solution?

Example 1:

Input: root = [1,3,null,null,2]
Output: [3,1,null,null,2]
Explanation: 3 cannot be a left child of 1 because 3 > 1. Swapping 1 and 3 makes the BST valid.
"""

def recoverTree(root):
    """
    :rtype: void Do not return anything, modify root in-place instead.
    """
    stack = []
    x = y = pred = None
    while stack or root:
        while root:
            stack.append(root)
            root = root.left
        root = stack.pop()
        if pred and root.val < pred.val:
            y = root
            if x is None:
                x = pred 
            else:
                break
        pred = root
        root = root.right

    x.val, y.val = y.val, x.val
    
#Constant Space O(1)
def recoverTreeConstant(root):
    """
    Do not return anything, modify root in-place instead.
    """
    
    def aux(curr, last, res):
		    # traverse by inorder
        if curr.left:
            aux(curr.left, last, res)
        if last[0]:
            if curr.val < last[0].val:
                res.append(last[0])
                res.append(curr)
        last[0] = curr
        if curr.right:
            aux(curr.right, last, res)
    
    res = []
    aux(root, [None], res)
    vals = [x.val for x in res]
    vals.sort()
    for i in range(len(res)):
        res[i].val = vals[i]

"""
22. Generate Parentheses

Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.

Example 1:

Input: n = 3
Output: ["((()))","(()())","(())()","()(())","()()()"]
"""

def generateParenthesis(n):
    """
    :type n: int
    :rtype: List[str]
    """
    output = []
    def backtrack(i, j, ans):
        if i == 0 and j == 0:
            output.append(ans)
            return
        if j >= i:
            if i > 0 :
                backtrack(i-1, j , ans + '(')
            backtrack(i, j-1 , ans + ')')
        return
    
    
    backtrack(n, n, '') 
    return output

"""
4. Median of Two Sorted Arrays

Given two sorted arrays nums1 and nums2 of size m and n respectively, return the median of the two sorted arrays.

Follow up: The overall run time complexity should be O(log (m+n)).

Example 1:

Input: nums1 = [1,3], nums2 = [2]
Output: 2.00000
Explanation: merged array = [1,2,3] and median is 2.
"""

def findMedianSortedArrays(nums1,nums2):
    nums1.extend(nums2)
    nums1.sort()
    
    if len(nums1)%2 == 0:
        l = len(nums1)//2
        r = len(nums1)//2 - 1
        return (nums1[l]+nums1[r])/2
    if len(nums1)%2 == 1:
        mid = len(nums1)//2
        return nums1[mid]
    
"""
253. Meeting Rooms II

Given an array of meeting time intervals consisting of start and end times [[s1,e1],[s2,e2],...] (si < ei), find the minimum number of conference rooms required.

Example 1:

Input: [[0, 30],[5, 10],[15, 20]]
Output: 2
"""

def  minMeetingRooms(intervals):
    time_lookup = defaultdict(int)
    for start,end in intervals:
        time_lookup[start] += 1
        time_lookup[end] -= 1
    print(time_lookup)
    room = 0
    ans = 0
    for time in sorted(time_lookup):
        print(time)
        room += time_lookup[time]
        ans = max(ans,room)
    return ans

"""
340. Longest Substring with At Most K Distinct Characters

Given a string s and an integer k, return the length of the longest substring of s that contains at most k distinct characters.

Example 1:

Input: s = "eceba", k = 2
Output: 3
Explanation: The substring is "ece" with length 3.
"""        
def lengthOfLongestSubstringKDistinct(s,k):
    n = len(s)
    if n * k == 0:
        return 0
    # sliding window left and right pointers
    left, right = 0, 0
    # hashmap character -> its rightmost position
    # in the sliding window
    hashmap = defaultdict()
    max_len = 1
    while right < n:
        # add new character and move right pointer
        hashmap[s[right]] = right
        right += 1
        if len(hashmap) == k + 1:
            # delete the leftmost character
            del_idx = min(hashmap.values())
            del hashmap[s[del_idx]]
            # move left pointer of the slidewindow
            left = del_idx + 1
        max_len = max(max_len, right - left)
    return max_len

"""
7. Reverse Integer

Given a signed 32-bit integer x, return x with its digits reversed. If reversing x causes the value to go outside the signed 32-bit integer range [-231, 231 - 1], then return 0.

Assume the environment does not allow you to store 64-bit integers (signed or unsigned).

Example 1:

Input: x = 123
Output: 321
"""
def reverse(x):
    rev = 0
    n = x
    if x < 0:
        n *= -1
    while n>0:
        rev = (rev*10) + n % 10
        n = n // 10
    if rev > pow(2,31):
        return 0
    if x < 0:
        rev *= -1
    return rev

"""
236. Lowest Common Ancestor of a Binary Tree
"""

def lowestCommonAncestor(root, p, q):
    """
    :type root: TreeNode
    :type p: TreeNode
    :type q: TreeNode
    :rtype: TreeNode
    """
    if root is None:
        return None
    
    if (root == p or root == q):
        return root
    
    left = self.lowestCommonAncestor(root.left, p, q)
    right = self.lowestCommonAncestor(root.right, p, q)
    
    if(left is not None and right is not None):
        return root
    
    if(left is None and right is None):
        return None
    
    if(left is not None):
        return left
    
    if(right is not None):
        return right

"""
354. Russian Doll Envelopes

You have a number of envelopes with widths and heights given as a pair of integers (w, h). One envelope can fit into another if and only if both the width and height of one envelope is greater than the width and height of the other envelope.

What is the maximum number of envelopes can you Russian doll? (put one inside other)

Note:
Rotation is not allowed.

Example:

Input: [[5,4],[6,4],[6,7],[2,3]]
Output: 3 
Explanation: The maximum number of envelopes you can Russian doll is 3 ([2,3] => [5,4] => [6,7]).
"""

def maxEnvelopes(envelopes):
    if not envelopes or len(envelopes) == 0:
        return 0
    
    envelopes = sorted(envelopes, key=lambda x:(x[0], x[1]))
    n = len(envelopes)
    dp = [1 for _ in range(n)]
    maxEnvelopes = 1
    
    for i in range(n):
        for j in range(i+1, n):               
            if envelopes[j][0] > envelopes[i][0] and envelopes[j][1] > envelopes[i][1]:
                dp[j] = max(1 + dp[i], dp[j])
                maxEnvelopes = max(maxEnvelopes, dp[j])
            
    return maxEnvelopes
"""
21. Merge Two Sorted Lists

Merge two sorted linked lists and return it as a sorted list. The list should be made by splicing together the nodes of the first two lists.

Example 1:
    
Input: l1 = [1,2,4], l2 = [1,3,4]
Output: [1,1,2,3,4,4]
"""
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        head = ListNode(-1)
        prev = head
        while l1 and l2:
            if l1.val >= l2.val:
                prev.next = l2
                l2 = l2.next
            else:
                prev.next = l1
                l1 = l1.next
            
            prev = prev.next
        
        prev.next = l1 if l1 is not None else l2
        return head.next

"""
56. Merge Intervals

Given an array of intervals where intervals[i] = [starti, endi], merge all overlapping intervals, and return an array of the non-overlapping intervals that cover all the intervals in the input.

Example 1:

Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
Output: [[1,6],[8,10],[15,18]]
Explanation: Since intervals [1,3] and [2,6] overlaps, merge them into [1,6].
"""

def merge(intervals):
    intervals.sort()
    i = 0
    while i < len(intervals)-1:
        if intervals[i][1] >= intervals[i+1][0]:
            intervals[i][1] = max(intervals[i][1], intervals[i+1][1])
            del intervals[i+1]
        else:
            i += 1
    return intervals

"""
102. Binary Tree Level Order Traversal

Given the root of a binary tree, return the level order traversal of its nodes' values. (i.e., from left to right, level by level).

Example 1:

Input: root = [3,9,20,null,null,15,7]
Output: [[3],[9,20],[15,7]]
"""
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, val):
#         self.val = val
#         self.left = None
#         self.right = None
class Solution(object):
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        
        
        q = deque()
        ans = []
        if root is None:
            return ans
        q.append(root)
        while q:
            n = len(q)
            val = []
            while n > 0:
                node = q.popleft()
                if node.left is not None:
                    q.append(node.left)
                if node.right is not None:
                    q.append(node.right)
                val.append(node.val)
                n -= 1
            ans.append(val)
        return ans
"""
297. Serialize and Deserialize Binary Tree

Serialization is the process of converting a data structure or object into a sequence of bits so that it can be stored in a file or memory buffer, or transmitted across a network connection link to be reconstructed later in the same or another computer environment.

Design an algorithm to serialize and deserialize a binary tree. There is no restriction on how your serialization/deserialization algorithm should work. You just need to ensure that a binary tree can be serialized to a string and this string can be deserialized to the original tree structure.

Clarification: The input/output format is the same as how LeetCode serializes a binary tree. You do not necessarily need to follow this format, so please be creative and come up with different approaches yourself.


Example 1:


Input: root = [1,2,3,null,null,4,5]
Output: [1,2,3,null,null,4,5]
Example 2:
"""  
class Codec:
    def serialize(self, root):
        def serializeUtil(root, s):
            if root is None:
                s += 'null,'
                return s
            s += str(root.val) + ","
            s = serializeUtil(root.left, s) # Important assign s from recursion or it will have only the first one
            s = serializeUtil(root.right, s)
            return s
        
        sr =  serializeUtil(root, "")
        print (sr)
        return sr

    def deserialize(self, data):
        def deserializeUtil(lst):
            if lst[0] == 'null':
                lst.pop(0)
                return None
            val = lst.pop(0)
            root = TreeNode(int(val))
            root.left = deserializeUtil(lst)
            root.right = deserializeUtil(lst)
            return root
        
        lst = data.split(",")
        return deserializeUtil(lst)

"""
655. Print Binary Tree

Print a binary tree in an m*n 2D string array following these rules:

The row number m should be equal to the height of the given binary tree.
The column number n should always be an odd number.
The root node's value (in string format) should be put in the exactly middle of the first row it can be put. The column and the row where the root node belongs will separate the rest space into two parts (left-bottom part and right-bottom part). You should print the left subtree in the left-bottom part and print the right subtree in the right-bottom part. The left-bottom part and the right-bottom part should have the same size. Even if one subtree is none while the other is not, you don't need to print anything for the none subtree but still need to leave the space as large as that for the other subtree. However, if two subtrees are none, then you don't need to leave space for both of them.
Each unused space should contain an empty string "".
Print the subtrees following the same rules.
Example 1:
Input:
     1
    /
   2
Output:
[["", "1", ""],
 ["2", "", ""]]
"""   
def printTree(root):
    if not root:
        return [[""]]
    def getLevel(node):
        q = [node]
        step = 0
        while q:
            sz = len(q)
            for i in range(sz):
                cur = q.pop(0)
                if cur.left:
                    q.append(cur.left)
                if cur.right:
                    q.append(cur.right)
            step += 1
        return step
    m = getLevel(root)
    n = 2**m - 1
    res = [[""]*n for _ in range(m)]
    i = 0
    j = n//2
    def dfs(node, i, j_start, j_end):
        mid = (j_start+j_end)//2
        res[i][mid] = str(node.val)
        if node.left:
            dfs(node.left, i+1, j_start, mid-1)
        if node.right:
            dfs(node.right, i+1, mid+1, j_end)
    dfs(root, i, 0, n-1)
    return res

"""
133. Clone Graph

Given a reference of a node in a connected undirected graph.

Return a deep copy (clone) of the graph.

Each node in the graph contains a val (int) and a list (List[Node]) of its neighbors.

class Node {
    public int val;
    public List<Node> neighbors;
}
 

Test case format:

For simplicity sake, each node's value is the same as the node's index (1-indexed). For example, the first node with val = 1, the second node with val = 2, and so on. The graph is represented in the test case using an adjacency list.
"""
class Node(object):
    def __init__(self, val, neighbors):
        self.val = val
        self.neighbors = neighbors

from collections import deque
class Solution(object):

    def cloneGraph(self, node):
        """
        :type node: Node
        :rtype: Node
        """
        if not node:
            return node

        # Dictionary to save the visited node and it's respective clone
        # as key and value respectively. This helps to avoid cycles.
        visited = {}

        # Put the first node in the queue
        queue = deque([node])
        # Clone the node and put it in the visited dictionary.
        visited[node] = Node(node.val, [])
        # Start BFS traversal
        while queue:
            # Pop a node say "n" from the from the front of the queue.
            n = queue.popleft()
            # Iterate through all the neighbors of the node
            for neighbor in n.neighbors:
                if neighbor not in visited:
                    # Clone the neighbor and put in the visited, if not present already
                    visited[neighbor] = Node(neighbor.val, [])
                    # Add the newly encountered node to the queue.
                    queue.append(neighbor)
                # Add the clone of the neighbor to the neighbors of the clone node "n".
                visited[n].neighbors.append(visited[neighbor])

        # Return the clone of the node from visited.
        return visited[node]

"""
529. Minesweeper

Let's play the minesweeper game (Wikipedia, online game)!

You are given a 2D char matrix representing the game board. 'M' represents an unrevealed mine, 'E' represents an unrevealed empty square, 'B' represents a revealed blank square that has no adjacent (above, below, left, right, and all 4 diagonals) mines, digit ('1' to '8') represents how many mines are adjacent to this revealed square, and finally 'X' represents a revealed mine.

Now given the next click position (row and column indices) among all the unrevealed squares ('M' or 'E'), return the board after revealing this position according to the following rules:

If a mine ('M') is revealed, then the game is over - change it to 'X'.
If an empty square ('E') with no adjacent mines is revealed, then change it to revealed blank ('B') and all of its adjacent unrevealed squares should be revealed recursively.
If an empty square ('E') with at least one adjacent mine is revealed, then change it to a digit ('1' to '8') representing the number of adjacent mines.
Return the board when no more squares will be revealed.
 

Example 1:

Input: 

[['E', 'E', 'E', 'E', 'E'],
 ['E', 'E', 'M', 'E', 'E'],
 ['E', 'E', 'E', 'E', 'E'],
 ['E', 'E', 'E', 'E', 'E']]

Click : [3,0]

Output: 

[['B', '1', 'E', '1', 'B'],
 ['B', '1', 'M', '1', 'B'],
 ['B', '1', '1', '1', 'B'],
 ['B', 'B', 'B', 'B', 'B']]

Explanation:

Example 2:

Input: 

[['B', '1', 'E', '1', 'B'],
 ['B', '1', 'M', '1', 'B'],
 ['B', '1', '1', '1', 'B'],
 ['B', 'B', 'B', 'B', 'B']]

Click : [1,2]

Output: 

[['B', '1', 'E', '1', 'B'],
 ['B', '1', 'X', '1', 'B'],
 ['B', '1', '1', '1', 'B'],
 ['B', 'B', 'B', 'B', 'B']]

Explanation:

"""

class Solution(object):
    def updateBoard(self, board, click):
        """
        :type board: List[List[str]]
        :type click: List[int]
        :rtype: List[List[str]]
        """
        i, j = click[0], click[1]
        self.adj = [[1,0], [-1,0], [0,1], [0,-1], [-1,-1], [-1,1], [1,1], [1,-1]]
        visited = [[False]*len(board[0]) for _ in range(len(board))]
        if board[i][j] == 'M':
            board[i][j] = 'X'
            return board
        if board[i][j] == 'E':
            self.emptySquare(board, i, j, visited)
        return board
            
    def emptySquare(self, board, i, j, visited):
        flag = False
        cnt = 0
        visited[i][j] = True
        for a in self.adj:
            x, y = i+a[0], j+a[1]
            if 0 <= x < len(board) and 0 <= y < len(board[0]) and visited[x][y] == False and board[x][y] == 'M':
                flag = True
                cnt += 1
        if flag == True:
            board[i][j] = str(cnt)
        else:
            board[i][j] = 'B'
            for a in self.adj:
                x, y = i+a[0], j+a[1]
                if 0 <= x < len(board) and 0 <= y < len(board[0]) and visited[x][y] == False and board[x][y] == 'E':
                    self.emptySquare(board, x, y, visited)

"""
53. Maximum Subarray

Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.

Example 1:

Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
Output: 6
Explanation: [4,-1,2,1] has the largest sum = 6.
"""                    

def maxSubArray(nums):
    n = len(nums)
    curr_sum = max_sum = nums[0]

    for i in range(1, n):
        curr_sum = max(nums[i], curr_sum + nums[i])
        max_sum = max(max_sum, curr_sum)
        
    return max_sum

"""
121. Best Time to Buy and Sell Stock

You are given an array prices where prices[i] is the price of a given stock on the ith day.

You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.

Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0.

Example 1:

Input: prices = [7,1,5,3,6,4]
Output: 5
Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
Note that buying on day 2 and selling on day 1 is not allowed because you must buy before you sell.
"""
def maxProfit(prices):
    i = 0
    j = 1
    profit = 0
    while i <len(prices) and j<len(prices):
        if prices[j] - prices[i] < 0:
            i = j
        else:
            if prices[j]-prices[i] >= profit:
                profit = prices[j] - prices[i]
        j += 1
    return profit

"""
122. Best Time to Buy and Sell Stock II

You are given an array prices for which the ith element is the price of a given stock on day i.

Find the maximum profit you can achieve. You may complete as many transactions as you like (i.e., buy one and sell one share of the stock multiple times).

Note: You may not engage in multiple transactions simultaneously (i.e., you must sell the stock before you buy again).

Example 1:

Input: prices = [7,1,5,3,6,4]
Output: 7
Explanation: Buy on day 2 (price = 1) and sell on day 3 (price = 5), profit = 5-1 = 4.
Then buy on day 4 (price = 3) and sell on day 5 (price = 6), profit = 6-3 = 3.
"""
def maxProfit1(self, prices):
    """
    :type prices: List[int]
    :rtype: int
    """
    maxprofit = 0
    for i in range(1,len(prices)): # we start at 1 because we are going to compare with previous price value.
        if (prices[i] > prices[i-1]): # for the first iteration, price[1] > price[0] will be checked.
            maxprofit += prices[i] - prices[i-1] 
    return maxprofit

"""
13. Roman to Integer

Roman numerals are represented by seven different symbols: I, V, X, L, C, D and M.

Symbol       Value
I             1
V             5
X             10
L             50
C             100
D             500
M             1000
For example, 2 is written as II in Roman numeral, just two one's added together. 12 is written as XII, which is simply X + II. The number 27 is written as XXVII, which is XX + V + II.

Roman numerals are usually written largest to smallest from left to right. However, the numeral for four is not IIII. Instead, the number four is written as IV. Because the one is before the five we subtract it making four. The same principle applies to the number nine, which is written as IX. There are six instances where subtraction is used:

I can be placed before V (5) and X (10) to make 4 and 9. 
X can be placed before L (50) and C (100) to make 40 and 90. 
C can be placed before D (500) and M (1000) to make 400 and 900.
Given a roman numeral, convert it to an integer.

 

Example 1:

Input: s = "III"
Output: 3
"""
def romanToInt(s):
    values = {"I": 1,"V": 5,"X": 10,"L": 50,"C": 100,"D": 500,"M": 1000}
    total = 0
    i = 0
    while i < len(s):
        if i+1 < len(s) and values[s[i]] < values[s[i+1]]:
            total += values[s[i+1]] - values[s[i]]
            i += 2
        else:
            total += values[s[i]]
            i += 1
    return total

"""
41. First Missing Positive

Given an unsorted integer array nums, find the smallest missing positive integer.

Example 1:

Input: nums = [1,2,0]
Output: 3
Example 2:

Input: nums = [3,4,-1,1]
Output: 2
"""
def firstMissingPositive(nums):
    # Swap numbers until 1 is at nums[0], 2 is at nums[1], 3 is at nums[2], and so on
    # Ignoring numbers that are not positive or would otherwise be out of bounds of the array
    for i in range(len(nums)):
        while nums[i] > 0 and nums[i] < len(nums) and nums[i] != i+1:
            print(i,nums)
            if nums[nums[i]-1] == nums[i]:
                print("dup",i)
                # Turn duplicates into a number we can ignore
                nums[i] = -1
                continue
            n = nums[i]
            nums[i], nums[n-1] = nums[n-1], nums[i]
            print("nums",nums)
    # Search for a missing positive integer. The first one we encounter is the answer.
    # If we don't enocounter one, then the array looked like [1, 2, ... n] and so the answer is n+1
    for i in range(len(nums)):
        if nums[i] != i+1:
            return i+1
    return len(nums) + 1

"""
3. Longest Substring Without Repeating Characters

Given a string s, find the length of the longest substring without repeating characters.

Example 1:

Input: s = "abcabcbb"
Output: 3
Explanation: The answer is "abc", with the length of 3.
"""

def lengthOfLongestSubstring(s):
    str_list = []
    max_length = 0
    for x in s:
        if x in str_list:
            str_list = str_list[str_list.index(x)+1:]
        str_list.append(x)    
        max_length = max(max_length, len(str_list))
    
    return max_length

"""
2. Add Two Numbers

You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.

Example 1:

Input: l1 = [2,4,3], l2 = [5,6,4]
Output: [7,0,8]
Explanation: 342 + 465 = 807.
"""
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        result = ListNode(0)
        prev = result
        carry = 0
        
        while l1 or l2 or carry:
            val1 = (l1.val if l1 else 0)
            val2 = (l2.val if l2 else 0)
            TSum = (val1 + val2+carry)%10
            carry = (val1 + val2+carry)//10
            prev.next = ListNode(TSum)
            prev = prev.next
            
            l1 = (l1.next if l1 else None)
            l2 = (l2.next if l2 else None)
        
        return result.next
    
"""
445. Add Two Numbers II

You are given two non-empty linked lists representing two non-negative integers. The most significant digit comes first and each of their nodes contain a single digit. Add the two numbers and return it as a linked list.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.

Follow up:
What if you cannot modify the input lists? In other words, reversing the lists is not allowed.

Example:

Input: (7 -> 2 -> 4 -> 3) + (5 -> 6 -> 4)
Output: 7 -> 8 -> 0 -> 7
"""
class LinkedListAddition:
    def reverseList(self, head):
        last = None
        while head:
            # keep the next node
            tmp = head.next
            # reverse the link
            head.next = last
            # update the last node and the current node
            last = head
            head = tmp
        
        return last
    
    def addTwoNumbers(self, l1, l2):
        # reverse lists
        l1 = self.reverseList(l1)
        l2 = self.reverseList(l2)
        
        head = None
        carry = 0
        while l1 or l2:
            # get the current values 
            x1 = l1.val if l1 else 0
            x2 = l2.val if l2 else 0
            # current sum and carry
            val = (carry + x1 + x2) % 10
            carry = (carry + x1 + x2) // 10
            # update the result: add to front
            curr = ListNode(val)
            curr.next = head
            head = curr
            # move to the next elements in the lists
            l1 = l1.next if l1 else None
            l2 = l2.next if l2 else None
        if carry:
            curr = ListNode(carry)
            curr.next = head
            head = curr
        return head
"""
628. Maximum Product of Three Numbers

Given an integer array nums, find three numbers whose product is maximum and return the maximum product.

Example 1:

Input: nums = [1,2,3]
Output: 6
"""

def maximumProduct(nums):
    
    nums.sort()
    pre = nums[0] * nums[1] * nums[-1]
    latter = nums[-1] * nums[-2] * nums[-3]
    return max(pre, latter)

"""
69. Sqrt(x)

Given a non-negative integer x, compute and return the square root of x.

Since the return type is an integer, the decimal digits are truncated, and only the integer part of the result is returned.

 

Example 1:

Input: x = 4
Output: 2
Example 2:

Input: x = 8
Output: 2
Explanation: The square root of 8 is 2.82842..., and since the decimal part is truncated, 2 is returned.
"""

def mySqrt(x):
    left, right = 0, x
    
    while left <= right:
        mid = (left + right) // 2
        mid_sq = mid * mid
        
        if mid_sq > x:
            right = mid - 1
        elif mid_sq < x:
            left = mid + 1
        else:
            return mid
    
    return left - 1
"""
78. Subsets

Given an integer array nums of unique elements, return all possible subsets (the power set).

The solution set must not contain duplicate subsets. Return the solution in any order.

Example 1:

Input: nums = [1,2,3]
Output: [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
"""
def subsets(nums):   
    subset = []
    subsets = []
    def subsetsUtil(nums, subset, start):
        subsets.append(subset[:])
        for i in range(start, len(nums)):
            subset.append(nums[i])
            subsetsUtil(nums, subset, i+1)
            subset.pop()
    subsetsUtil(nums, subset, 0)
    print (subsets)
    return subsets
"""
283. Move Zeroes

Given an array nums, write a function to move all 0's to the end of it while maintaining the relative order of the non-zero elements.

Example:

Input: [0,1,0,3,12]
Output: [1,3,12,0,0]
"""

def moveZeroes(nums):
    nonzeroidx = 0
    for i in range(len(nums)):
        if nums[i] != 0:
            nums[nonzeroidx], nums[i] = nums[i], nums[nonzeroidx]
            nonzeroidx += 1
    return nums
"""
39. Combination Sum

Given an array of distinct integers candidates and a target integer target, return a list of all unique combinations of candidates where the chosen numbers sum to target. You may return the combinations in any order.

The same number may be chosen from candidates an unlimited number of times. Two combinations are unique if the frequency of at least one of the chosen numbers is different.

It is guaranteed that the number of unique combinations that sum up to target is less than 150 combinations for the given input.

Example 1:

Input: candidates = [2,3,6,7], target = 7
Output: [[2,2,3],[7]]
Explanation:
2 and 3 are candidates, and 2 + 2 + 3 = 7. Note that 2 can be used multiple times.
7 is a candidate, and 7 = 7.
These are the only two combinations.
"""
def combinationSum(candidates, target):
    result = []
    def backtrack(remain,comb,start):
        if remain == 0:
            result.append(list(comb))
            return
        if remain < 0:
            return
        for i in range(start, len(candidates)):
            comb.append(candidates[i])
            backtrack(remain-candidates[i],comb,i)
            comb.pop()
    
    backtrack(target,[],0)
    return result
"""
518. Coin Change 2

You are given coins of different denominations and a total amount of money. Write a function to compute the number of combinations that make up that amount. You may assume that you have infinite number of each kind of coin.

Example 1:

Input: amount = 5, coins = [1, 2, 5]
Output: 4
Explanation: there are four ways to make up the amount:
5=5
5=2+2+1
5=2+1+1+1
5=1+1+1+1+1
"""

def change(amount,coins):
    dp = [0] * (amount + 1)
    dp[0] = 1
    for coin in coins:
        for x in range(coin, amount + 1):
            dp[x] += dp[x - coin]
    return dp[amount]
"""
315. Count of Smaller Numbers After Self

You are given an integer array nums and you have to return a new counts array. The counts array has the property where counts[i] is the number of smaller elements to the right of nums[i].
Example 1:

Input: nums = [5,2,6,1]
Output: [2,1,1,0]
Explanation:
To the right of 5 there are 2 smaller elements (2 and 1).
To the right of 2 there is only 1 smaller element (1).
To the right of 6 there is 1 smaller element (1).
To the right of 1 there is 0 smaller element.
"""
class Solution:
    def countSmaller(self,nums):
        counts = []
        sorted_nums = []

        for number in reversed(nums):
            count = self._search_insert_position(sorted_nums, number)
            counts.append(count)
            sorted_nums.insert(count, number)
        return counts[::-1]

    def _search_insert_position(self, sorted_nums, target):
        # return the last position nums[i] < target, nums[i + 1] >= target
        # return 0 if target is the smallest or sorted_nums is empty
        if not sorted_nums or len(sorted_nums) == 0:
            return 0
        start, end = 0, len(sorted_nums) - 1
        while start + 1 < end:
            mid = (start + end) // 2
            if sorted_nums[mid] == target:
                end = mid
            elif sorted_nums[mid] < target:
                start = mid
            else:
                # sorted_nums[mid][1] > target:
                end = mid
        if sorted_nums[end] < target:
            return end + 1
        if sorted_nums[start] < target:
            return start + 1
        return 0
"""
n denote the length of the input array

Time:
for loop each element O(n)
binary search for the insert position O(log(n))
Overall O(nlog(n))
Space:
O(2n) -> O(n)
"""
"""
31. Next Permutation

Implement next permutation, which rearranges numbers into the lexicographically next greater permutation of numbers.

If such an arrangement is not possible, it must rearrange it as the lowest possible order (i.e., sorted in ascending order).

The replacement must be in place and use only constant extra memory.

Example 1:
                    
Input: nums = [1,2,3]
Output: [1,3,2]
Example 2:

Input: nums = [3,2,1]
Output: [1,2,3]
"""

def nextPermutation(nums):
    def reverse(L, start, end):
        while start < end:
            L[start], L[end] = L[end], L[start]
            start, end = start + 1, end - 1
    i, n = len(nums) - 1, len(nums)
    while i >= 1 and nums[i] <= nums[i-1]:
        i -= 1
    if i != 0:
        j = i
        while j + 1 < n and nums[j+1] > nums[i - 1]:
            j += 1
        nums[i-1], nums[j] = nums[j], nums[i-1]
    reverse(nums, i, n - 1)
    return nums

"""
332. Reconstruct Itinerary

Given a list of airline tickets represented by pairs of departure and arrival airports [from, to], reconstruct the itinerary in order. All of the tickets belong to a man who departs from JFK. Thus, the itinerary must begin with JFK.

Note:

If there are multiple valid itineraries, you should return the itinerary that has the smallest lexical order when read as a single string. For example, the itinerary ["JFK", "LGA"] has a smaller lexical order than ["JFK", "LGB"].
All airports are represented by three capital letters (IATA code).
You may assume all tickets form at least one valid itinerary.
One must use all the tickets once and only once.
Example 1:

Input: [["MUC", "LHR"], ["JFK", "MUC"], ["SFO", "SJC"], ["LHR", "SFO"]]
Output: ["JFK", "MUC", "LHR", "SFO", "SJC"]
"""

def findItinerary(tickets):
    graph = collections.defaultdict(list)
    for x,y in tickets:
        graph[x].append(y)
    for city in graph: # Check if needed to do
        graph[city] = sorted(graph[city])
    ans = []
    stack = ['JFK']
    while stack:
        top = stack[-1]
        if not graph[top]:
            ans.append(top)
            stack.pop()
        else:
            stack.append(graph[top][0])
            del graph[top][0]
    return ans[::-1]
"""
415. Add Strings

Given two non-negative integers num1 and num2 represented as string, return the sum of num1 and num2.
"""
def addStrings(num1, num2):
    n1, n2 = len(num1), len(num2)
    if n1 > n2:
        num2 = '0' * abs(n1-n2) +num2
    else:
        num1 = '0' * abs(n2-n1) + num1
    print(num1,num2)
    res, carry = '', 0
    for i,j in zip(num1[::-1], num2[::-1]):
        res = str((int(i)+int(j)+carry) % 10) + res
        carry = (int(i) + int(j) + carry) // 10
    
    return res if not carry else str(carry)+res
    
"""
84. Largest Rectangle in Histogram

Given n non-negative integers representing the histogram's bar height where the width of each bar is 1, find the area of largest rectangle in the histogram.

Above is a histogram where width of each bar is 1, given height = [2,1,5,6,2,3].

The largest rectangle is shown in the shaded area, which has area = 10 unit.

Example:

Input: [2,1,5,6,2,3]
Output: 10
"""
def largestRectangleArea(heights):
    stack = [-1]
    max_area = 0
    for i in range(len(heights)):
        while stack[-1] != -1 and heights[stack[-1]] >= heights[i]:
            current_height = heights[stack.pop()]
            current_width = i - stack[-1] - 1
            max_area = max(max_area, current_height * current_width)
            print(i,current_height,current_width,max_area,stack)
        stack.append(i)

    while stack[-1] != -1:
        current_height = heights[stack.pop()]
        current_width = len(heights) - stack[-1] - 1
        max_area = max(max_area, current_height * current_width)
    return max_area

"""
234. Palindrome Linked List

Given a singly linked list, determine if it is a palindrome.

Example 1:

Input: 1->2
Output: false
Example 2:

Input: 1->2->2->1
Output: true
"""
def isPalindrome(head):
    vals = []
    current_node = head
    while current_node is not None:
        vals.append(current_node.val)
        current_node = current_node.next
    return vals == vals[::-1]

"""
134. Gas Station

There are n gas stations along a circular route, where the amount of gas at the ith station is gas[i].

You have a car with an unlimited gas tank and it costs cost[i] of gas to travel from the ith station to its next (i + 1)th station. You begin the journey with an empty tank at one of the gas stations.

Given two integer arrays gas and cost, return the starting gas station's index if you can travel around the circuit once in the clockwise direction, otherwise return -1. If there exists a solution, it is guaranteed to be unique

Example 1:

Input: gas = [1,2,3,4,5], cost = [3,4,5,1,2]
Output: 3
Explanation:
Start at station 3 (index 3) and fill up with 4 unit of gas. Your tank = 0 + 4 = 4
Travel to station 4. Your tank = 4 - 1 + 5 = 8
Travel to station 0. Your tank = 8 - 2 + 1 = 7
Travel to station 1. Your tank = 7 - 3 + 2 = 6
Travel to station 2. Your tank = 6 - 4 + 3 = 5
Travel to station 3. The cost is 5. Your gas is just enough to travel back to station 3.
Therefore, return 3 as the starting index.
"""

def canCompleteCircuit(gas, cost):
    """
    :type gas: List[int]
    :type cost: List[int]
    :rtype: int
    """
    n = len(gas)
    
    total_tank, curr_tank = 0, 0
    starting_station = 0
    for i in range(n):
        total_tank += gas[i] - cost[i]
        curr_tank += gas[i] - cost[i]
        # If one couldn't get here,
        if curr_tank < 0:
            # Pick up the next station as the starting one.
            starting_station = i + 1
            # Start with an empty tank.
            curr_tank = 0
    
    return starting_station if total_tank >= 0 else -1
"""
Implement a thread safe bounded blocking queue that has the following methods:

BoundedBlockingQueue(int capacity) The constructor initializes the queue with a maximum capacity.
void enqueue(int element) Adds an element to the front of the queue. If the queue is full, the calling thread is blocked until the queue is no longer full.
int dequeue() Returns the element at the rear of the queue and removes it. If the queue is empty, the calling thread is blocked until the queue is no longer empty.
int size() Returns the number of elements currently in the queue.
Your implementation will be tested using multiple threads at the same time. Each thread will either be a producer thread that only makes calls to the enqueue method or a consumer thread that only makes calls to the dequeue method. The size method will be called after every test case.

Please do not use built-in implementations of bounded blocking queue as this will not be accepted in an interview.

 

Example 1:

Input:
1
1
["BoundedBlockingQueue","enqueue","dequeue","dequeue","enqueue","enqueue","enqueue","enqueue","dequeue"]
[[2],[1],[],[],[0],[2],[3],[4],[]]

Output:
[1,0,2,2]
"""

import threading
class BoundedBlockingQueue(object):

    def __init__(self, capacity: int):
        self.cv = threading.Condition()
        self.q = deque()
        self.cap = capacity

    def enqueue(self, element: int) -> None:
        with self.cv:
            while self.cap == len(self.q):
                self.cv.wait()
            self.q.append(element)
            self.cv.notify()

    def dequeue(self) -> int:
        ans = 0
        with self.cv:
            while not self.q:
                self.cv.wait()
            ans = self.q.popleft()
            self.cv.notify()
        return ans
        
    def size(self) -> int:
        return len(self.q)
    
"""
1004. Max Consecutive Ones III

Given an array A of 0s and 1s, we may change up to K values from 0 to 1.

Return the length of the longest (contiguous) subarray that contains only 1s. 

Example 1:

Input: A = [1,1,1,0,0,0,1,1,1,1,0], K = 2
Output: 6
Explanation: 
[1,1,1,0,0,1,1,1,1,1,1]
Bolded numbers were flipped from 0 to 1.  The longest subarray is underlined.
Example 2:

Input: A = [0,0,1,1,0,0,1,1,1,0,1,1,0,0,0,1,1,1,1], K = 3
Output: 10
Explanation: 
[0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1]
Bolded numbers were flipped from 0 to 1.  The longest subarray is underlined.
"""   
def longestOnes(A, K):
    left = 0
    for right in range(len(A)):
        # If we included a zero in the window we reduce the value of K.
        # Since K is the maximum zeros allowed in a window.
        K -= 1 - A[right]
        # A negative K denotes we have consumed all allowed flips and window has
        # more than allowed zeros, thus increment left pointer by 1 to keep the window size same.
        if K < 0:
            # If  the left element to be thrown out is zero we increase K.
            K += 1 - A[left]
            left += 1
    return right - left + 1


"""
206. Reverse Linked List
Easy

6377

122

Add to List

Share
Given the head of a singly linked list, reverse the list, and return the reversed list.

 

Example 1:


Input: head = [1,2,3,4,5]
Output: [5,4,3,2,1]
"""

class Node:
 
    # Constructor to initialize the node object
    def __init__(self, data):
        self.data = data
        self.next = None
 
class LinkedList:
 
    # Function to initialize head
    def __init__(self):
        self.head = None
 
    # Function to reverse the linked list
    def reverse(self):
        prev = None
        current = self.head
        while(current is not None):
            next = current.next
            current.next = prev
            prev = current
            current = next
        self.head = prev
 
    # Function to insert a new node at the beginning
    def push(self, new_data):
        new_node = Node(new_data)
        new_node.next = self.head
        self.head = new_node
 
    # Utility function to print the linked LinkedList
    def printList(self):
        res = []
        temp = self.head
        while(temp):
            res.append(temp.data)
            temp = temp.next
        return res
 
 
# Driver code
llist = LinkedList()
llist.push(20)
llist.push(4)
llist.push(15)
llist.push(85)
 
print ("Given Linked List")
print(llist.printList())
llist.reverse()
print ("\nReversed Linked List")
print(llist.printList())

"""
678. Valid Parenthesis String

Given a string s containing only three types of characters: '(', ')' and '*', return true if s is valid.

The following rules define a valid string:

Any left parenthesis '(' must have a corresponding right parenthesis ')'.
Any right parenthesis ')' must have a corresponding left parenthesis '('.
Left parenthesis '(' must go before the corresponding right parenthesis ')'.
'*' could be treated as a single right parenthesis ')' or a single left parenthesis '(' or an empty string "".
 

Example 1:

Input: s = "()"
Output: true
Example 2:

Input: s = "(*)"
Output: true
"""
def checkValidString(s):
	
    stars = []
    brackets = []

    for i,char in enumerate(s):
        if char == '(':
            brackets.append(i)
        elif char == ')':
            if brackets != [] and brackets[len(brackets) - 1] < i :
                brackets.pop()
            elif stars  != [] and stars[len(stars) - 1] < i:
                stars.pop()
            else:
                return False
        elif char == "*":
            stars.append(i)
    
    while(len(brackets) > 0 and len(stars) > 0 and stars[len(stars) - 1] > brackets[len(brackets) - 1]):
        stars.pop()
        brackets.pop()
    
    if brackets != []:
        return False
    
    return True
"""
757. Set Intersection Size At Least Two

An integer interval [a, b] (for integers a < b) is a set of all consecutive integers from a to b, including a and b.

Find the minimum size of a set S such that for every integer interval A in intervals, the intersection of S with A has a size of at least two.

Example 1:

Input: intervals = [[1,3],[1,4],[2,5],[3,5]]
Output: 3
Explanation: Consider the set S = {2, 3, 4}.  For each interval, there are at least 2 elements from S in the interval.
Also, there isn't a smaller size set that fulfills the above condition.
Thus, we output the size of this set, which is 3.
"""
def intersectionSizeTwo(intervals):
    intervals.sort(key = lambda x: (x[1], -x[0]))
    print(intervals)
    answer = start = end = 0
    for b, a in intervals:
        print(start,end,a,b)
        if start == end == 0 or end < b:
            answer += 2
            start = a - 1
            end = a
            print("Inside: If",start,end,a,b)
        elif start < b:
            print("Inside Elif",start,end,a,b)
            answer += 1
            start = end
            end = a
    return answer

"""
427. Construct Quad Tree

Given a n * n matrix grid of 0's and 1's only. We want to represent the grid with a Quad-Tree.

Return the root of the Quad-Tree representing the grid.

Notice that you can assign the value of a node to True or False when isLeaf is False, and both are accepted in the answer.

A Quad-Tree is a tree data structure in which each internal node has exactly four children. Besides, each node has two attributes:

val: True if the node represents a grid of 1's or False if the node represents a grid of 0's. 
isLeaf: True if the node is leaf node on the tree or False if the node has the four children.
"""


# Definition for a QuadTree node.
class Node:
    def __init__(self, val, isLeaf, topLeft, topRight, bottomLeft, bottomRight):
        self.val = val
        self.isLeaf = isLeaf
        self.topLeft = topLeft
        self.topRight = topRight
        self.bottomLeft = bottomLeft
        self.bottomRight = bottomRight

class QuadTree:

    def construct(self,grid):
        
        '''
        Make a helper function helper(x, y, size) where x,y is coord of top left. 
        
        If size is 1, set val to grid[x][y], isLeaf true, and children to None.
        
        Create node for all children.
        
        If all children are leaves with same value, set same value, set isLeaf to True and children to None. 
        Otherwise value can be whatever, set isLeaf false.
        
        topleft : helper(x, y, size // 2)
        topright : helper(x, y + size // 2, size // 2)
        bottomleft: helper(x + size // 2, y, size // 2)
        bottomright: helper(x + size // 2, y + size // 2, size // 2)
        
        '''
        
        def helper(x, y, size):
            if size == 1:
                return Node(grid[x][y], True, None, None, None, None)
            half = size // 2
            topLeft = helper(x, y, half)
            topRight = helper(x, y + half, half)
            bottomLeft = helper(x + half, y, half)
            bottomRight = helper(x + half, y + half, half)
            
            children = [topLeft, topRight, bottomLeft, bottomRight]
            allLeaves = all(c.isLeaf for c in children)
            allMatch = all(c.val == topLeft.val for c in children)
            isLeaf = allLeaves and allMatch
            if isLeaf:
                return Node(topLeft.val, True, None, None, None, None)
            else:
                return Node(1, False, topLeft, topRight, bottomLeft, bottomRight)
        
        node =  helper(0, 0, len(grid))
        return node
        #return serialize(node,[])
        #return serHelper(node,[])
        
    def serialize_tree(self,root_node):
        """ Given a tree root node (some object with a 'data' attribute
            and a 'children' attribute which is a list of child nodes),
            serialize it to a list, each element of which is either a
            pair (data, has_children_flag), or None (which signals an
            end of a sibling chain).
        """
        lst = []
        def serialize_aux(node):
            # Recursive visitor function
            if node.isLeaf:
                # The node has children, so:
                #  1. add it to the list & mark that it has children
                #  2. recursively serialize its children
                #  3. finally add a null entry to signal that the children
                #     of this node have ended
                lst.append((node.val, True))

            else:
                # The node is child-less, so simply add it to
                # the list & mark that it has no chilren
                lst.append((node.val, False))
                while node.isLeaf == True:
                    serialize_aux(node)
                lst.append((node.topLeft.val,True))
                lst.append((node.topRight.val,True))
                lst.append((node.bottomLeft.val,True))
                lst.append((node.bottomRight.val,True))
        serialize_aux(root_node)
        return lst
            

s = QuadTree()
root = s.construct([[0,1],[1,0]])
print("Quad Tree Node",s.serialize_tree(root))
"""
 Complexity: Time O(n^2), Space O(1)
"""

"""
934. Shortest Bridge

In a given 2D binary array A, there are two islands.  (An island is a 4-directionally connected group of 1s not connected to any other 1s.)

Now, we may change 0s to 1s so as to connect the two islands together to form 1 island.

Return the smallest number of 0s that must be flipped.  (It is guaranteed that the answer is at least 1.)

Example 1:

Input: A = [[0,1],[1,0]]
Output: 1
"""

def shortestBridge(A):
    m, n = len(A), len(A[0])
    i, j = next((i, j) for i in range(m) for j in range(n) if A[i][j])
    print(i,j)
    # dfs 
    stack = [(i, j)]
    seen = set(stack)
    while stack: 
        i, j = stack.pop()
        seen.add((i, j)) # mark as visited 
        for ii, jj in (i-1, j), (i, j-1), (i, j+1), (i+1, j): 
            if 0 <= ii < m and 0 <= jj < n and A[ii][jj] and (ii, jj) not in seen: 
                stack.append((ii, jj))
                seen.add((ii, jj))
    
    # bfs 
    ans = 0
    queue = list(seen)
    while queue:
        newq = []
        for i, j in queue: 
            for ii, jj in (i-1, j), (i, j-1), (i, j+1), (i+1, j): 
                if 0 <= ii < m and 0 <= jj < n and (ii, jj) not in seen: 
                    if A[ii][jj] == 1: return ans 
                    newq.append((ii, jj))
                    seen.add((ii, jj))
        queue = newq
        ans += 1
"""
[0,1]
[1,0]
Analysis
Time complexity O(N)
Space complexity O(N)

"""

"""
130. Surrounded Regions

Given an m x n matrix board containing 'X' and 'O', capture all regions surrounded by 'X'.

A region is captured by flipping all 'O's into 'X's in that surrounded region.

Example 1:

Input: board = [["X","X","X","X"],["X","O","O","X"],["X","X","O","X"],["X","O","X","X"]]
Output: [["X","X","X","X"],["X","X","X","X"],["X","X","X","X"],["X","O","X","X"]]
Explanation: Surrounded regions should not be on the border, which means that any 'O' on the border of the board are not flipped to 'X'. Any 'O' that is not on the border and it is not connected to an 'O' on the border will be flipped to 'X'. Two cells are connected if they are adjacent cells connected horizontally or vertically.
"""
class SurroundedRegions:
    def solve(self,board):
        """
        Do not return anything, modify board in-place instead.
        """
        if not board or not board[0]:
            return
    
        self.ROWS = len(board)
        self.COLS = len(board[0])
        print("Rows AND Cols",self.ROWS,self.COLS)
    
        # Step 1). retrieve all border cells
        from itertools import product
        borders = list(product(range(self.ROWS), [0, self.COLS-1])) \
                + list(product([0, self.ROWS-1], range(self.COLS)))
        print(borders)
        # Step 2). mark the "escaped" cells, with any placeholder, e.g. 'E'
        for row, col in borders:
            self.DFS(board, row, col)
        # Step 3). flip the captured cells ('O'->'X') and the escaped one ('E'->'O')
        for r in range(self.ROWS):
            for c in range(self.COLS):
                #print("Current Value", board[r][c])
                if board[r][c] == 'O':   board[r][c] = 'X'  # captured
                elif board[r][c] == 'E': board[r][c] = 'O'  # escaped
        print(board)
    def DFS(self,board, row, col):
        if board[row][col] != 'O':
            return
        board[row][col] = 'E'
        if col < self.COLS-1: self.DFS(board, row, col+1)
        if row < self.ROWS-1: self.DFS(board, row+1, col)
        if col > 0: self.DFS(board, row, col-1)
        if row > 0: self.DFS(board, row-1, col)
        
"""
322. Coin Change

You are given coins of different denominations and a total amount of money amount. Write a function to compute the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return -1.

You may assume that you have an infinite number of each kind of coin.

 

Example 1:

Input: coins = [1,2,5], amount = 11
Output: 3
Explanation: 11 = 5 + 5 + 1
"""     
def coinChange(coins, amount):
    '''
    BFS solution
    '''
    if amount == 0:
        return 0
    
    seen = set()
    
    from collections import deque
    q = deque([[0,0]])
    
    while q:
        curr,level = q.popleft()
        
        if level != 0 and curr == amount:
            return level
        
        for c in coins:
            if curr + c not in seen and curr + c <= amount:
                q.append([curr + c, level + 1])
                seen.add(curr + c)
    return -1
"""
85. Maximal Rectangle

Given a rows x cols binary matrix filled with 0's and 1's, find the largest rectangle containing only 1's and return its area.

Example 1:

Input: matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
Output: 6
Explanation: The maximal rectangle is shown in the above picture.
"""
def maximalRectangle(matrix):
    maxarea = 0
    dp = [[0] * len(matrix[0]) for _ in range(len(matrix))]
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == '0': continue
            # compute the maximum width and update dp with it
            width = dp[i][j] = dp[i][j-1] + 1 if j else 1
            # compute the maximum area rectangle with a lower right corner at [i, j]
            for k in range(i, -1, -1):
                width = min(width, dp[k][j])
                maxarea = max(maxarea, width * (i-k+1))
    return maxarea
"""
986. Interval List Intersections

You are given two lists of closed intervals, firstList and secondList, where firstList[i] = [starti, endi] and secondList[j] = [startj, endj]. Each list of intervals is pairwise disjoint and in sorted order.

Return the intersection of these two interval lists.

A closed interval [a, b] (with a < b) denotes the set of real numbers x with a <= x <= b.

The intersection of two closed intervals is a set of real numbers that are either empty or represented as a closed interval. For example, the intersection of [1, 3] and [2, 4] is [2, 3].

Example 1:

Input: firstList = [[0,2],[5,10],[13,23],[24,25]], secondList = [[1,5],[8,12],[15,24],[25,26]]
Output: [[1,2],[5,5],[8,10],[15,23],[24,24],[25,25]]
"""
def intervalIntersection(A, B):
    ans = []
    i = j = 0

    while i < len(A) and j < len(B):
        # Let's check if A[i] intersects B[j].
        # lo - the startpoint of the intersection
        # hi - the endpoint of the intersection
        lo = max(A[i][0], B[j][0])
        hi = min(A[i][1], B[j][1])
        if lo <= hi:
            ans.append([lo, hi])

        # Remove the interval with the smallest endpoint
        if A[i][1] < B[j][1]:
            i += 1
        else:
            j += 1

    return ans

"""
646. Maximum Length of Pair Chain

You are given n pairs of numbers. In every pair, the first number is always smaller than the second number.

Now, we define a pair (c, d) can follow another pair (a, b) if and only if b < c. Chain of pairs can be formed in this fashion.

Given a set of pairs, find the length longest chain which can be formed. You needn't use up all the given pairs. You can select pairs in any order.

Example 1:
Input: [[1,2], [2,3], [3,4]]
Output: 2
Explanation: The longest chain is [1,2] -> [3,4]
"""
def findLongestChain(pairs):
    pairs.sort()
    dp = [1] * len(pairs)

    for j in range(len(pairs)):
        for i in range(j):
            if pairs[i][1] < pairs[j][0]:
                dp[j] = max(dp[j], dp[i] + 1)

    return max(dp)

"""
977. Squares of a Sorted Array

Given an integer array nums sorted in non-decreasing order, return an array of the squares of each number sorted in non-decreasing order.

Example 1:

Input: nums = [-4,-1,0,3,10]
Output: [0,1,9,16,100]
Explanation: After squaring, the array becomes [16,1,0,9,100].
After sorting, it becomes [0,1,9,16,100].
Example 2:

Input: nums = [-7,-3,2,3,11]
Output: [4,9,9,49,121]
"""
def sortedSquares(nums):
    n = len(nums)
    result = [0] * n
    left = 0
    right = n - 1
    for i in range(n - 1, -1, -1):
        if abs(nums[left]) < abs(nums[right]):
            square = nums[right]
            right -= 1
        else:
            square = nums[left]
            left += 1
        result[i] = square * square
    return result

"""
773. Sliding Puzzle

On a 2x3 board, there are 5 tiles represented by the integers 1 through 5, and an empty square represented by 0.

A move consists of choosing 0 and a 4-directionally adjacent number and swapping it.

The state of the board is solved if and only if the board is [[1,2,3],[4,5,0]].

Given a puzzle board, return the least number of moves required so that the state of the board is solved. If it is impossible for the state of the board to be solved, return -1.

Examples:

Input: board = [[1,2,3],[4,0,5]]
Output: 1
Explanation: Swap the 0 and the 5 in one move.
Input: board = [[1,2,3],[5,4,0]]
Output: -1
Explanation: No number of moves will make the board solved.
"""
def slidingPuzzle(board):
    
    # make tuple of the full board  as 1 D and save it in vistied
    # save index of 0 in the q in order to get the neighbors to be swapped with
    # do usual bfs until we hit the target state [1,2,3][4,5,0]. If we never reach this target state just return -1 else return level of bfs
    
    rlen = len(board)
    clen = len(board[0])
    
    src = tuple(board[i][j] for i in range(rlen) for j in range(clen))
    
    target = (1,2,3,4,5,0)
    q = deque()
    visited = set()
    q.append((src,src.index(0),0))
    visited.add(src)
    dirs = [-1,1,-clen,clen] # -1,1 give neighbors to left and right, -clen,clen give neighbor top and bottom in a 1d array (which is actually a representation of 2d array)
    while q:
        state, idx0, level = q.popleft()
        # check if target state
        if state == target:
            return level
        for d in dirs:
            nextpos = idx0+d
            # compute row,col of current index of 0 and next index for validity
            idx0row = idx0//3
            nextposrow = nextpos//3
            idx0col = idx0%3
            nextposcol = nextpos%3
            # if not either the same row or col, then they must be diagonally opposite - example index 2 and 3 - they are neighbors 1 distant but not adjacent!
            if idx0row!=nextposrow and idx0col!=nextposcol:
                continue
            if nextpos<0 or nextpos>= (rlen*clen): # If out of bounds - let's say 0 is in second tuple but there is no neighbor below
                continue
            # since tuples are immutable, convert to list, swap 0 position and convert to tuple
            nextstate = list(state)
            nextstate[idx0],nextstate[nextpos] = nextstate[nextpos],nextstate[idx0]
            ntup = tuple(nextstate)
            if ntup not in visited:
                q.append((ntup,nextpos,level+1))
                visited.add(ntup)
    return -1
"""
981. Time Based Key-Value Store

Create a timebased key-value store class TimeMap, that supports two operations.

1. set(string key, string value, int timestamp)

Stores the key and value, along with the given timestamp.
2. get(string key, int timestamp)

Returns a value such that set(key, value, timestamp_prev) was called previously, with timestamp_prev <= timestamp.
If there are multiple such values, it returns the one with the largest timestamp_prev.
If there are no values, it returns the empty string ("").

Example 1:

Input: inputs = ["TimeMap","set","get","get","set","get","get"], inputs = [[],["foo","bar",1],["foo",1],["foo",3],["foo","bar2",4],["foo",4],["foo",5]]
Output: [null,null,"bar","bar",null,"bar2","bar2"]
Explanation:   
TimeMap kv;   
kv.set("foo", "bar", 1); // store the key "foo" and value "bar" along with timestamp = 1   
kv.get("foo", 1);  // output "bar"   
kv.get("foo", 3); // output "bar" since there is no value corresponding to foo at timestamp 3 and timestamp 2, then the only value is at timestamp 1 ie "bar"   
kv.set("foo", "bar2", 4);   
kv.get("foo", 4); // output "bar2"   
kv.get("foo", 5); //output "bar2"   
"""
class TimeMap:

    def __init__(self):
        self.hashmap = defaultdict(list)
        

    def set(self, key: str, value: str, timestamp: int) -> None:
        self.hashmap[key].append((timestamp, value))

    def get(self, key: str, timestamp: int) -> str:
        return self.find_val(key, timestamp)
    
    def find_val(self, key, timestamp):
        lo = 0
        hi = len(self.hashmap[key]) - 1
        ans_ind = -1
        while(lo <= hi):
            mid = lo + (hi - lo) // 2
            if self.hashmap[key][mid][0] <= timestamp:
                ans_ind = mid
                lo = mid + 1
            else:
                hi = mid - 1
        return self.hashmap[key][ans_ind][1] if ans_ind != -1 else ""
    
"""
1640. Check Array Formation Through Concatenation

You are given an array of distinct integers arr and an array of integer arrays pieces, where the integers in pieces are distinct. Your goal is to form arr by concatenating the arrays in pieces in any order. However, you are not allowed to reorder the integers in each array pieces[i].

Return true if it is possible to form the array arr from pieces. Otherwise, return false.

Example 1:

Input: arr = [85], pieces = [[85]]
Output: true
"""
def canFormArray(arr, pieces):
    n = len(arr)
    # initialize hashmap
    mapping = {p[0]: p for p in pieces}

    i = 0
    while i < n:
        # find target piece
        if arr[i] not in mapping:
            return False
        # check target piece
        target_piece = mapping[arr[i]]
        for x in target_piece:
            if x != arr[i]:
                return False
            i += 1

    return True

"""
1027. Longest Arithmetic Subsequence

Given an array A of integers, return the length of the longest arithmetic subsequence in A.

Recall that a subsequence of A is a list A[i_1], A[i_2], ..., A[i_k] with 0 <= i_1 < i_2 < ... < i_k <= A.length - 1, and that a sequence B is arithmetic if B[i+1] - B[i] are all the same value (for 0 <= i < B.length - 1).

Example 1:

Input: A = [3,6,9,12]
Output: 4
Explanation: 
The whole array is an arithmetic sequence with steps of length = 3.
"""
def longestArithSeqLength(A):
    mem = dict()
    for i in range(len(A)):
        for j in range(i+1, len(A)):
            d = A[j] - A[i]
            if (i,d) in mem:
                mem[j,d] = mem[i,d] + 1
            else:
                mem[j,d] = 2
    
    return max(mem.values())

"""
452. Minimum Number of Arrows to Burst Balloons

There are some spherical balloons spread in two-dimensional space. For each balloon, provided input is the start and end coordinates of the horizontal diameter. Since it's horizontal, y-coordinates don't matter, and hence the x-coordinates of start and end of the diameter suffice. The start is always smaller than the end.

An arrow can be shot up exactly vertically from different points along the x-axis. A balloon with xstart and xend bursts by an arrow shot at x if xstart ≤ x ≤ xend. There is no limit to the number of arrows that can be shot. An arrow once shot keeps traveling up infinitely.

Given an array points where points[i] = [xstart, xend], return the minimum number of arrows that must be shot to burst all balloons.

Example 1:

Input: points = [[10,16],[2,8],[1,6],[7,12]]
Output: 2
Explanation: One way is to shoot one arrow for example at x = 6 (bursting the balloons [2,8] and [1,6]) and another arrow at x = 11 (bursting the other two balloons).
"""
def findMinArrowShots(points):
    if len(points)==0:
        return 0
    if len(points)==1:
        return 1
    count=0
    points.sort()
    print(points)
    a,b=points[0][0],points[0][1]
    for i in range(1,len(points)):
        if b>=points[i][0]:
            b=min(b,points[i][1])
        else:
            count+=1
            a=points[i][0]
            b=points[i][1]
    return count+1

"""
1281. Subtract the Product and Sum of Digits of an Integer

Given an integer number n, return the difference between the product of its digits and the sum of its digits.

Example 1:

Input: n = 234
Output: 15 
Explanation: 
Product of digits = 2 * 3 * 4 = 24 
Sum of digits = 2 + 3 + 4 = 9 
Result = 24 - 9 = 15

"""
def subtractProductAndSum(n):
    sum = 0
    prod = 1
    for i in str(n):
        prod *= int(i)
        sum += int(i)
    
    return prod-sum
        
"""
Suppose Andy and Doris want to choose a restaurant for dinner, and they both have a list of favorite restaurants represented by strings.

You need to help them find out their common interest with the least list index sum. If there is a choice tie between answers, output all of them with no order requirement. You could assume there always exists an answer.

Example 1:

Input: list1 = ["Shogun","Tapioca Express","Burger King","KFC"], list2 = ["Piatti","The Grill at Torrey Pines","Hungry Hunter Steakhouse","Shogun"]
Output: ["Shogun"]
Explanation: The only restaurant they both like is "Shogun".

"""

def findRestaurant(list1, list2):
    set1, set2 = set(list1),set(list2)
    common = list(set1.intersection(set2))
    if len(common) == 1:
        return common
    else:
        d = {}
        for i in common:
            d[i] = list1.index(i) + list2.index(i)
        print(d)
        d = {k:v for k,v in sorted(d.items(),key = lambda x:x[1])}
        print(d)
        arr = []
        for i in d:
            print(i)
            if len(arr)==0:
                arr.append(i)
            else:
                if d[arr[-1]] == d[i]:
                    arr.append(i)
        return arr

"""
716. Max Stack

Design a max stack data structure that supports the stack operations and supports finding the stack's maximum element.

Implement the MaxStack class:

MaxStack() Initializes the stack object.
void push(int x) Pushes element x onto the stack.
int pop() Removes the element on top of the stack and returns it.
int top() Gets the element on the top of the stack without removing it.
int peekMax() Retrieves the maximum element in the stack without removing it.
int popMax() Retrieves the maximum element in the stack and removes it. If there is more than one maximum element, only remove the top-most one.
 

Example 1:

Input
["MaxStack", "push", "push", "push", "top", "popMax", "top", "peekMax", "pop", "top"]
[[], [5], [1], [5], [], [], [], [], [], []]
Output
[null, null, null, null, 5, 5, 1, 5, 1, 5]

Explanation
MaxStack stk = new MaxStack();
stk.push(5);   // [5] the top of the stack and the maximum number is 5.
stk.push(1);   // [5, 1] the top of the stack is 1, but the maximum is 5.
stk.push(5);   // [5, 1, 5] the top of the stack is 5, which is also the maximum, because it is the top most one.
stk.top();     // return 5, [5, 1, 5] the stack did not change.
stk.popMax();  // return 5, [5, 1] the stack is changed now, and the top is different from the max.
stk.top();     // return 1, [5, 1] the stack did not change.
stk.peekMax(); // return 5, [5, 1] the stack did not change.
stk.pop();     // return 1, [5] the top of the stack and the max element is now 5.
stk.top();     // return 5, [5] the stack did not change.
"""

class MaxStack:
    
    #data = []
    #c_max = -float('inf')

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.data = []
        self.c_max = -float(inf)

    def push(self, x: int) -> None:
        '''
        Keep track of current max
        '''
        
        if(x > self.c_max):
            self.c_max = x
            
        self.data.append(x)

    def pop(self) -> int:
        
        removed = self.data.pop()
        '''
        Find the new max after pop operation
        '''
        n_c_max = -float(inf)
        for i in self.data:
            if(i > n_c_max):
                n_c_max = i
                
        self.c_max = n_c_max
        
        return removed

    def top(self) -> int:
        
        return self.data[len(self.data)-1]

    def peekMax(self) -> int:
        
        return self.c_max
        

    def popMax(self) -> int:
        
        result = self.c_max
        '''
        Remove the last max element
        '''
        for i in range(len(self.data)-1, -1, -1):
            if(self.data[i] == self.c_max):
                self.data.pop(i)
                break
        '''
        Find the new max in case the only max element was removed
        '''
        
        n_c_max = -float(inf)
        for i in self.data:
            if(i > n_c_max):
                n_c_max = i
                
        self.c_max = n_c_max
        
        return result
    
"""
1135. Connecting Cities With Minimum Cost

There are N cities numbered from 1 to N.

You are given connections, where each connections[i] = [city1, city2, cost] represents the cost to connect city1 and city2 together.  (A connection is bidirectional: connecting city1 and city2 is the same as connecting city2 and city1.)

Return the minimum cost so that for every pair of cities, there exists a path of connections (possibly of length 1) that connects those two cities together.  The cost is the sum of the connection costs used. If the task is impossible, return -1.

Example 1:

Input: N = 3, connections = [[1,2,5],[1,3,6],[2,3,1]]
Output: 6
Explanation: 
Choosing any 2 edges will connect all cities so we choose the minimum 2.
"""
import heapq
def minimumCost(N, connections):
    edges=len(connections)  
    if edges < N-1:    #Edge case if edges are less than the vertices
        return -1
    heaplist=[]  
    parents={}
    def getParent(node):    #Helper function to get the parents
        parent = parents[node]
        while node!=parent:
            node=parent
            parent=parents[parent]
        return node     
    for node in range(N+1):   #Initialize the parent map, initially the node's parent is itself
        parents[node]=node
    for vert1, vert2 , cost in connections: #Maintain a min heap (You can sort the list as well) to have the minimum edge node always on top of the heap
        heapq.heappush(heaplist,[cost,vert1,vert2])
    min_cost=0
    edge_count=0
    while heaplist:
        #Pop the min edge 
        cost, parent, child = heapq.heappop(heaplist)    
        #Get the parents of the vertices of parent and child
        p1=getParent(parent)
        p2=getParent(child)    
        #Check if the parents are not same
        if p1 != p2:
            parents[p2] = p1  #Update the parent of the child
            min_cost+=cost
            edge_count+=1        
            #End Case: If the edge count is N-1
            if edge_count == N-1:
                return min_cost
    return -1   

"""
1136. Parallel Courses

You are given an integer n which indicates that we have n courses, labeled from 1 to n. You are also given an array relations where relations[i] = [a, b], representing a prerequisite relationship between course a and course b: course a has to be studied before course b.

In one semester, you can study any number of courses as long as you have studied all the prerequisites for the course you are studying.

Return the minimum number of semesters needed to study all courses. If there is no way to study all the courses, return -1.

Example 1:

Input: n = 3, relations = [[1,3],[2,3]]
Output: 2
Explanation: In the first semester, courses 1 and 2 are studied. In the second semester, course 3 is studied.
"""
def minimumSemester(n,relations):
    graph = {i:[] for i in range(1,n+1)}
    for start_node,end_node in relations:
        graph[start_node].append(end_node)
    visited = {}
    def dfs(node):
        if node in visited:
            return visited[node]
        else:
            visited[node] = -1
        maxLen = 1
        for end_node in graph[node]:
            length = dfs(end_node)
            if length == -1:
                return -1
            else:
                maxLen = max(length+1,maxLen)
        
        visited[node] = maxLen
        return maxLen
    maxLength = -1
    for node in graph.key():
        length = dfs(node)
        if length == -1:
            return -1
        else:
            maxLength = max(maxLength,length)
    return maxLength
#Time Complexity = O(N+E)
#Space = O(N+E)

"""
1400. Construct K Palindrome Strings

Given a string s and an integer k. You should construct k non-empty palindrome strings using all the characters in s.

Return True if you can use all the characters in s to construct k palindrome strings or False otherwise.

Example 1:

Input: s = "annabelle", k = 2
Output: true
Explanation: You can construct two palindromes using all characters in s.
Some possible constructions "anna" + "elble", "anbna" + "elle", "anellena" + "b"
""" 
from collections import Counter
def canConstruct(s, k):
    if len(s) < k:
        return False
    freq = Counter(s)
    count = 0
    for char in freq.values():
        if char % 2 != 0:
            count += 1
    if count > k:
        return False
    else:
        return True

"""
871. Minimum Number of Refueling Stops

A car travels from a starting position to a destination which is target miles east of the starting position.

Along the way, there are gas stations.  Each station[i] represents a gas station that is station[i][0] miles east of the starting position, and has station[i][1] liters of gas.

The car starts with an infinite tank of gas, which initially has startFuel liters of fuel in it.  It uses 1 liter of gas per 1 mile that it drives.

When the car reaches a gas station, it may stop and refuel, transferring all the gas from the station into the car.

What is the least number of refueling stops the car must make in order to reach its destination?  If it cannot reach the destination, return -1.

Note that if the car reaches a gas station with 0 fuel left, the car can still refuel there.  If the car reaches the destination with 0 fuel left, it is still considered to have arrived.

Example 1:

Input: target = 1, startFuel = 1, stations = []
Output: 0
Explanation: We can reach the target without refueling.
"""
def minRefuelStops(target, tank, stations):
    pq = []  # A maxheap is simulated using negative values
    stations.append((target, float('inf')))

    ans = prev = 0
    for location, capacity in stations:
        tank -= location - prev
        while pq and tank < 0:  # must refuel in past
            tank += -heapq.heappop(pq)
            ans += 1
        if tank < 0: return -1
        heapq.heappush(pq, -capacity)
        prev = location

    return ans

"""
48. Rotate Image

You are given an n x n 2D matrix representing an image, rotate the image by 90 degrees (clockwise).

You have to rotate the image in-place, which means you have to modify the input 2D matrix directly. DO NOT allocate another 2D matrix and do the rotation.

Example 1:

Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
Output: [[7,4,1],[8,5,2],[9,6,3]]
"""
def rotate(arr):
    visited = set()
    for i in range(0, len(arr)):
        for j in range (0, len(arr)):
            if i != j and (i,j) not in visited: 
                temp = arr[i][j]
                arr[i][j] = arr[j][i]
                arr[j][i] = temp
                visited.add((j,i))
    for i in range(len(arr)):
            arr[i] = arr[i][::-1]

"""
1352. Product of the Last K Numbers

Implement the class ProductOfNumbers that supports two methods:

1. add(int num)

Adds the number num to the back of the current list of numbers.
2. getProduct(int k)

Returns the product of the last k numbers in the current list.
You can assume that always the current list has at least k numbers.
At any time, the product of any contiguous sequence of numbers will fit into a single 32-bit integer without overflowing.

Example:

Input
["ProductOfNumbers","add","add","add","add","add","getProduct","getProduct","getProduct","add","getProduct"]
[[],[3],[0],[2],[5],[4],[2],[3],[4],[8],[2]]

Output
[null,null,null,null,null,null,20,40,0,null,32]

Explanation
ProductOfNumbers productOfNumbers = new ProductOfNumbers();
productOfNumbers.add(3);        // [3]
productOfNumbers.add(0);        // [3,0]
productOfNumbers.add(2);        // [3,0,2]
productOfNumbers.add(5);        // [3,0,2,5]
productOfNumbers.add(4);        // [3,0,2,5,4]
productOfNumbers.getProduct(2); // return 20. The product of the last 2 numbers is 5 * 4 = 20
productOfNumbers.getProduct(3); // return 40. The product of the last 3 numbers is 2 * 5 * 4 = 40
productOfNumbers.getProduct(4); // return 0. The product of the last 4 numbers is 0 * 2 * 5 * 4 = 0
productOfNumbers.add(8);        // [3,0,2,5,4,8]
productOfNumbers.getProduct(2); // return 32. The product of the last 2 numbers is 4 * 8 = 32 
"""
class ProductOfNumbers:

    def __init__(self):
        self.nums = [1]
        self.z = -1
        

    def add(self, num):
        l = len(self.nums)
        if num == 0:
            self.z = l
            self.nums.append(1)
        else:
            self.nums.append(num*self.nums[-1])

    def getProduct(self, k):
        l = len(self.nums)
        a = l-k
        if self.z >= a:
            return 0
        else:
            return self.nums[-1]//(1 if a==0 else self.nums[a-1])
"""
1254. Number of Closed Islands

Given a 2D grid consists of 0s (land) and 1s (water).  An island is a maximal 4-directionally connected group of 0s and a closed island is an island totally (all left, top, right, bottom) surrounded by 1s.

Return the number of closed islands.

Example 1:

Input: grid = [[1,1,1,1,1,1,1,0],[1,0,0,0,0,1,1,0],[1,0,1,0,1,1,1,0],[1,0,0,0,0,1,0,1],[1,1,1,1,1,1,1,0]]
Output: 2
Explanation: 
Islands in gray are closed because they are completely surrounded by water (group of 1s).
"""          
class Islands:
    def closedIsland(self, grid):
    
        row = len(grid)
        col = len(grid[0])
        visited = [[False for _ in range(col)] for _ in range(row)]
    
        total_islands = 0
        for i in range(1, row - 1):
            for j in range(1, col - 1):
                if grid[i][j] == 0 and not visited[i][j]:
                    if self.dfs(grid, i, j, row, col, visited):
                        total_islands += 1
        return total_islands
    
    def dfs(self, grid, i, j, row, col, visited):
    
        if i < 0 or j < 0 or i >= row or j >= col:
            return False
    
        if grid[i][j] == 1 or visited[i][j]:
            return True
    
        visited[i][j] = True
    
        left  = self.dfs(grid, i, j - 1, row, col, visited)
        right = self.dfs(grid, i, j + 1, row, col, visited)
        up    = self.dfs(grid, i - 1, j, row, col, visited)
        down  = self.dfs(grid, i + 1, j, row, col, visited)
    
        return left and right and up and down

"""
1368. Minimum Cost to Make at Least One Valid Path in a Grid

Given a m x n grid. Each cell of the grid has a sign pointing to the next cell you should visit if you are currently in this cell. The sign of grid[i][j] can be:
1 which means go to the cell to the right. (i.e go from grid[i][j] to grid[i][j + 1])
2 which means go to the cell to the left. (i.e go from grid[i][j] to grid[i][j - 1])
3 which means go to the lower cell. (i.e go from grid[i][j] to grid[i + 1][j])
4 which means go to the upper cell. (i.e go from grid[i][j] to grid[i - 1][j])
Notice that there could be some invalid signs on the cells of the grid which points outside the grid.

You will initially start at the upper left cell (0,0). A valid path in the grid is a path which starts from the upper left cell (0,0) and ends at the bottom-right cell (m - 1, n - 1) following the signs on the grid. The valid path doesn't have to be the shortest.

You can modify the sign on a cell with cost = 1. You can modify the sign on a cell one time only.

Return the minimum cost to make the grid have at least one valid path.

Example 1:

Input: grid = [[1,1,1,1],[2,2,2,2],[1,1,1,1],[2,2,2,2]]
Output: 3
Explanation: You will start at point (0, 0).
The path to (3, 3) is as follows. (0, 0) --> (0, 1) --> (0, 2) --> (0, 3) change the arrow to down with cost = 1 --> (1, 3) --> (1, 2) --> (1, 1) --> (1, 0) change the arrow to down with cost = 1 --> (2, 0) --> (2, 1) --> (2, 2) --> (2, 3) change the arrow to down with cost = 1 --> (3, 3)
The total cost = 3.
"""
def minCost(grid):
    # hops, x, y
    que = deque([(0,0,0)])
    seen = {(0,0)}
    N = len(grid)
    M = len(grid[0])
    nxtque = deque()
    nxtseen = set()
    while que:
        hop, x, y = que.popleft()
        if x == N-1 and y == M-1: return hop
        for i, (xx, yy) in enumerate([(x,y+1),(x,y-1),(x+1,y),(x-1,y)]):
            if 0 <= xx < N and 0 <= yy < M:
                if (xx,yy) not in seen:
                    if grid[x][y] - 1 == i:
                        que.append((hop,xx,yy))
                        seen.add((xx,yy))
                    else:
                        if (xx,yy) not in nxtseen:
                            nxtque.append((hop+1,xx,yy))
                            nxtseen.add((xx,yy))
        if not que:
            que = nxtque
            nxtque = deque()
            seen.update(nxtseen)
            nxtseen= set()

"""
454. 4Sum II

Given four lists A, B, C, D of integer values, compute how many tuples (i, j, k, l) there are such that A[i] + B[j] + C[k] + D[l] is zero.

To make problem a bit easier, all A, B, C, D have same length of N where 0 ≤ N ≤ 500. All integers are in the range of -228 to 228 - 1 and the result is guaranteed to be at most 231 - 1.

Example:

Input:
A = [ 1, 2]
B = [-2,-1]
C = [-1, 2]
D = [ 0, 2]

Output:
2
Explanation:
The two tuples are:
1. (0, 0, 0, 1) -> A[0] + B[0] + C[0] + D[1] = 1 + (-2) + (-1) + 2 = 0
2. (1, 1, 0, 0) -> A[1] + B[1] + C[0] + D[0] = 2 + (-1) + (-1) + 0 = 0
"""
def fourSumCount(A, B, C, D):
    count=0
    seen={}
    
    for a in A:
        for b in B:
            if(a+b in seen):
                seen[a+b]+=1
            else:
                seen[a+b]=1
                
    for c in C:
        for d in D:
            cd = (c+d)*-1
            if(cd in seen):
                count+=seen[cd]    
    return count
"""
1002. Find Common Characters

Given an array A of strings made only from lowercase letters, return a list of all characters that show up in all strings within the list (including duplicates).  For example, if a character occurs 3 times in all strings but not 4 times, you need to include that character three times in the final answer.

You may return the answer in any order.

Example 1:

Input: ["bella","label","roller"]
Output: ["e","l","l"]
"""
def commonChars(A):
    myDict = {}
    stringList = A[0]
    for i in stringList:
        if i in myDict:
            myDict[i] += 1
        else:
            myDict[i] = 1
    lenA = len(A)
    for key in myDict.keys():
        for i in range(1,lenA):
            n = A[i].count(key)
            if myDict[key] > n:
                myDict[key] = n
    
    res = []
    for key in myDict.keys():
        value = myDict[key]
        if value > 0:
            res += [key] * value
    
    return res



"""
1200. Minimum Absolute Difference

Given an array of distinct integers arr, find all pairs of elements with the minimum absolute difference of any two elements. 

Return a list of pairs in ascending order(with respect to pairs), each pair [a, b] follows

a, b are from arr
a < b
b - a equals to the minimum absolute difference of any two elements in arr
 
Example 1:

Input: arr = [4,2,1,3]
Output: [[1,2],[2,3],[3,4]]
Explanation: The minimum absolute difference is 1. List all pairs with difference equal to 1 in ascending order.
"""
def minimumAbsDifference(arr):
    x = arr[:]
    x.sort()
    ans = []
    minDiff = float('inf')
    for i in range(len(x)-1):
        temp = x[i+1] - x[i]
        if temp < minDiff:
            minDiff = temp
    
    for i in range(len(x)-1):
        temp = x[i+1]-x[i]
        if temp == minDiff:
            ans.append([x[i],x[i+1]])
    
    return ans

"""
1626. Best Team With No Conflicts

You are the manager of a basketball team. For the upcoming tournament, you want to choose the team with the highest overall score. The score of the team is the sum of scores of all the players in the team.

However, the basketball team is not allowed to have conflicts. A conflict exists if a younger player has a strictly higher score than an older player. A conflict does not occur between players of the same age.

Given two lists, scores and ages, where each scores[i] and ages[i] represents the score and age of the ith player, respectively, return the highest overall score of all possible basketball teams.

Example 1:

Input: scores = [1,3,5,10,15], ages = [1,2,3,4,5]
Output: 34
Explanation: You can choose all the players.
"""
def bestTeamScore(scores, ages):
    # Making a zipped list 
    adj=list(zip(ages,scores))
    # Sorting on the basis of age as by default first parameter
    adj.sort()
    # Making a dp list and storing that individual score as it can always be the answer
    dp=[-1 for i in scores]
    for i in range(len(scores)):
        dp[i]=adj[i][1]
        for j in range(i):
            if(adj[i][1]>=adj[j][1]):   # If the score of the ith age (which surely is large then jth age) as sorted  is greater or equal
                dp[i]=max(dp[i],dp[j]+adj[i][1])
    return max(dp)

"""
1052. Grumpy Bookstore Owner

Today, the bookstore owner has a store open for customers.length minutes.  Every minute, some number of customers (customers[i]) enter the store, and all those customers leave after the end of that minute.

On some minutes, the bookstore owner is grumpy.  If the bookstore owner is grumpy on the i-th minute, grumpy[i] = 1, otherwise grumpy[i] = 0.  When the bookstore owner is grumpy, the customers of that minute are not satisfied, otherwise they are satisfied.

The bookstore owner knows a secret technique to keep themselves not grumpy for X minutes straight, but can only use it once.

Return the maximum number of customers that can be satisfied throughout the day.

Example 1:

Input: customers = [1,0,1,2,1,1,7,5], grumpy = [0,1,0,1,0,1,0,1], X = 3
Output: 16
Explanation: The bookstore owner keeps themselves not grumpy for the last 3 minutes. 
The maximum number of customers that can be satisfied = 1 + 1 + 1 + 1 + 7 + 5 = 16.
"""
def maxSatisfied(customers, grumpy, X):
    s1 = 0 #calculate the summation of customers in non-grumpy days 
    s2 = 0 
    l = []
    m = 0 #store the maximum value for grumpy days in X periods 

    for i in range(len(grumpy)):
        if grumpy[i] == 0:
            s1 += customers[i]
        else:
            l.append(i)
            s2+= customers[i]
            while(i-l[0]>=X): #always limit the range of grumpy days
                index = l.pop(0)
                s2-=customers[index]
        m = max(m,s2)
    return s1+m

"""
945. Minimum Increment to Make Array Unique

Given an array of integers A, a move consists of choosing any A[i], and incrementing it by 1.

Return the least number of moves to make every value in A unique.

Example 1:

Input: [1,2,2]
Output: 1
Explanation:  After 1 move, the array could be [1, 2, 3].
"""
def minIncrementForUnique(arr):
    if not arr:
        return 0
    arr.sort()
    s, ans = arr[0], 0
    for i in arr:
        ans += max(0, s - i)
        s = max(s + 1, i + 1)
    return ans

"""
1668. Maximum Repeating Substring

For a string sequence, a string word is k-repeating if word concatenated k times is a substring of sequence. The word's maximum k-repeating value is the highest value k where word is k-repeating in sequence. If word is not a substring of sequence, word's maximum k-repeating value is 0.

Given strings sequence and word, return the maximum k-repeating value of word in sequence.

Example 1:

Input: sequence = "ababc", word = "ab"
Output: 2
Explanation: "abab" is a substring in "ababc".
Example 2:

Input: sequence = "ababc", word = "ba"
Output: 1
Explanation: "ba" is a substring in "ababc". "baba" is not a substring in "ababc".
"""
def maxRepeating(seq,word):
    count = 0
    resword = ""
    while True:
        resword = resword + word
        val = seq.find(resword)
        if val == -1:
            break
        else:
            count += 1
    return count

"""
659. Split Array into Consecutive Subsequences

Given an integer array nums that is sorted in ascending order, return true if and only if you can split it into one or more subsequences such that each subsequence consists of consecutive integers and has a length of at least 3.

Example 1:

Input: nums = [1,2,3,3,4,5]
Output: true
Explanation:
You can split them into two consecutive subsequences : 
1, 2, 3
3, 4, 5
"""
from collections import Counter,  defaultdict
def isPossible(nums):
    """
    :type nums: List[int]
    :rtype: bool
    """
    count_map = Counter(nums)
    # Maintains count of seq ending at index
    tail = defaultdict(int)
    # Sorted already
    for num in nums:
        if count_map[num] == 0:
            continue        
        if tail[num] > 0:
            tail[num] -= 1
            tail[num+1] += 1   
            count_map[num] -= 1
        elif count_map[num+1] > 0 and count_map[num+2] > 0:
            count_map[num] -= 1
            count_map[num+1] -=1
            count_map[num+2] -=1
            tail[num+3] += 1
        else:
            return False    
    return True
"""
1696. Jump Game VI

You are given a 0-indexed integer array nums and an integer k.

You are initially standing at index 0. In one move, you can jump at most k steps forward without going outside the boundaries of the array. That is, you can jump from index i to any index in the range [i + 1, min(n - 1, i + k)] inclusive.

You want to reach the last index of the array (index n - 1). Your score is the sum of all nums[j] for each index j you visited in the array.

Return the maximum score you can get.

Example 1:

Input: nums = [1,-1,-2,4,-7,3], k = 2
Output: 7
Explanation: You can choose your jumps forming the subsequence [1,-1,4,3] (underlined above). The sum is 7.

DP+ Deque
Time Complexity - O(N)
Space Complexity - O(N)

"""
def maxResult(nums,k):
    n = len(nums)
    score = [0]*n
    score[0] = nums[0]
    dp = collections.deque()
    dp.append(0)
    for i in range(1,n):
        print("Out",dp,score)
        while dp and dp[0]< i-k:
            print("While1",dp,score)
            dp.popleft()
        score[i] = score[dp[0]] + nums[i]
        while dp and score[i] >= score[dp[-1]]:
            print("While2",dp,score)
            dp.pop() 
        dp.append(i)    
    return score[-1]

"""
Given the root of a binary search tree, and an integer k, return the kth (1-indexed) smallest element in the tree.

Example 1:

Input: root = [3,1,4,null,2], k = 1
Output: 1
"""

def kth_smallest(root, k):
    stack = []
    while root or stack:
        while root:
            print("While Root",root.key)
            stack.append(root)
            root = root.left
        root = stack.pop()
        print("the root",root.key,root.left,root.right,stack)
        k -= 1
        if k == 0:
            break
        root = root.right
    return root.key
"""
Invert a binary tree.
"""
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def invertTree(root):
        if root is None:
            return None
        
        def dfs(root):
            if root.left:
                dfs(root.left)
            if root.right:
                dfs(root.right)
            
            root.left,root.right = root.right,root.left
        
        dfs(root)
        return root
    
# Definition for Employee.
class Employee:
    def __init__(self,id, importance, subordinates):
        self.id = id
        self.importance = importance
        self.subordinates = subordinates


def getImportance(employees, id):
    emap = {e.id:e for e in employees}
    def dfs(eid):
        employee = emap[eid]
        return (employee.importance + sum(dfs(eid) for eid in employee.subordinates))
    return dfs(id)

e = [Employee(1,5,[2,3]),Employee(2,3,[]),Employee(3,3,[])]
print("Employee Importance",getImportance(e,1))



def f(x):
    return x + 10

def verify(g):
    x = 5
    return g(x)

print(verify(f))
