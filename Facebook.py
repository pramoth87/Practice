# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 11:47:42 2020

@author: prchandr
"""
from collections import defaultdict, Counter
"""
23. Merge k Sorted Lists

Given an array of linked-lists lists, each linked list is sorted in ascending order.

Merge all the linked-lists into one sort linked-list and return it.

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
class Node: 
    # Function to initialise the node object 
    def __init__(self, data): 
        self.data = data  # Assign data 
        self.next = None  # Initialize next as null 
        
class mergeNodes:
    def mergeKLists(self,lists):
        self.Nodes = []
        head = points = Node(0)
        for l in lists():
            while l:
                self.Nodes.append(l.data)
                l = l.next
        for x in sorted(self.Nodes):
            points.next = Node(x)
            points = points.next
        
        return head.next
#######################################################################################################
"""
29. Divide Two Integers

Given two integers dividend and divisor, divide two integers without using multiplication, division and mod operator.

Return the quotient after dividing dividend by divisor.

The integer division should truncate toward zero, which means losing its fractional part. For example, truncate(8.345) = 8 and truncate(-2.7335) = -2.

Example 1:

Input: dividend = 10, divisor = 3
Output: 3
Explanation: 10/3 = truncate(3.33333..) = 3.
"""

def divider(dividend, divisor):
    MAX_INT = 2**31
    MIN_INT = -2**31
    
    if dividend == MAX_INT and divisor == 1:
        return MAX_INT
    if dividend == MIN_INT and divisor == 1:
        return MIN_INT
    if dividend == MIN_INT and divisor == -1:
        return MAX_INT
    negative = 2
    if dividend > 0:
        negative -= 1
        dividend = - dividend
    if divisor > 0:
        negative -= 1
        divisor = - divisor
    
    quotient = 0
    while dividend - divisor <= 0:
        quotient -= 1
        dividend -= divisor
    
    return -quotient if negative != 1 else quotient

"""
31. Next Permutation

Implement next permutation, which rearranges numbers into the lexicographically next greater permutation of numbers.

If such arrangement is not possible, it must rearrange it as the lowest possible order (ie, sorted in ascending order).

The replacement must be in-place and use only constant extra memory.

Here are some examples. Inputs are in the left-hand column and its corresponding outputs are in the right-hand column.

1,2,3 → 1,3,2
3,2,1 → 1,2,3
1,1,5 → 1,5,1

"""  

def nextPermutation(nums):
    n = len(nums)
    found = False
    i =  len(nums)-2
    while i >= 0:
        if nums[i] < nums[i+1]:
            found = True
            break
        i -= 1
    if not found:
        nums.sort()
        return nums
    else:
        j = i+1
        while j < n and nums[j] > nums[i]:
            j += 1
        j -= 1
        nums[i], nums[j] = nums[j], nums[i]
        nums[i+1:] = nums[i+1][::-1]
        
        return nums



"""
34. Find First and Last Position of Element in Sorted Array

Given an array of integers nums sorted in ascending order, find the starting and ending position of a given target value.

Your algorithm's runtime complexity must be in the order of O(log n).

If the target is not found in the array, return [-1, -1].

Example 1:

Input: nums = [5,7,7,8,8,10], target = 8
Output: [3,4]
Example 2:

Input: nums = [5,7,7,8,8,10], target = 6
Output: [-1,-1]
"""

def searchRange(nums,target):
    def findPos(nums,target,leftMost):
        lo = 0
        hi = len(nums)-1
        res = -1
        while lo <= hi:
            mid = (lo+hi)//2
            if nums[mid] == target:
                res = mid
                if leftMost:
                    hi = mid -1
                    continue
                else:
                    lo = mid + 1
                    continue
            if target >= nums[lo] and target < nums[mid]:
                hi = mid -1
            else:
                lo = mid + 1
        return res
    
    leftidx = findPos(nums,target,True)
    if leftidx == -1:
        return [-1,-1]
    rightidx = findPos(nums, target,False)
    return [leftidx,rightidx]

def myPow(x,n):
    if n == 0:
        return 1
    
    curProduct = x
    res = 1
    while n > 0:
        if n % 2 == 1:
            res *= curProduct
        curProduct *= curProduct
        n //= 2
    return res

def addSum(a,b):
    sumNum = int(a,2) + int(b,2)
    return bin(sumNum)[2:]

def merge(nums1, nums2):
    m = len(nums1)
    n = len(nums2)
    i = 0
    j = 0
    res = []
    while i < m and j < n:
        if nums1[i] == nums2[j]:
            res.append(nums1[i])
            res.append(nums2[j])
            i += 1
            j += 1
        if nums1[i] > nums2[j]:
            res.append(nums2[j])
            j += 1
        else:
            res.append(nums1[i])
            i += 1
    return res + nums1[i:] + nums2[j:]

"""
SImplify the path
"""

def simplifyPath(path):
    if not path:
        return path
    res = []
    for ch in path.split("/"):
        if ch == "..":
            if res:
                res.pop()
        elif ch == "." or not ch:
            continue
        else:
            res.append(ch)

    return "/"+ "/".join(res)

"""
Closest value in the binary search tree
"""
def closestValue(root, target):
    stack, pred = [], float('-inf')
    
    while stack or root:
        while root:
            stack.append(root)
            root = root.left
        root = stack.pop()
        
        if pred <= target and target < root.val:
            return min(pred,root.val, key = lambda x: abs(target- x))
        
        pred = root.val
        root = root.right
    
    return pred

def closesValueRec(root,target):
    def inOrder(root):
        return inOrder(root.left) + [root.value] + inOrder(root.right) if root else []
    return min(inOrder(root), key = lambda x : abs(target-x))

"""
124. Binary Tree Maximum Path Sum

Given a non-empty binary tree, find the maximum path sum.

For this problem, a path is defined as any sequence of nodes from some starting node to any node in the tree along the parent-child connections. The path must contain at least one node and does not need to go through the root.

Example 1:

Input: [1,2,3]
"""
def maxPathSum(root):    
    def solution(root):
        nonlocal maxSum
        if not root:
            return 
        right = 0
        left = 0
        if root.left is not None:
            left = solution(root.left)
        if root.right is not None:
            right = solution(root.right)
        
        localSum = max(max(left,right) +root.val, root.val)
        maxSum = max(localSum, left+right+root.val, root.val, maxSum)
        return maxSum
    maxSum = float('-inf')
    solution(root)
    return maxSum

"""
Product of the Array:
    
    Given an array nums of n integers where n > 1,  return an array output such that output[i] is equal to the product of all the elements of nums except nums[i].

Example:

Input:  [1,2,3,4]
Output: [24,12,8,6]
Constraint: It's guaranteed that the product of the elements of any prefix or suffix of the array (including the whole array) fits in a 32 bit integer.

Note: Please solve it without division and in O(n).
"""
def productExceptSelf1(nums):
    # The length of the input array 
    length = len(nums)
    # The answer array to be returned
    answer = [0]*length
    answer[0] = 1
    for i in range(1, length):
        answer[i] = nums[i - 1] * answer[i - 1]
    print(answer)
    R = 1
    for i in reversed(range(length)):
        print(i)
        answer[i] = answer[i] * R
        R *= nums[i]
        print(answer)
    print(answer)
    return answer

"""
Given a string s and an integer array indices of the same length.

The string s will be shuffled such that the character at the ith position moves to indices[i] in the shuffled string.

Return the shuffled string.

Example 1:
Input: s = "codeleet", indices = [4,5,6,7,0,2,1,3]
Output: "leetcode"
Explanation: As shown, "codeleet" becomes "leetcode" after shuffling.
"""
def restoreString(s,indices):
    myDict = {}
    for i in range(len(s)):
        myDict[indices[i]] = s[i]
    
    print(myDict)
    ans = ""
    for i in range(len(s)):
        ans += myDict[i]
    return ans
"""
140. Word Break II

Given a non-empty string s and a dictionary wordDict containing a list of non-empty words, add spaces in s to construct a sentence where each word is a valid dictionary word. Return all such possible sentences.

Note:

The same word in the dictionary may be reused multiple times in the segmentation.
You may assume the dictionary does not contain duplicate words.
Example 1:

Input:
s = "catsanddog"
wordDict = ["cat", "cats", "and", "sand", "dog"]
Output:
[
  "cats and dog",
  "cat sand dog"
]
"""
def wordBreak(s,wordDict):
    wordSet = set(wordDict)
    memo = defaultdict(list)
    
    def topDownDp(s):
        if not s:
            return [[]]
        if s in memo:
            return memo[s]
        
        for endIndex in range(1, len(s)+1):
            word = s[:endIndex]
            if word in wordSet:
                for sub in topDownDp(s[endIndex:]):
                    memo[s].append([word] + sub)
        
        return memo[s]
    topDownDp(s)
    return [" ".join(words) for words in memo[s]]

class Solution:
    def __init__(self):
        self.buf4 = []*4
        self.count = 0
        self.count4 = 0
        
    def read(self,buf,n):
        i = 0
        while i < n:
            if self.count == self.count4:
                self.count, self.count4 = 0, read4(self.buf4)
                if self.count4 == 0:
                    break
            
            buf[i] = self.buf4[self.count]
            self.count += 1
            i += 1
        return i

class BSTIterator:
    def __init__(self,root):
        self.sorted_item = []
        self.InOrder(root)
        self.index = -1
    
    def InOrder(self,root):
        if not root:
            return
        self.InOrder(root.left)
        self.sorted_item.append(root.val)
        self.InOrder(root.right)
    
    def next(self,root):
        self.index += 1
        return self.sorted_item[self.index]
    
    def hasNext(self):
        return self.index + 1 < len(self.sorted_item)
    

def rightSideViewBST(root):
    if not root:
        return []
    
    next_Level = deque([root,])
    right_Node = []
    
    while next_Level:
        curr_Level = next_Level
        next_Level = deque()
        while curr_Level:
            node = curr_Level.popleft()
            
            if node.left:
                next_Level.append(node.left)
            if node.right:
                next_level.append(node.right)
            
        right_Node.append(node.val)
    
    return right_Node

def rightSideViewDFSBST(root):
    if not root:
        return []
    rightSide = []

    def helper(root,level):
        if level == len(rightSide):
            rightSide.append(root.val)
        
        for child in [root.right,root.left]:
            if child:
                helper(child,level+1)  
    
    helper(root,0)
    return rightSide
"""
Find kth largest value in unsorted array
"""
def findkthValue(arr,k):
    def partition(pivot,start,end):
        left = start
        right = end
        
        while left <= right:
            while left < length and arr[pivot] > arr[left]:
                left += 1
            while right > pivot and arr[pivot] < arr[right]:
                right -= 1
            
            if left == right :
                break
            
            elif left < right:
                arr[left],arr[right] = arr[right], arr[left]
                left += 1
                right -= 1
                
        tmp = arr[pivot] 
        arr[pivot] = arr[left-1]
        arr[left-1] = tmp
        return right
    length = len(arr)
    pivot = 0
    start = pivot + 1
    end = length-1
    
    while True:
        pivot = partition(pivot,start,end)
        
        if pivot == (length - k):
            break
        elif pivot < (length - k):
            pivot += 1
            start = pivot + 1
            
        else:
            end = pivot - 1
            pivot = 0
            start = pivot + 1
    
    return arr[length-k]


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
249. Group Shifted Strings

Given a string, we can "shift" each of its letter to its successive letter, for example: "abc" -> "bcd". We can keep "shifting" which forms the sequence:

"abc" -> "bcd" -> ... -> "xyz"
Given a list of non-empty strings which contains only lowercase alphabets, group all strings that belong to the same shifting sequence.

Example:

Input: ["abc", "bcd", "acef", "xyz", "az", "ba", "a", "z"],
Output: 
[
  ["abc","bcd","xyz"],
  ["az","ba"],
  ["acef"],
  ["a","z"]
]
"""
def groupStrings(strings):
    diff = lambda s: tuple((ord(a)-ord(b))%26 for a,b in zip(s,s[1:]))
    #print(diff)
    d = defaultdict(list)
    for s in strings:
        d[diff(s)].append(s)
    print(d, diff("abc"), diff("xyz"))
    return d.values()
"""
278. First Bad Version
You are a product manager and currently leading a team to develop a new product. Unfortunately, the latest version of your product fails the quality check. Since each version is developed based on the previous version, all the versions after a bad version are also bad.

Suppose you have n versions [1, 2, ..., n] and you want to find out the first bad one, which causes all the following ones to be bad.

You are given an API bool isBadVersion(version) which will return whether version is bad. Implement a function to find the first bad version. You should minimize the number of calls to the API.

Example:

Given n = 5, and version = 4 is the first bad version.

call isBadVersion(3) -> false
call isBadVersion(5) -> true
call isBadVersion(4) -> true

Then 4 is the first bad version. 
"""
def firstBadVersion(n):
    l = 0
    h = n
    if n == 1:
        if isBadVersion(1) == True:
            return 1
        else:
            return False
    
    while l <= h:
        m = (l+h) // 2
        
        if isBadVersion(m-1) == False and isBadVersion(m) == True:
            return m
        
        elif isBadVersion(m-1) == False and isBadVersion(m) == False:
            l = m + 1
        
        elif isBadVersion(m-1) == True and isBadVersion(m) == True:
            h = m
    
    return False

from datetime import datetime
def daysBetweenDates(date1,date2):
    
    date_format = "%y-%m-%d"
    a = datetime.strptime(date1,date_format)
    b = datetime.strptime(date2, date_format)
    delta = a - b
    return abs(delta.days)

def leftMostColumnWithOne(binaryMatrix):
    #dimensions =  binaryMatrix.dimensions()
    n = len(binaryMatrix)
    m = len(binaryMatrix[0])
    print(n,m)
    i = 0
    j = m -1
    leftMost = -1
    while i < n and j >= 0:
        print(i,j,m,n)
        result = binaryMatrix[i][j]
        if result == 0:
            i += 1
        else:
            leftMost = j
            j -= 1
    return leftMost

def leftMostColumnBS(binaryMatrix):
    rows = len(binaryMatrix)
    col = len(binaryMatrix[0])
    
    l = 0
    r = col-1
    def find_Ones(arr,col):
        l = 0
        r = col
        while l <= r:
            mid = (l + r) //2
            if mid == 0 and arr[mid] == 1:
                print("Mid 0",arr[mid])
                return mid
            if mid > 0:
                print(l,mid, r)
                if arr[mid] == 1 and arr[mid-1] == 0:
                    print("Value:",arr[mid])
                    return mid
                if arr[mid] == 1 and arr[mid-1] == 1:
                    r = mid - 1
                else:
                    l = mid + 1
        return -1
    while l < rows and r >= 0:
        print(binaryMatrix[l],l,r)
        result = find_Ones(binaryMatrix[l], r)
        if result != -1:
            l += 1
            r -= 1
        else:
            l += 1
    
    return result
            

def addOperators(num,target):
    n = len(num)
    res = []
    def dfs(idx,cur,pre,path):
        if idx >= n:
            if cur + pre == target:
                res.append(path)
                return
        
        for i in range(idx+1, n+1):
            s = num[idx:i]
            if s != str(int(s)): return
            
            k = int(s)
            if idx == 0:
                dfs(i, cur+pre, k, str(k))
            
            else:
                dfs(i,cur+pre, k, path + "+" + str(k))
                dfs(i,cur+pre, -k, path + "-" + str(k))
                dfs(i,cur, pre*k, path + "*" + str(k))    
    dfs(0,0,0,"")
    return res
"""
297. Serialize and Deserialize Binary Tree

Serialization is the process of converting a data structure or object into a sequence of bits so that it can be stored in a file or memory buffer, or transmitted across a network connection link to be reconstructed later in the same or another computer environment.

Design an algorithm to serialize and deserialize a binary tree. There is no restriction on how your serialization/deserialization algorithm should work. You just need to ensure that a binary tree can be serialized to a string and this string can be deserialized to the original tree structure.

Example: 

You may serialize the following tree:

    1
   / \
  2   3
     / \
    4   5

as "[1,2,3,null,null,4,5]"
"""
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
    
class code:
    def serialize(self,root):
        def serializeDFS(root,string):
            if root is None:
                string += 'None'
            else:
                string += str(root.val) + ','
                string = serializeDFS(root.left, string)
                string = serializeDFS(root.right, string)
        
            return string
        
        return serializeDFS(root, '')
    
    def deserialize(self,data):
        def deSerializeDFS(string):
            
            if string[0] == 'None':
                string.pop(0)
                return None
            root = TreeNode(string[0])
            string.pop(0)
            root.left = deSerializeDFS(string)
            root.right = deSerializeDFS(string)
            return root
        data_list = data.split(',')
        root = deSerializeDFS(data_list)
        return root
"""
311. Sparse Matrix Multiplication

Given two sparse matrices A and B, return the result of AB.

You may assume that A's column number is equal to B's row number.

Example:

Input:

A = [
  [ 1, 0, 0],
  [-1, 0, 3]
]

B = [
  [ 7, 0, 0 ],
  [ 0, 0, 0 ],
  [ 0, 0, 1 ]
]

Output:

     |  1 0 0 |   | 7 0 0 |   |  7 0 0 |
AB = | -1 0 3 | x | 0 0 0 | = | -7 0 3 |
                  | 0 0 1 |
"""
def multiply(A,B):
    ans = [[0 for i in range(len(B[0]))] for j in range(len(A))]
    dict1 = {i:[] for i in range(len(A))}
    for i in range(len(A)):
        for j in range(len(A[0])):
            if A[i][j]!=0:
                dict1[i].append(j)
    print(dict1)            
    dict2 = {i:[] for i in range(len(B[0]))}
    for i in range(len(B)):
        for j in range(len(B[0])):
            if B[i][j]!=0:
                dict2[j].append(i)
    print(dict2)
    for i in range(len(A)):
        for j in range(len(B[0])):
            set1 = set(dict1[i])
            set2 = set(dict2[j])
            mul = list(set1.intersection(set2))
            print(mul,set1,set2,i,j)             
            for t in mul:
                ans[i][j]+=A[i][t]*B[t][j]
    return ans
"""

"""

def verticalTranversalBFS(root):
    if not root:
        return []
    table = collections.defaultdict(list)
    queue = deque([(root,0)])
    while queue:
        node, col = queue.popleft()
        if node is not None:
            table[col].append(node.key)
            queue.append((node.left,col-1))
            queue.append((node.right,col+1))
    
    return [table[x] for x in sorted(table.keys())]

def verticalTraversal(root):
    node_list = []
    
    def BFS(root):
        queue = deque([(root, 0, 0)])
        while queue:
            node, row, column = queue.popleft()
            if node is not None:
                node_list.append((column, row, node.val))
                queue.append((node.left, row + 1, column - 1))
                queue.append((node.right, row + 1, column + 1))
    
    # step 1). construct the global node list, with the coordinates
    BFS(root)
    
    # step 2). sort the global node list, according to the coordinates
    node_list.sort()
    
    # step 3). retrieve the sorted results partitioned by the column index
    ret = OrderedDict()
    for column, row, value in node_list:
        if column in ret:
            ret[column].append(value)
        else:
            ret[column] = [value]

    return ret.values()
"""
lengthOfLongestSubstringDistrinct
"""

def lengthOfLongestSubstringDistrinct(s,k):
    n = len(s)
    if n == 0 and k == 0:
        return 0
    d = defaultdict()
    left = 0
    right = 0
    maxLength = 1
    while right < n:
        d[s[right]] = right
        right += 1
        if len(d) == k + 1:
            temp = min(d.values())
            del d[s[temp]]
            left = temp+1
        maxLength = max(maxLength, right-left)
    return maxLength

"""
Random Pick
"""

class Solution:
    def __init__(self,nums):
        self.dict = defaultdict(list)
        for i in range(len(nums)):
            self.dict[nums[i]].append(i)
    
    def pick(self, target):
        
        return random.choice(self.dict[target])



"""
Nested Sum
"""

def depthSum(nestedList):
    sum = 0
    stack = [(l,1) for l in nestedList]
    while stack:
        val, index = stack.pop(0)
        value = val.getInteger()
        if value is not None:
            sum += value * index
        else:
            for item in val.getList():
                stack.append((item,index+1))
    
    return sum

def depthSumrec(nestedList):
    return recSum(nestedList, 1)
		
def recSum(nestedList, depth):
    sum = 0
    for item in nestedList:
        if item.isInteger():
            sum += (item.getInteger() * depth)
        else:
            sum += self.recSum(item.getList(), depth+1)		
    return sum

"""
Add Two string
"""

def addStrings(nums1, nums2):
    
    res = []
    carry = 0
    p1 = len(nums1)-1
    p2 = len(nums2)-1
    while p1 >= 0 and p2 >= 0:
        x1 = ord(nums1[p1])- ord('0') if p1>=0 else 0
        x2 = ord(nums2[p2])- ord('0') if p2>=0 else 0
        value = (x1+x2+carry) % 10
        carry = (x1+x2+carry) // 10
        res.append(value)
        p1-=1
        p2-=1
    
    if carry:
        res.append(carry)
    
    return ''.join(str(x) for x in res[::-1])

"""
438. Find All Anagrams in a String

Given a string s and a non-empty string p, find all the start indices of p's anagrams in s.

Strings consists of lowercase English letters only and the length of both strings s and p will not be larger than 20,100.

The order of output does not matter.

Example 1:

Input:
s: "cbaebabacd" p: "abc"

Output:
[0, 6]

Explanation:
The substring with start index = 0 is "cba", which is an anagram of "abc".
The substring with start index = 6 is "bac", which is an anagram of "abc".

"""

def findAnagram(s,p):
    ns = len(s)
    np = len(p)
    if ns < np:
        return []
    pcount = Counter(p)
    scount = Counter()
    output = []
    for i in range(ns):
        scount[s[i]] += 1
        if i >= np:
            print(scount,pcount,ns,np)
            if scount[s[i-np]] == 1:
                del scount[s[i-np]]
            else:
                scount[s[i-np]] -= 1
        if pcount == scount:
            output.append(i-np + 1)
    
    return output
"""
523. Continuous Subarray Sum

Given a list of non-negative numbers and a target integer k, write a function to check if the array has a continuous subarray of size at least 2 that sums up to a multiple of k, that is, sums up to n*k where n is also an integer.

Example 1:

Input: [23, 2, 4, 6, 7],  k=6
Output: True
Explanation: Because [2, 4] is a continuous subarray of size 2 and sums up to 6.
"""
def checkSubArraySum(nums,k):
    res = {0:-1}
    sum = 0
    for i in range(len(nums)):
        print(res,sum)
        sum += nums[i]
        print(res,sum)
        if k != 0:
            sum = sum%k
        if sum in res:
            if i - res[sum] > 1:
                return True
        else:
            res[sum] = i
    return False

"""
528. Random Pick with Weight

You are given an array of positive integers w where w[i] describes the weight of ith index (0-indexed).

We need to call the function pickIndex() which randomly returns an integer in the range [0, w.length - 1]. pickIndex() should return the integer proportional to its weight in the w array. For example, for w = [1, 3], the probability of picking the index 0 is 1 / (1 + 3) = 0.25 (i.e 25%) while the probability of picking the index 1 is 3 / (1 + 3) = 0.75 (i.e 75%).

More formally, the p


ability of picking index i is w[i] / sum(w).
"""

class Solution:
    def __init__(self, w):
        """
        :type w: List[int]
        """
        self.prefix_sums = []
        prefix_sum = 0
        for weight in w:
            prefix_sum += weight
            self.prefix_sums.append(prefix_sum)
        self.total_sum = prefix_sum

    def pickIndex(self):
        """
        :rtype: int
        """
        target = self.total_sum * random.random()
        # run a binary search to find the target zone
        low, high = 0, len(self.prefix_sums)
        while low < high:
            mid = low + (high - low) // 2
            if target > self.prefix_sums[mid]:
                low = mid + 1
            else:
                high = mid
        return low

"""
543. Diameter of Binary Tree

Given a binary tree, you need to compute the length of the diameter of the tree. The diameter of a binary tree is the length of the longest path between any two nodes in a tree. This path may or may not pass through the root.

Example:
Given a binary tree
          1
         / \
        2   3
       / \     
      4   5    
Return 3, which is the length of the path [4,2,1,3] or [5,2,1,3].
"""

def diameterBST(root):
    ans = 1
    def depth(node):
        nonlocal ans
        if not node:
            return 0
        l = depth(node.left)
        r = depth(node.right)
        ans = max(ans, l+r+1)
        return max(l,r) + 1
    depth(root)
    return ans - 1

"""
560. Subarray Sum Equals K

Given an array of integers and an integer k, you need to find the total number of continuous subarrays whose sum equals to k.

Example 1:

Input:nums = [1,1,1], k = 2
Output: 2
"""

def subarraySum(nums, k):
    myDict = defaultdict(list)
    sums = 0
    count = 0
    
    for i in range(len(nums)):
        sums += nums[i]
        if sums == k:
            count += 1
        if sums-k in myDict:
            count += len(myDict[sums-k])
        else:
            myDict[sums].append(i)
    
    return count

"""
621. Task Scheduler

Given a characters array tasks, representing the tasks a CPU needs to do, where each letter represents a different task. Tasks could be done in any order. Each task is done in one unit of time. For each unit of time, the CPU could complete either one task or just be idle.

However, there is a non-negative integer n that represents the cooldown period between two same tasks (the same letter in the array), that is that there must be at least n units of time between any two same tasks.

Return the least number of units of times that the CPU will take to finish all the given tasks.

Example 1:

Input: tasks = ["A","A","A","B","B","B"], n = 2
Output: 8
Explanation: 
A -> B -> idle -> A -> B -> idle -> A -> B
There is at least 2 units of time between any two same tasks.
"""

def leastInterval(tasks, n):
    counts = defaultdict(int)
    for t in tasks:
        counts[t] += 1
    lst = sorted(counts.values())
    print(lst,counts)
    maxNumber = lst[-1]
    counter = 0
    print(maxNumber,counter)
    while lst and lst[-1] == maxNumber:
        print(lst)
        lst.pop()
        counter += 1
    ret = (maxNumber-1)*(n+1)+counter
    print(ret,maxNumber,counter)
    return max(len(tasks), ret)
"""
636. Exclusive Time of Functions

On a single threaded CPU, we execute some functions.  Each function has a unique id between 0 and N-1.

We store logs in timestamp order that describe when a function is entered or exited.

Each log is a string with this format: "{function_id}:{"start" | "end"}:{timestamp}".  
For example, "0:start:3" means the function with id 0 started at the beginning of timestamp 3. 
 "1:end:2" means the function with id 1 ended at the end of timestamp 2.

A function's exclusive time is the number of units of time spent in this function.  
Note that this does not include any recursive calls to child functions.

The CPU is single threaded which means that only one function is being executed at a given time unit.

Return the exclusive time of each function, sorted by their function id.

Example 1:

Input:
n = 2
logs = ["0:start:0","1:start:2","1:end:5","0:end:6"]
Output: [3, 4]
Explanation:
Function 0 starts at the beginning of time 0, then it executes 2 units of time and reaches the end of time 1.
Now function 1 starts at the beginning of time 2, executes 4 units of time and ends at time 5.
Function 0 is running again at the beginning of time 6, and also ends at the end of time 6, thus executing for 1 unit of time. 
So function 0 spends 2 + 1 = 3 units of total time executing, and function 1 spends 4 units of total time executing.
"""
def exclusiveTime(n,logs):
    counts = [0] * n
    stack = []
    for log in logs:
        curr = log.split(":")
        if curr[1] == "start":
            stack.append([curr,0])
        print(stack)
        if curr[1] =="end":
            prev,taken = stack.pop()
            time = int(curr[2]) - int(prev[2]) + 1
            counts[int(curr[0])] += time - taken
            if stack:
                stack[-1][1] += time
        print("end",stack,counts)
    return counts

"""
670. Maximum Swap

Given a non-negative integer, you could swap two digits at most once to get the maximum valued number. Return the maximum valued number you could get.

Example 1:
Input: 2736
Output: 7236
Explanation: Swap the number 2 and the number 7.
"""

def maxSwap(num):
    s = list(str(num))
    left,right = -1,-1
    maxIndex, maxDigit = -1, chr(0)
    print(s,maxIndex, maxDigit)
    for i in range(len(s)-1,-1,-1):
        if s[i] > maxDigit:
            maxDigit = s[i]
            maxIndex = i    
        elif s[i] < maxDigit:
            left,right = i, maxIndex
        print(s,maxIndex, maxDigit, left,right)
    s[left], s[right] = s[right], s[left]
    return int(''.join(s))

"""
680. Valid Palindrome II

Given a non-empty string s, you may delete at most one character. Judge whether you can make it a palindrome.

Example 1:
Input: "aba"
Output: True
"""

def validPalindrome(s):
    l,r = 0 ,len(s)-1
    
    while l < r:
        if s[l] == s[r]:
            l += 1
            r -= 1
        
        else:
            tem1 = s[:l] + s[l+1:]
            tem2 = s[:r] + s[r+1:]
            print(tem1,tem2)
            return tem1==tem1[::-1] or tem2==tem2[::-1]
    return True

def isPalindrome(s):

    i, j = 0, len(s) - 1

    while i < j:
        while i < j and not s[i].isalnum():
            i += 1
        while i < j and not s[j].isalnum():
            j -= 1

        if i < j and s[i].lower() != s[j].lower():
            return False

        i += 1
        j -= 1

    return True

"""
721. Accounts Merge

Given a list accounts, each element accounts[i] is a list of strings, where the first element accounts[i][0] is a name, and the rest of the elements are emails representing emails of the account.

Now, we would like to merge these accounts. Two accounts definitely belong to the same person if there is some email that is common to both accounts. Note that even if two accounts have the same name, they may belong to different people as people could have the same name. A person can have any number of accounts initially, but all of their accounts definitely have the same name.

After merging the accounts, return the accounts in the following format: the first element of each account is the name, and the rest of the elements are emails in sorted order. The accounts themselves can be returned in any order.

Example 1:
Input: 
accounts = [["John", "johnsmith@mail.com", "john00@mail.com"], ["John", "johnnybravo@mail.com"], ["John", "johnsmith@mail.com", "john_newyork@mail.com"], ["Mary", "mary@mail.com"]]
Output: [["John", 'john00@mail.com', 'john_newyork@mail.com', 'johnsmith@mail.com'],  ["John", "johnnybravo@mail.com"], ["Mary", "mary@mail.com"]]
Explanation: 
The first and third John's are the same person as they have the common email "johnsmith@mail.com".
The second John and Mary are different people as none of their email addresses are used by other accounts.
We could return these lists in any order, for example the answer [['Mary', 'mary@mail.com'], ['John', 'johnnybravo@mail.com'], 
['John', 'john00@mail.com', 'john_newyork@mail.com', 'johnsmith@mail.com']] would still be accepted.
"""
def accountsMerge(accounts):
    em_to_name = {}
    graph = collections.defaultdict(set)
    for acc in accounts:
        name = acc[0]
        for email in acc[1:]:
            graph[acc[1]].add(email)
            graph[email].add(acc[1])
            em_to_name[email] = name
    seen = set()
    ans = []
    for email in graph:
        print(email)
        if email not in seen:
            seen.add(email)
            stack = [email]
            component = []
            while stack:
                node = stack.pop()
                component.append(node)
                for nei in graph[node]:
                    if nei not in seen:
                        seen.add(nei)
                        stack.append(nei)
            ans.append([em_to_name[email]] + sorted(component))
    return ans
"""
953. Verifying an Alien Dictionary

In an alien language, surprisingly they also use english lowercase letters, but possibly in a different order. The order of the alphabet is some permutation of lowercase letters.

Given a sequence of words written in the alien language, and the order of the alphabet, return true if and only if the given words are sorted lexicographicaly in this alien language.

 

Example 1:

Input: words = ["hello","leetcode"], order = "hlabcdefgijkmnopqrstuvwxyz"
Output: true
Explanation: As 'h' comes before 'l' in this language, then the sequence is sorted.
"""

def isAlienSorted(words,order):
    orderInd = {c:i for i,c in enumerate(order)} 
    for i in range(len(words)-1):
        word1 = words[i]
        word2 = words[i+1]     
        for k in range(min(len(word1) , len(word2))):
            if word1[k] != word2[k]:
                if orderInd[word1[k]] > orderInd[word2[k]]:
                    return False
                break
        else:
            if len(word1)>len(word2):
                return False     
        return True
    
"""
1026. Maximum Difference Between Node and Ancestor

Given the root of a binary tree, find the maximum value V for which there exists different nodes A and B where V = |A.val - B.val| and A is an ancestor of B.

(A node A is an ancestor of B if either: any child of A is equal to B, or any child of A is an ancestor of B.)

Example 1:

Input: [8,3,10,1,6,null,14,null,null,4,7,13]
Output: 7
Explanation: 
We have various ancestor-node differences, some of which are given below :
|8 - 3| = 5
|3 - 7| = 4
|8 - 1| = 7
|10 - 13| = 3
Among all possible differences, the maximum value of 7 is obtained by |8 - 1| = 7.
"""
def maxAncestorDiff(root):
    """
    :type root: TreeNode
    :rtype: int
    """
    maxDiff = 0
    
    def dfs(root, maxAncestor, minAncestor):
        nonlocal maxDiff
        if not root:
            return
        maxDiff = max(maxDiff, abs(maxAncestor - root.val), abs(minAncestor - root.val))
        maxAncestor, minAncestor = max(root.val, maxAncestor), min(root.val, minAncestor)
        dfs(root.left, maxAncestor, minAncestor)
        dfs(root.right, maxAncestor, minAncestor)
    
    dfs(root, root.val, root.val)
    return maxDiff
"""
1249. Minimum Remove to Make Valid Parentheses
Given a string s of '(' , ')' and lowercase English characters. 

Your task is to remove the minimum number of parentheses ( '(' or ')', in any positions ) so that the resulting parentheses string is valid and return any valid string.

Formally, a parentheses string is valid if and only if:

It is the empty string, contains only lowercase characters, or
It can be written as AB (A concatenated with B), where A and B are valid strings, or
It can be written as (A), where A is a valid string.
 

Example 1:

Input: s = "lee(t(c)o)de)"
Output: "lee(t(c)o)de"
Explanation: "lee(t(co)de)" , "lee(t(c)ode)" would also be accepted.
"""
def minRemoveToMakeValid(s):
    open_par = []
    s = list(s)
    
    for idc, char in enumerate(s):
        if char == '(':
            open_par.append(idc)
        elif char == ')':
            if open_par:
                open_par.pop()
            else:
                s[idc] = ""
    while open_par:
        s[open_par.pop()] = ""
        
    return "".join(s)


"""
364. Nested List Weight Sum II

Given a nested list of integers, return the sum of all integers in the list weighted by their depth.

Each element is either an integer, or a list -- whose elements may also be integers or other lists.

Different from the previous question where weight is increasing from root to leaf, now the weight is defined from bottom up. i.e., the leaf level integers have weight 1, and the root level integers have the largest weight.

Example 1:

Input: [[1,1],2,[1,1]]
Output: 8 
Explanation: Four 1's at depth 1, one 2 at depth 2.
"""

def depthSumInverse(nestedList):
    maxdepth = 1
    ans = []
    def dfs(nl, depth=1):
        nonlocal maxdepth
        maxdepth=max(maxdepth, depth)
        if nl.isInteger():
            ans.append([nl.getInteger(), depth])
            return
        for i in nl.getList():
            dfs(i, depth+1)
    for nl in nestedList:         
        dfs(nl)
    print(ans,maxdepth)
    return sum(i * (maxdepth + 1 - d) for i, d in ans)  


def numbits(n):
    count = 0
    while n:
        count += n & 1
        n = n >> 1
    return count
        


def numFriendRequests(ages):
    count = [0] * 121 # Age Limit
    for i in ages:
        count[i] += 1
    print(count)
    ans = 0
    for ageA, cA in enumerate(count):
        for ageB, cB in enumerate(count):
            print(ageA,ageB,cA,cB)
            if ageA*0.5 + 7 >= ageB:
                continue
            if ageA < ageB:
                continue
            if ageA < 100 and ageB > 100:
                continue
            
            ans += cA*cB
            if ageA == ageB:
                ans -= cA
    return ans      
        
"""
1379. Find a Corresponding Node of a Binary Tree in a Clone of That Tree

Given two binary trees original and cloned and given a reference to a node target in the original tree.

The cloned tree is a copy of the original tree.

Return a reference to the same node in the cloned tree.

Note that you are not allowed to change any of the two trees or the target node and the answer must be a reference to a node in the cloned tree.

Follow up: Solve the problem if repeated values on the tree are allowed.

 

Example 1:


Input: tree = [7,4,3,null,null,6,19], target = 3
Output: 3
Explanation: In all examples the original and cloned trees are shown. The target node is a green node from the original tree. The answer is the yellow node from the cloned tree.
"""      
        
def getTargetCopy(original, cloned, target):
    queue_o = deque([original,])
    queue_c = deque([cloned,])

    while queue_o:
        node_o = queue_o.popleft()
        node_c = queue_c.popleft()
        
        if node_o is target:
            return node_c

        if node_o:
            queue_o.append(node_o.left)
            queue_o.append(node_o.right)
            
            queue_c.append(node_c.left)
            queue_c.append(node_c.right) 
        
"""
724. Find Pivot Index

Given an array of integers nums, calculate the pivot index of this array.

The pivot index is the index where the sum of all the numbers strictly to the left of the index is equal to the sum of all the numbers strictly to the index's right.

If the index is on the left edge of the array, then the left sum is 0 because there are no elements to the left. This also applies to the right edge of the array.

Return the leftmost pivot index. If no such index exists, return -1.

Example 1:

Input: nums = [1,7,3,6,5,6]
Output: 3
Explanation:
The pivot index is 3.
Left sum = nums[0] + nums[1] + nums[2] = 1 + 7 + 3 = 11
Right sum = nums[4] + nums[5] = 5 + 6 = 11
""" 
        
def pivotIndex(nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    sumL = 0
    sumR = sum(nums)
    for i in range(len(nums)):
        sumR -= nums[i]
        if sumL == sumR:
            return i
        sumL += nums[i]
    return -1
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        