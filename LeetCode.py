# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 18:28:33 2020



@author: prchandr
"""
from itertools import combinations
from collections import defaultdict
from heapq import heappop, heappush, heapify
import sys


def lengthOfLongestSubstring(word):
    n = len(word)
    longest = 0
    for i in range(n):
        seen = set()
        for j in range(i, n):
            print(j)
            if word[j] in seen: break
            seen.add(word[j])
        longest = max(len(seen), longest)
        print(seen)
    return longest

def threeSum(arr):
    n = len(arr)
    if n < 3:
        return False
    found = False
    for i in range(n-1):
        s = set()
        for j in range(i+1, n):
            x = - (arr[i] + arr[j])
            if x in s:
                print(x,arr[i],arr[j])
                found = True
            else:
                s.add(arr[j])
    if found == False:
        print("No Triplets")

def threeSumwithCombinations(arr):
    res  = set()
    for i in combinations(arr,3):
        if sum(i) == 0:
            tup = sorted(i)
            res.add(" ".join([str(x) for x in tup]))
            print(i)
    out = "/n" .join(res)
    return res

def removeDuplicate(arr):
    count = len(arr)
    for i in range(count-1):
        if arr[i] == arr[i+1]:
            count -= 1           
    return count

#def nextPermutation(nums):
class ListNode(object): 
    def __init__(self, x):
        self.val = x
        self.next = None
        
    def mergeKLists(lists):
        """
        :type lists: List[ListNode]
        :rtype: ListNode
        """
        nodes = []
        head = point = ListNode(0)
        for l in lists:
            while l:
                nodes.append(l)
                l = l.next
        for x in sorted(nodes):
            point.next = ListNode(x)
            point = point.next
        return head.next

def searchRange(nums,target):
    #indexList = []
    for i in range(len(nums)):
        if nums[i]==target:
            leftIndex = i
            break
    else:
        return [-1,-1]
    for j in range(len(nums)-1,-1,-1):
        if nums[j] == target:
            rightIndex = j
            break
    return [leftIndex, rightIndex]

def insertInterval(arr,interval):
    newSet = []
    go = True
    for v in arr:
        if go:
            if v[0] <= interval[0] <= v[1] or interval[0] <= v[0]:
                newSet.append(((min(v[0],interval[0])), max(v[1],interval[1])))
                go = False
            else:
                newSet.append(v)
        else:
            newSet = insertInterval(newSet,v)
    if go:
        newSet.append(interval)
    return newSet

def validNumber(st):
    count = 0
    for a in st:
        if (a.isnumeric()) == True:
            count += 1
        else:
            return False
            break
    if count == len(st):
        return True
            

"""String Section"""
""" Given a string, find the length of the longest substring without repeating characters.
"""
def longestSubstring(words):
    n = len(words)
    longest = 0
    for i in range(n):
        seen = set()
        for j in range(i,n):
            if words[j] in seen: 
                break
            else:
                seen.add(words[j])
                longest = max(len(seen),longest)
    return longest

def lengthOfLongestSubstring(s):
    """
    :type s: str
    :rtype: int
    """
    
    str_list = []
    max_length = 0
    
    for x in s:
        if x in str_list:
            str_list = str_list[str_list.index(x)+1:]
        str_list.append(x)
        max_length = max(max_length, len(str_list))
        
    return max_length

"""Given a string s, find the longest palindromic substring in s. You may assume that the maximum length of s is 1000."""
def longestPalindrome(s):
    m = ''
    n = len(s)
    for i in range(n):
        print(s[i])
        for j in range(n,i,-1):
            if len(m) >= j-i:
                break
            elif s[i:j] == s[i:j][::-1]:
                m = s[i:j]
                print(m)
                break
    return len(m)

def longestPalindroneFaster(s):
    result = ""
    n = len(s)
    for i in range(n):
        j = i + 1
        while j <= len(s) and len(result) <= j-i:
            if s[i:j] == s[i:j][::-1] and len(s[i:j]) > len(result):
                result = s[i:j]
            j += 1 
    return result

"""
The string "PAYPALISHIRING" is written in a zigzag pattern on a given number of rows like this: (you may want to display this pattern in a fixed font for better legibility)

Input: s = "PAYPALISHIRING", numRows = 3
Output: "PAHNAPLSIIGYIR"

"""

def zigZag(s, numRows):
    if numRows == 1:
        return s 
    n = len(s)
    print("lenghth of string",n)
    cycle = 2*numRows - 2
    strlist = []
    for i in range(numRows):
        print("i",i)
        for j in range(i, n, cycle):
            print("j",j)
            strlist.append(s[j])
            print(strlist)
            if i != numRows-1 and i != 0 and j+cycle-2*i < n:
                print("if stmt",i, j+cycle-2*i)
                strlist.append(s[j+cycle-2*i])             
    newstr = ''.join(strlist)
    return newstr

"""Two Sum where sum of the target should be returned
twosum([2, 7, 11, 15],9) -> Input
Output - > [0,1]
"""
def twoSum(num,target):
    tempDict = {}
    tempDict[num[0]] = 0
    for i in range(1, len(num)):
        checkSum = target - num[i]
        if checkSum in tempDict:
            return [tempDict[checkSum],i]
        else:
            tempDict[num[i]] = i
            
def atoi(st):
    pointer = solution = 0
    isNegative = False
    while pointer<len(st) and st[pointer]== ' ':
        pointer +=1
    if pointer == len(st):
        return 0
    if st[pointer]=='-':
        isNegative = True
        pointer += 1
    elif st[pointer] == '+':
        isNegative = False
        pointer += 1
    for pointer in range(pointer, len(st)):
        if not st[pointer].isdigit():
            break
        else:
            solution *= 10
            solution += int(st[pointer])
    if not isNegative and solution > 2147483647:
        return 2147483647
    elif isNegative and solution > 2147483648:
        return -2147483648
    if isNegative:
        return -1*solution
    else:
        return solution
    
def intToRoman(num):
    num_map = [(1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'), (100, 'C'), (90, 'XC'),
       (50, 'L'), (40, 'XL'), (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')]
    roman = ''
    while num > 0:
        for i, r in num_map:
            while num >= i:
                roman += r
                num -= i
        return roman

def RomantoInt(roman): 
    values = {"I": 1,"V": 5,"X": 10,"L": 50,"C": 100,"D": 500,"M": 1000}
    total = 0
    i = 0
    while i < len(roman):
        if i+1 < len(roman) and values[roman[i]] < values[roman[i+1]]:
            total += values[roman[i+1]] - values[roman[i]]
            i += 2
        else:
            total += values[roman[i]]
            i += 1
    return total

"""

Write a function to find the longest common prefix string amongst an array of strings.

If there is no common prefix, return an empty string "".

Example 1:

Input: ["flower","flow","flight"]
Output: "fl"
"""

def longestCommonPrefix(strs):   
    if len(strs) == 0:
        return '' 
    res = ''
    strs = sorted(strs)
    for i in strs[0]:
        if strs[-1].startswith(res+i):
            print(i)
            res += i
        else:
            break
    return res

"""
Given an integer n, return a string with n characters such that each character in such string occurs an odd number of times.

The returned string must contain only lowercase English letters. If there are multiples valid strings, return any of them.  

 

Example 1:

Input: n = 4
Output: "pppz"
Explanation: "pppz" is a valid string since the character 'p' occurs three times and the character 'z' occurs once. Note that there are many other valid strings such as "ohhh" and "love".
"""

def generateTheString(n):
    return 'a' * n if n % 2 == 1 else 'a' * (n-1) + 'b'

"""
Given a string s , find the length of the longest substring t  that contains at most 2 distinct characters.

Example 1:

Input: "eceba"
Output: 3
Explanation: t is "ece" which its length is 3.

"""   
             
def lengthofLongestDistinct(s):
    n = len(s) 
    if n < 3:
        return n
    # sliding window left and right pointers
    left, right = 0, 0
    # hashmap character -> its rightmost position 
    # in the sliding window
    hashmap = defaultdict()
    max_len = 2
    while right < n:
        print(hashmap,right,left)
    # slidewindow contains less than 3 characters
        if len(hashmap) < 3:
            hashmap[s[right]] = right
            right += 1
            print("inside if",hashmap)
    # slidewindow contains 3 characters
        if len(hashmap) == 3:
        # delete the leftmost character
            del_idx = min(hashmap.values())
            print("del",del_idx)
            del hashmap[s[del_idx]]
            # move left pointer of the slidewindow
            left = del_idx + 1   
        print(right,left)
        max_len = max(max_len, right - left)
        print("max_len",max_len)
    
    return max_len
    

def lengthofchar(strs):
    count = 0; 
    flag = False; 
    length = len(strs)-1; 
    while(length != 0): 
        if(strs[length] == ' '): 
            return count
        else: 
            count += 1; 
        length -= 1; 
    return count;


def editDistance(str1,str2,m,n):
    if m==0:
        print("m",n)
        return n
    if n==0:
        print("n",m)
        return m
    if str1[m-1] == str2[n-1]:
        print("inside if",str1[m-1],str2[n-1])
        return editDistance(str1,str2,m-1,n-1)
    else:
        return 1 + min(editDistance(str1,str2,m,n-1),
                       editDistance(str1,str2,m-1,n),
                       editDistance(str1,str2,m-1,n-1))

# Edit Distance using Dynamic Programming
                    
def editDistanceDP(str1,str2,m,n):
    dp = [[0 for i in range(n+1)] for j in range(m+1)]
    for i in range(m+1):
        for j in range(n+1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 +  min(dp[i][j-1],
                                    dp[i-1][j])
    return dp[m][n]

def minimumSteps(num):
    dp = [0 for i in range(num+1)]
    for i in range(num+1):
        dp[i] = -1
    return getMinSteps(num,dp)

def getMinSteps(num,dp):
    
    if num==1:
        return 0
    if dp[num] != -1:
        return dp[num]
    res = getMinSteps(num-1,dp)
    if num%2 == 0:
        res = min(res, getMinSteps(num//2,dp))
    if num%3 == 0:
        res = min(res, getMinSteps(num//3,dp))
    dp[num] = 1+res
    print(dp)
    return dp[num]

def getMinTabulation(num):
    table = [0] * (num+1)
    for i in range(num + 1):
        table[i] = num - i
    for i in range(num,0,-1):
        if i % 2 == 0:
            table[i//2] = min(table[i]+1, table[i//2])
        elif i% 3 == 0:
            table[i//3] = min(table[i]+1, table[i//3])
    return table[1]
        
def validParenthesis(s):
    openStack = ["[","(","{"]
    closeStack = ["]",")","}"]
    resStack = []
    for i in (s):
        if i in openStack:
            resStack.append(i)
        if i in closeStack:
            pos = closeStack.index(i)
            if len(resStack)>0 and openStack[pos]==resStack[len(resStack)-1]:
                resStack.pop()
    if len(resStack) == 0:
        return "Balanced"
    else:
        return "Unbalanced"

def reverseString(s):
    left, right = 0, len(s) - 1
    while left < right:
        s[left], s[right] = s[right], s[left]
        left, right = left + 1, right - 1



"""ARRAY SECTION"""


def twoSum(num, target):
    tempDict = {num[0]:0}
    for i in range(1,len(num)):
        checkNum = target-num[i]
        if(checkNum in tempDict.keys()):
            return[tempDict[checkNum],i]
            print("Here is the output", num[tempDict[checkNum]], num[i])
        else:
            tempDict[num[i]]=i

def threeSumRev(arr):
    n = len(arr)
    if n < 3: return []
    found = False
    Finalres = []
    for i in range(n-1):
        res = set()
        for j in range(i+1, n):
            x = - (arr[i] + arr[j])
            if x in res:
                Finalres.append([x, arr[i], arr[j]])
                found = True 
            else:
                res.add(arr[j])
    if found == False:
        return " NO Match Triplets Found"
    else:
        return list(set(tuple(sorted(sub)) for sub in Finalres))

def fourSum(arr,Target):
    arr.sort()
    n = len(arr)
    res = set()
    for i in range (n-3):
        print(i)
        for j in range(i+1,n-2):
            l = j+1
            r = n-1
            while l<r:
                if (arr[i]+arr[j]+arr[l]+arr[r] == Target):
                    res.add(tuple(sorted([arr[i],arr[j],arr[l],arr[r]])))
                    l += 1
                    r -= 1
                elif (arr[i]+arr[j]+arr[l]+arr[r] < Target):
                    l+=1
                else:
                    r -= 1
    return res

"""
Given a sorted array nums, remove the duplicates in-place such that each element appear only once and return the new length.

Do not allocate extra space for another array, you must do this by modifying the input array in-place with O(1) extra memory.

Example 1:

Given nums = [1,1,2],

Your function should return length = 2, with the first two elements of nums being 1 and 2 respectively.

It doesn't matter what you leave beyond the returned length.
"""

def removeDuplicates(nums):
    i = 0
    while i < len(nums)-1:
        if nums[i] == nums[i+1]:
            del nums[i]
        else:
            i += 1
    return len(nums), nums

def removeValues(num, val):
    n = len(num)
    for i in range(n):
        if num[i] == val:
            n -= 1
    return n

"""
Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.

(i.e., [0,1,2,4,5,6,7] might become [4,5,6,7,0,1,2]).

You are given a target value to search. If found in the array return its index, otherwise return -1.

You may assume no duplicate exists in the array.

Your algorithm's runtime complexity must be in the order of O(log n).

Example 1:

Input: nums = [4,5,6,7,0,1,2], target = 0
Output: 4

"""

def search(num, target):
    start, end = 0, len(num)-1
    while start <= end:
        mid = start+(end-start) // 2
        if num[mid] == target:
            return mid
        elif num[mid] >=num[start]:
            if target >= num[start] and target < num[mid]:
                end = mid-1
            else:
                start = mid+1
        else:
            if target <= num[end] and target > num[mid]:
                start = mid-1
            else:
                end = mid - 1
    return -1

"""
Given an array of integers nums sorted in ascending order, find the starting and ending position of a given target value.

Your algorithm's runtime complexity must be in the order of O(log n).

If the target is not found in the array, return [-1, -1].

Example 1:

Input: nums = [5,7,7,8,8,10], target = 8
Output: [3,4]
"""
def searchRange(num, target, key):
    lo = 0
    hi = len(num)
    while lo < hi:
        mid = (lo + hi) // 2
        if num[mid] > target or (key and num[mid] == target):
            hi = mid
        else:
            lo = mid + 1
    return lo

def searchFinding(num,target):
    leftidx = searchRange(num,target,True)
    
    if leftidx == len(num) or num[leftidx] != target:
        return [-1,-1]
    
    return (leftidx, searchRange(num, target, False)-1)

"""
Given a sorted array and a target value, return the index if the target is found. If not, return the index where it would be if it were inserted in order.

You may assume no duplicates in the array.

Example 1:

Input: [1,3,5,6], 5
Output: 2
"""

def searchInsert(nums, target):
    lo = 0
    hi = len(nums)
    while lo <= hi:
        mid = (lo+hi)//2
        if nums[mid] == target:
            return mid
        elif nums[mid] <target:
            lo = mid + 1
        else:
            hi = mid -1

""" To solve the above with the later part"""

def searchInsertright(nums,target):
    n=len(nums)
    i=n-1
    while i>=0 and nums[i]>=target:
        i-=1
    return i+1

"""
42. Trapping Rain Water

Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it is able to trap after raining.


The above elevation map is represented by array [0,1,0,2,1,0,1,3,2,1,2,1]. In this case, 6 units of rain water (blue section) are being trapped. Thanks Marcos for contributing this image!

Example:

Input: [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6

"""

def trap(height):
    areas = 0
    max_l = max_r = 0
    l = 0
    r = len(height)-1
    while l < r:
        if height[l] < height[r]:
            if height[l] > max_l:
                max_l = height[l]
            else:
                areas += max_l - height[l]                
            l +=1
        else:
            if height[r] > max_r:
                max_r = height[r]
            else:
                areas += max_r - height[r]
            r -=1
    return areas

"""
Given an unsorted integer array, find the smallest missing positive integer.

Example 1:

Input: [1,2,0]
Output: 3
"""

def firstMissingPositive(nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    n = len(nums)
    
    # Base case.
    if 1 not in nums:
        return 1
    
    # nums = [1]
    if n == 1 and nums[0] == 1:
        return 2
    
    # Replace negative numbers, zeros,
    # and numbers larger than n by 1s.
    # After this convertion nums will contain 
    # only positive numbers.
    for i in range(n):
        if nums[i] <= 0 or nums[i] > n:
            nums[i] = 1
    
    # Use index as a hash key and number sign as a presence detector.
    # For example, if nums[1] is negative that means that number `1`
    # is present in the array. 
    # If nums[2] is positive - number 2 is missing.
    for i in range(n): 
        a = abs(nums[i])
        # If you meet number a in the array - change the sign of a-th element.
        # Be careful with duplicates : do it only once.
        if a == n:
            nums[0] = - abs(nums[0])
            print("Inside If",nums)
        else:
            nums[a] = - abs(nums[a])
            print("Inside Else",nums)
        
    # Now the index of the first positive number 
    # is equal to first missing positive.
    for i in range(1, n):
        if nums[i] > 0:
            print("inside for i third", i)
            return i
    
    if nums[0] > 0:
        return n
        
    return n + 1

"""
Given a set of non-overlapping intervals, insert a new interval into the intervals (merge if necessary).

You may assume that the intervals were initially sorted according to their start times.

Example 1:

Input: intervals = [[1,3],[6,9]], newInterval = [2,5]
Output: [[1,5],[6,9]]
"""
def insertInterval(arr,interval):
    newSet = []
    go = True
    for v in arr:
        if go:
            if v[0] <= interval[0] <= v[1] or interval[0] <= v[0] and interval[1] >= v[1]:
                newSet.append(((min(v[0],interval[0])), max(v[1],interval[1])))
                go = False
            else:
                newSet.append(v)
        else:
            newSet = insertInterval(newSet,v)
    if go:
        newSet.append(interval)
    return sorted(newSet)

"""
Given a collection of intervals, merge all overlapping intervals.

Example 1:

Input: [[1,3],[2,6],[8,10],[15,18]]
Output: [[1,6],[8,10],[15,18]]
Explanation: Since intervals [1,3] and [2,6] overlaps, merge them into [1,6].

"""

def mergeIntervals(arr):
    newSet =[]
    i = 0
    while i <= len(arr)-1:
        if i == len(arr)-1:
            newSet.append(arr[i])
            break
        elif arr[i][0] <= arr[i+1][0] <= arr[i][1] or arr[i+1][0] <= arr[i][0]:
            newSet.append([(min(arr[i][0],arr[i+1][0])), max(arr[i][1], arr[i+1][1])])
            i += 2
        else:
            newSet.append(arr[i])
            i += 1
    return newSet

"""
62. Unique Paths
Medium

2684

187

Add to List

Share
A robot is located at the top-left corner of a m x n grid (marked 'Start' in the diagram below).

The robot can only move either down or right at any point in time. The robot is trying to reach the bottom-right corner of the grid (marked 'Finish' in the diagram below).

How many possible unique paths are there?
Input: m = 3, n = 2
Output: 3
Explanation:
From the top-left corner, there are a total of 3 ways to reach the bottom-right corner:
1. Right -> Right -> Down
2. Right -> Down -> Right
3. Down -> Right -> Right
"""

def uniquePath(m,n):
    d = [[1] *n for _ in range(m)]
    print(d)
    for col in range(1,m):
        for row in range(1,n):
            d[col][row] = d[col-1][row] + d[col][row-1]
            
    return d[m-1][n-1]


"""
Given a m x n grid filled with non-negative numbers, find a path from top left to bottom right which minimizes the sum of all numbers along its path.

Note: You can only move either down or right at any point in time.

Example:

Input:
[
  [1,3,1],
  [1,5,1],
  [4,2,1]
]
Output: 7
Explanation: Because the path 1→3→1→1→1 minimizes the sum.
"""

def minPathSum(grid):
    if not grid or not grid[0]:
        return 0
    m = len(grid)
    n = len(grid[0])
    result = [[0 for j in range(n)] for i in range(m)]
    for i in range(m-1,-1,-1):
        for j in range(n-1,-1,-1):
            print(i,j,result)
            if i == m-1 and j == n-1:
                result[i][j] = grid[i][j]
                continue
            bottom = float('inf') if i == m-1 else result[i+1][j]
            right = float('inf') if j == n-1 else result[i][j+1]
            result[i][j] = min(grid[i][j]+bottom, grid[i][j]+right)
    print(result)
    return result[0][0]

"""
Given two words (beginWord and endWord), and a dictionary's word list, find all shortest transformation sequence(s) from beginWord to endWord, such that:

Only one letter can be changed at a time
Each transformed word must exist in the word list. Note that beginWord is not a transformed word.
Note:

Return an empty list if there is no such transformation sequence.
All words have the same length.
All words contain only lowercase alphabetic characters.
You may assume no duplicates in the word list.
You may assume beginWord and endWord are non-empty and are not the same.
Example 1:

Input:
beginWord = "hit",
endWord = "cog",
wordList = ["hot","dot","dog","lot","log","cog"]

Output:
[
  ["hit","hot","dot","dog","cog"],
  ["hit","hot","lot","log","cog"] BFS and DFS
]
"""

def wordLadder(beginWord,endWord,wordList):
    if beginWord not in wordList or endWord not in wordList or len(wordList) == 0 or beginWord == None or endWord == None:
        return []
    

"""
Given a list of words and two words word1 and word2, return the shortest distance between these two words in the list.

Example:
Assume that words = ["practice", "makes", "perfect", "coding", "makes"].

Input: word1 = “coding”, word2 = “practice”
Output: 3
"""

def wordDistance(words, word1, word2):
    indx = []
    for i in range(len(words)):
        if words[i]== word1 or words[i]==word2:
            if i not in indx and len(indx) != 0:
                result = abs(indx[0]-i)
                break
            else:
                indx.append(i)
    return result

def wordDistance2(words,word1,word2):
    #word_map = defaultdict(list)
    word_map = {}
    for idx,word in enumerate(words):
        if word not in word_map:
            word_map[word] = [idx]
        else:
            word_map[word].append(idx)
    print(word_map)
    res = float('inf')
    if word1!= word2:
        i=j=0
        l1 = word_map[word1]
        l2 = word_map[word2]
        n=len(l1)
        m=len(l2)
        while i<n and j<m:
            res = min(res, abs(l1[i]-l2[j]))
            if l1[i] > l2[j]:
                j+=1
            else:
                i+=1
    else:
        l1 = word_map[word1]
        for i in range(len(l1)):
            res = min(res, abs(l1[i+1]-l1[i]))
    return res



def findStrobogrammatic(n):
    if not n:
        return ['']
    if n == 1:
        return ['0', '1', '8']
    return [ num for num in helper(n) if num[0] != '0']
        
def helper(n):
    if n == 1:
        return ['0', '1', '8']
    if n == 2:
        return ['00', '11', '69', '88', '96']
    inner = helper(n-2)
    print(inner)
    outer = helper(2)
    print(outer)
    return [o[0]+i+o[1] for o in outer for i in inner]


"""
The Tribonacci sequence Tn is defined as follows: 

T0 = 0, T1 = 1, T2 = 1, and Tn+3 = Tn + Tn+1 + Tn+2 for n >= 0.

Given n, return the value of Tn.

"""

def tribonacci(n):
    if n == 0:
        return 0
    elif n==1:
        return 1
    elif n==2:
        return 1
    return tribonacci(n-1) + tribonacci(n-2) + tribonacci(n-3) 
    
    
def tribonacci_dp(n):
    dp = [0 for i in range(n+1)]
    if n == 0 or n ==1:
        dp[n] = n
    elif n ==2:
        dp[n] = 1
    else:
        dp[n] = tribonacci_dp(n-1) + tribonacci_dp(n-2) + tribonacci_dp(n-3)
    return dp[n]
    
        
"""
Given the root node of a binary search tree, return the sum of values of all nodes with value between L and R (inclusive).

The binary search tree is guaranteed to have unique values.

 

Example 1:

Input: root = [10,5,15,3,7,null,18], L = 7, R = 15
Output: 32
"""

def rangeSumBST(root, L, R):
    if not root:
        return 0

    if R < root.val:
        return rangeSumBST(root.left, L, R)
    elif L > root.val:
        return rangeSumBST(root.right, L, R)
    else:
        return rangeSumBST(root.left, L, R) + root.val + rangeSumBST(root.right, L, R)

"""
Given an array with n objects colored red, white or blue, sort them in-place so that objects of the same color are adjacent, with the colors in the order red, white and blue.

Here, we will use the integers 0, 1, and 2 to represent the color red, white, and blue respectively.

Note: You are not suppose to use the library's sort function for this problem.

Example:

Input: [2,0,2,1,1,0]
Output: [0,0,1,1,2,2]
"""

def sortColors(nums):
    lo,mid,hi = 0,0,len(nums)
    while mid<hi:
        if nums[mid] < 1:
            nums[lo], nums[mid] = nums[mid],nums[lo]
            lo += 1
            mid += 1
        elif nums[mid] > 1:
            hi -= 1
            print("Inside ElseIF",nums[hi], hi)
            nums[mid], nums[hi] = nums[hi], nums[mid]
        else:
            mid += 1
        print(lo,mid,hi,nums)
    return nums

def rmDuplicates(arr):
    res = []
    [res.append(x) for x in arr if x not in res]
    print(res)

def reverseString(string):
    print(string[::-1])
    return ' ' .join(words[::-1] for words in string.split(" "))

def reverseStringSpecial(strg):
    myList = list(strg)
    leftstring = 0 
    rightString = len(strg)-1
    while leftstring < rightString:
        if not strg[leftstring].isalpha():
            leftstring += 1
        elif not strg[rightString].isalpha():
            rightString -= 1
        else:
            myList[leftstring], myList[rightString] = myList[rightString], myList[leftstring]
            leftstring += 1
            rightString -= 1
    return ''.join(myList)

            
def maximumGap(num):
    l = len(num)
    num.sort()
    if l < 2:
        return 0
    m = num[1]-num[0]
    if l == 2:
        return m
    for i in range(l-1):
        if num[i+1]-num[i] <= m:
            continue
        else:
            m = num[i+1]-num[i]
    return m



"""
220. Contains Duplicate III

Given an array of integers, find out whether there are two distinct indices i and j in the array such that the absolute difference between nums[i] and nums[j] is at most t and the absolute difference between i and j is at most k.

Example 1:

Input: nums = [1,2,3,1], k = 3, t = 0
Output: true
"""

def containsNearbyAlmostDuplicate(nums,k,t):
    if t == 0 and len(set(nums)) == len(nums):
        return False
    l,r = 0,0
    while r<len(nums):
        if r-l>k:
            l += 1
        for i in nums[l:r]:
            print(i,l,r)
            if abs(nums[r]-i)<=t:
                return True
        r +=1 
    return False



"""
853. Car Fleet

N cars are going to the same destination along a one lane road.  The destination is target miles away.

Each car i has a constant speed speed[i] (in miles per hour), and initial position position[i] miles towards the target along the road.

A car can never pass another car ahead of it, but it can catch up to it, and drive bumper to bumper at the same speed.

The distance between these two cars is ignored - they are assumed to have the same position.

A car fleet is some non-empty set of cars driving at the same position and same speed.  Note that a single car is also a car fleet.

If a car catches up to a car fleet right at the destination point, it will still be considered as one car fleet.


How many car fleets will arrive at the destination?

 

Example 1:

Input: target = 12, position = [10,8,0,5,3], speed = [2,4,1,1,3]
Output: 3
Explanation:
The cars starting at 10 and 8 become a fleet, meeting each other at 12.
The car starting at 0 doesn't catch up to any other car, so it is a fleet by itself.
The cars starting at 5 and 3 become a fleet, meeting each other at 6.
Note that no other cars meet these fleets before the destination, so the answer is 3.
"""
def carFleet(target,position,speed):
    n = len(speed)
    arrivals = [(target-position[i])/speed[i] for i in range(n)]
    print(arrivals)
    data = sorted(list(zip(arrivals, position)), reverse=True)
    print(data)
    count = 0
    curr_max = -1
    for ch in data:
        print(ch[1])
        if ch[1]>curr_max:
            curr_max=ch[1]
            count+=1
    return count
    
"""
969. Pancake Sorting
Given an array A, we can perform a pancake flip: We choose some positive integer k <= A.length, then reverse the order of the first k elements of A.  We want to perform zero or more pancake flips (doing them one after another in succession) to sort the array A.

Return the k-values corresponding to a sequence of pancake flips that sort A.  Any valid answer that sorts the array within 10 * A.length flips will be judged as correct.

Example 1:

Input: [3,2,4,1]
Output: [4,2,4,3]
Explanation: 
We perform 4 pancake flips, with k values 4, 2, 4, and 3.
Starting state: A = [3, 2, 4, 1]
After 1st flip (k=4): A = [1, 4, 2, 3]
After 2nd flip (k=2): A = [4, 1, 2, 3]
After 3rd flip (k=4): A = [3, 2, 1, 4]
After 4th flip (k=3): A = [1, 2, 3, 4], which is sorted. 
"""
def panCake(A):
    ans = []

    N = len(A)
    B = sorted(range(1, N+1), key = lambda i: -A[i-1])
    print(B,N)
    for i in B:
        for f in ans:
            print("inside For",i,f)
            if i <= f:
                i = f+1 - i
        ans.extend([i, N])
        print("Answer",ans)
        N -= 1
    return ans
def pancake(A):
    end = len(A)
    ans =[]
def flip(idx):
    
    A[:idx+1] = A[:idx+1][::-1]
    for i in range(len(A)):
        max_idx = A[:end].index(max(A[:end]))
        if max_idx!=end:
            ans.append(max_idx+1)
            ans.append(end)
            flip(max_idx)
            flip(end-1)
        end-=1
    return ans
    

        
"""
973. K Closest Points to Origin
We have a list of points on the plane.  Find the K closest points to the origin (0, 0).

(Here, the distance between two points on a plane is the Euclidean distance.)

You may return the answer in any order.  The answer is guaranteed to be unique (except for the order that it is in.)
Example 1:
Input: points = [[1,3],[-2,2]], K = 1
Output: [[-2,2]]
Explanation: 
The distance between (1, 3) and the origin is sqrt(10).
The distance between (-2, 2) and the origin is sqrt(8).
Since sqrt(8) < sqrt(10), (-2, 2) is closer to the origin.
We only want the closest K = 1 points from the origin, so the answer is just [[-2,2]].
"""

def kClosest(points, K):
    points.sort(key = lambda P: P[0]**2 + P[1]**2)
    print(points)
    return points[:K]

"""
976. Largest Perimeter Triangle

Given an array A of positive lengths, return the largest perimeter of a triangle with non-zero area, formed from 3 of these lengths.

If it is impossible to form any triangle of non-zero area, return 0.

Example 1:

Input: [2,1,2]
Output: 5
"""
def largestPerimeter(A):
    A.sort()
    for i in range(len(A) - 3, -1, -1):
        print(i)
        if A[i] + A[i+1] > A[i+2]:
            return A[i] + A[i+1] + A[i+2]
    return 0

def splitLines(quotes,toys):
    myDict = {}
    for lines in quotes:
        words = lines.split()
        for w in words:
            if w in toys and w in myDict:
                myDict[w] += 1
            elif w in toys:
                myDict[w] = 1 
    sort_Orders = list(sorted(myDict.items(), key = lambda x:x[1], reverse = True))[:2]
    print(sort_Orders)
    result = list(sort_Orders)[:2]
    r = [i[0] for i in result]
    #print(r)


def largestItemAssociation(itemAssociation):
    # WRITE YOUR CODE HERE
    result = []
    for i in range(1,len(itemAssociation)-1):
        if itemAssociation[i][1] == itemAssociation[i+1][0]:
            result.extend([itemAssociation[i][0],itemAssociation[i+1][0],itemAssociation[i+1][1]])
        elif itemAssociation[i][0] == itemAssociation[i-1][1]:
            result[-1].extend([itemAssociation[i][0],itemAssociation[i+1][0],itemAssociation[i+1][1]])
    return result

"""
1057. Campus Bikes
On a campus represented as a 2D grid, there are N workers and M bikes, with N <= M. Each worker and bike is a 2D coordinate on this grid.
Our goal is to assign a bike to each worker. Among the available bikes and workers, we choose the (worker, bike) pair with the shortest Manhattan distance between each other, and assign the bike to that worker. (If there are multiple (worker, bike) pairs with the same shortest Manhattan distance, we choose the pair with the smallest worker index; if there are multiple ways to do that, we choose the pair with the smallest bike index). We repeat this process until there are no available workers.
The Manhattan distance between two points p1 and p2 is Manhattan(p1, p2) = |p1.x - p2.x| + |p1.y - p2.y|.
Return a vector ans of length N, where ans[i] is the index (0-indexed) of the bike that the i-th worker is assigned to.

Example 1:
Input: workers = [[0,0],[2,1]], bikes = [[1,2],[3,3]]
Output: [1,0]
Explanation: 
Worker 1 grabs Bike 0 as they are closest (without ties), and Worker 0 is assigned Bike 1. So the output is [1, 0].
"""


def assignBikes(workers, bikes):
    def dist(p1,p2):
        return abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])
    
    N = len(workers)
    M = len(bikes)
    distance = []
    for i in range(N):
        for j in range(M):
            distance.append((dist(workers[i], bikes[j]),i,j))
    distance.sort()
    print("Distance",distance)
    assigned = set()
    result = {}
    for dist, worker,bike in distance:
        if bike not in assigned:
            if worker not in result:
                result[worker] = bike
                assigned.add(bike)
    result = {k: result[k] for k in sorted(result)}
    return result.values()
    
ASCII_SIZE = 256
  
def getMaxOccuringChar(str): 
    # Create array to keep the count of individual characters 
    # Initialize the count array to zero 
    count = [0] * ASCII_SIZE 
  
    # Utility variables 
    max = -1
    c = '' 
  
    # Traversing through the string and maintaining the count of 
    # each character 
    for i in str: 
        count[ord(i)]+=1; 
  
    for i in str: 
        if max < count[ord(i)]: 
            max = count[ord(i)] 
            c = i 
  
    return c 
def findFulcrum(numbers):
    for mid in range(len(numbers)):
        left_total = sum(numbers[:mid])
        right_total = sum(numbers[mid+1:])
        print(left_total,right_total)
        if left_total == right_total:
            return mid
    return -1


def permuteRec(string, n, index = -1, curr = ""): 
    if index == n: 
        return
    if len(curr) > 0:
        print(curr) 
    for i in range(index + 1, n):
        #print("Value of i", i,index)
        curr += string[i]
        #print("Inside For Loop",curr)
        permuteRec(string, n, i, curr) 
        curr = curr[:len(curr) - 1]
        #print("After Recurssion", curr)

# Generates power set in lexicographic order 
def powerSet(string):
    string = ''.join(sorted(string))
    #print(string)
    permuteRec(string, len(string)) 

# Driver Code 
if __name__ == "__main__": 
	string = "cab"
	powerSet(string) 

"""
1305. All Elements in Two Binary Search Trees

Given two binary search trees root1 and root2.
Return a list containing all the integers from both trees sorted in ascending order.

Example 1:

Input: root1 = [2,1,4], root2 = [1,0,3]
Output: [0,1,1,2,3,4]
"""
def getAllElementsTwoBinaryTreeSorted(TreeNode1, TreeNode2):
    
    def inOrder(r):
        return inOrder(r.left) + [r.val] + inOrder(r.right) if r else []
    return sorted(inOrder(TreeNode1) + inOrder(TreeNode2))


# Python program to check whether it is possible to make 
# string palindrome by removing one character 

# Utility method to check if substring from 
# low to high is palindrome or not. 
def isPalindrome(string,low,high):
    while low < high:
        if string[low] != string[high]:
            return False
        low += 1
        high -= 1
    return True

def possiblepalinByRemovingOneChar(string):
    
    low = 0
    high = len(string) - 1
    while low < high:
        if string[low] == string[high]:
            low += 1
            high -= 1
        else:
            if isPalindrome(string, low + 1, high):
                return low
            if isPalindrome(string, low, high - 1):
                return high
            
            return -1
    return -2

# =============================================================================
# # Driver Code 
# if __name__ == "__main__": 
# 
# 	string = "raacecar"
# 	idx = possiblepalinByRemovingOneChar(string) 
# 
# 	if idx == -1: 
# 		print("Not possible") 
# 	elif idx == -2: 
# 		print("Possible without removig any character") 
# 	else: 
# 		print("Possible by removing character at index", idx) 
# =============================================================================

#Root of a number
def root(x, n):
  if x == 0:
    return 0
  lo = 0
  hi = max(1,x)
  mid = (hi+lo)/2
  while mid-lo >= 0.001:
    mid = (hi+lo)/2
    if mid ** n > x:
      hi = mid
    elif (mid**n) < x:
      lo = mid
    else:
      break
  return mid  

"""
1244. Design A Leaderboard
Design a Leaderboard class, which has 3 functions:

addScore(playerId, score): Update the leaderboard by adding score to the given player's score. If there is no player with such id in the leaderboard, add him to the leaderboard with the given score.
top(K): Return the score sum of the top K players.
reset(playerId): Reset the score of the player with the given id to 0 (in other words erase it from the leaderboard). It is guaranteed that the player was added to the leaderboard before calling this function.
Initially, the leaderboard is empty.
"""

class Leaderboard:
    def __init__(self):
        self.player = {}
    
    def addScore(self,playerID, score):
        if playerID in self.player:
            self.player[playerID] += score
        else:
            self.player[playerID] = score
    
    def top(self,k):
        scoreinsorted = sorted(self.player.values(), reverse=True)
        return sum(scoreinsorted[i] for i in range(k))
    
    def reset(self,playerID):
        self.player[playerID] = 0

"""
1333. Filter Restaurants by Vegan-Friendly, Price and Distance

Given the array restaurants where  restaurants[i] = [idi, ratingi, veganFriendlyi, pricei, distancei]. 
You have to filter the restaurants using three filters.

The veganFriendly filter will be either true (meaning you should only include restaurants with 
veganFriendlyi set to true) or false (meaning you can include any restaurant). 
In addition, you have the filters maxPrice and maxDistance which are the maximum value 
for price and distance of restaurants you should consider respectively.

Return the array of restaurant IDs after filtering, ordered by rating from highest to lowest. 
For restaurants with the same rating, order them by id from highest to lowest. 
For simplicity veganFriendlyi and veganFriendly take value 1 when it is true, and 0 when it is false.
"""


def filterRestaurant(restaurants, veganFriendly,maxPrice,maxDistance):
    canditates = []
    for rest in restaurants:
        i,r,vf,p,dist = rest[0],rest[1],rest[2],rest[3],rest[4]
        if veganFriendly == 1:
            if vf==1 and p<=maxPrice and dist <= maxDistance:
                canditates.append(rest)
                continue
            elif veganFriendly == 0:
                if p <= maxPrice and dist <= maxDistance:
                    canditates.append(rest)
    canditates.sort(key = lambda x:(x[1],x[0]), reverse = True)
    return [x[0] for x in canditates]

"""
1452. People Whose List of Favorite Companies Is Not a Subset of Another List

Given the array favoriteCompanies where favoriteCompanies[i] is the list of favorites companies for the ith person (indexed from 0).

Return the indices of people whose list of favorite companies is not a subset of any other list of favorites companies. You must return the indices in increasing order.


Example 1:

Input: favoriteCompanies = [["leetcode","google","facebook"],["google","microsoft"],["google","facebook"],["google"],["amazon"]]
Output: [0,1,4] 
Explanation: 
Person with index=2 has favoriteCompanies[2]=["google","facebook"] which is a subset of favoriteCompanies[0]=["leetcode","google","facebook"] corresponding to the person with index 0. 
Person with index=3 has favoriteCompanies[3]=["google"] which is a subset of favoriteCompanies[0]=["leetcode","google","facebook"] and favoriteCompanies[1]=["google","microsoft"]. 
Other lists of favorite companies are not a subset of another list, therefore, the answer is [0,1,4].
"""
def peopleIndexes(favoriteCompanies):
    new, d, cnt = [], {}, 1
    for i in favoriteCompanies:
        for j in i:
            if j not in d:
                d[j] = cnt
                cnt += 1
        new += set([d[j] for j in i]),
        print(new,d)
    
    res = []
    for i in range(len(new)):
        print(new[i])
        if sum([new[i] <= j for j in new]) == 1:  # to check if its superset
            res += i,
        
    return res

def latticePath(m, n):
    if m==0 and n ==0:
        return 1
    count = 0
    if m>0:
        count += latticePaths(m-1,n)
    if n>0:
        count += latticePaths(m,n-1)
    return count

dp = {(0,1):1,(1,0):1}
def lattice_paths(m,n):
    if (m,n) in dp:
        return dp[(m,n)]
    count = 0
    if m==0 or n==0:
        count = 1
    else:
        count = lattice_paths(m-1,n) + lattice_paths(m, n-1)
    dp[(m,n)] = count
    return count

def coinChange(S, m, n ): 
    # If n is 0 then there is 1 
    # solution (do not include any coin) 
    if (n == 0): 
        return 1
    # If n is less than 0 then no 
    # solution exists 
    if (n < 0): 
        return 0; 
  
    # If there are no coins and n 
    # is greater than 0, then no 
    # solution exist 
    if (m <=0 and n >= 1): 
        return 0
  
    # count is sum of solutions (i)  
    # including S[m-1] (ii) excluding S[m-1] 
    return coinChange( S, m - 1, n ) + coinChange( S, m, n-S[m-1] )

# Dynamic Programming Python implementation of Coin  
# Change problem 
def coinChangeDP(S, m, n): 
  
    # We need n+1 rows as the table is constructed  
    # in bottom up manner using the base case 0 value 
    # case (n = 0) 
    table = [[0 for x in range(m)] for x in range(n+1)] 
  
    # Fill the entries for 0 value case (n = 0) 
    for i in range(m): 
        table[0][i] = 1
  
    # Fill rest of the table entries in bottom up manner 
    for i in range(1, n+1): 
        for j in range(m): 
  
            # Count of solutions including S[j] 
            x = table[i - S[j]][j] if i-S[j] >= 0 else 0
  
            # Count of solutions excluding S[j] 
            y = table[i][j-1] if j >= 1 else 0
  
            # total count 
            table[i][j] = x + y 
  
    return table[n][m-1] 
  
# Driver program to test above function 
arr = [1, 2, 3] 
m = len(arr) 
n = 4
print(coinChangeDP(arr,m,n))

############################################################
###############  DO NOT TOUCH TEST BELOW!!!  ###############
############################################################

# custom assert function to handle tests
# input: count {List} - keeps track out how many tests pass and how many total
#        in the form of a two item array i.e., [0, 0]
# input: name {String} - describes the test
# input: test {Function} - performs a set of operations and returns a boolean
#        indicating if test passed
# output: {None}
def expect(count, name, test):
    if (count is None or not isinstance(count, list) or len(count) != 2):
        count = [0, 0]
    else:
        count[1] += 1

    result = 'false'
    error_msg = None
    try:
        if test():
            result = ' true'
            count[0] += 1
    except Exception as err:
        error_msg = str(err)

    print('  ' + (str(count[1]) + ')   ') + result + ' : ' + name)
    if error_msg is not None:
        print('       ' + error_msg + '\n')

print('Lattice Paths Tests')
test_count = [0, 0]


def test():
    example = lattice_paths(2, 3)
    return example == 10


expect(test_count, 'should work for a 2 x 3 lattice', test)


def test():
    example = lattice_paths(3, 2)
    return example == 10


expect(test_count, 'should work for a 3 x 2 lattice', test)


def test():
    example = lattice_paths(0, 0)
    return example == 1


expect(test_count, 'should work for a 0 x 0 lattice', test)


def test():
    example = lattice_paths(10, 10)
    return example == 184756


expect(test_count, 'should work for a 10 x 10 lattice (square input)', test)

def test():
    example = lattice_paths(20, 15)
    return example == 3247943160


expect(test_count, 'work for a 20 x 15 lattice (large input)', test)

print('PASSED: ' + str(test_count[0]) + ' / ' + str(test_count[1]) + '\n\n')

# Longest Palindrome
def longestPalindrome(string):
    
  def helper(seq, i, j):
    if i==j:
      return 1
    if seq[i] == seq[j] and i+1 ==j:
      return 2
    if seq[i] == seq[j]:
      return helper(seq , i+1,j-1) + 2
    return max(helper(seq,i,j-1), helper(seq,i+1,j))
  return helper(string,0,len(string)-1)

print(longestPalindrome("vtvvv"))


def knapsack(wt,val,W):
    n = len(val)
    memo = [[-1 for x in range(W+1)] for y in range(n+1)]
    def knapsackHelper(wt,val,W,n):
        if n ==0 or W == 0:
            return 0
        if memo[n][W] != -1:
            return memo[n][W]
        if wt[n-1] <= W:
            memo[n][W] = max(val[n-1] + knapsackHelper(wt,val,W-wt[n-1],n-1),knapsackHelper(wt,val,W,n-1))
            return memo[n][W]
        elif wt[n-1] > W:
            memo[n][W] = knapsackHelper(wt,val,W,n-1)
            return memo[n][w]

        
    
    print(knapsackHelper(wt,val,W,n))


def max_consecutive_sum(lst):
    maxSum = 0
    movingWin = 0
    for i in range(len(lst)):
        movingWin = movingWin + lst[i]
        if movingWin > maxSum:
            maxSum = movingWin
        elif movingWin < 0:
            movingWin = 0
    return maxSum

def bit_flip(lst, m):
    
    n = len(lst)
    WL= WR = 0
    bestL = bestWindow = 0
    zeroCount = 0
    while WR<n:
        print(WR,WL)
        if zeroCount <= m:
            if lst[WR] == 0:
                zeroCount += 1
            WR += 1
        if zeroCount > m:
            if lst[WL] == 0:
                zeroCount -= 1
            WL += 1
        if WR- WL > bestWindow and zeroCount <= m:
            bestWindow = WR-WL
            bestL = WL
    return bestWindow
    for i in range(0, bestWindow): 
        if lst[bestL + i] == 0: 
            print (bestL + i, end = " ") 
    return bestWindow, bestL



def mergeArrays(a, b):
    n = len(a)
    m = len(b)
    i = 0
    j = 0
    result = []
    while i < n and j < m:
        if a[i] < b[j]:
            result.append(a[i])
            i += 1
        else:
            result.append(b[j])
            j += 1  
    return result  + a[i:] + b[j:]    

def dnaComplement(s):
    if len(s)==0:
        return ""
    revs = s[::-1]
    res = ""
    dic = {"A":"T","C":"G","G":"C","T":"A"}
    for i in revs:
        res += dic[i]
    return res

def minWindow(s, t):
    targets = Counter(t)
    counts = Counter()
    diffs = len(t)
    j = 0
    result = s + ' '
    for i, c in enumerate(s):
        print(counts)
        counts[c] += 1
        if counts[c] <= targets[c]:
            diffs -= 1
        while diffs == 0:
            temp = s[j:i+1]
            if len(temp) < len(result):
                result = temp
            n = s[j]
            j += 1
            counts[n] -= 1
            if counts[n] < targets[n]:
                diffs += 1

    return result if len(result) <= len(s) else ''
            


def getMedian( ar1, ar2 , n): 
    i = 0 # Current index of i/p list ar1[] 
      
    j = 0 
    m1 = -1
    m2 = -1
    count = 0
    while count < n + 1:
        print("m1,m2",m1,m2,i,j,n,count)
        count += 1
        if i == n: 
            m1 = m2 
            m2 = ar2[0] 
            break
        # Below is to handle case where all  
        # elements of ar2[] are smaller than 
        # smallest(or first) element of ar1[] 
        elif j == n: 
            m1 = m2 
            m2 = ar1[0] 
            break
        # equals sign because if two  
        # arrays have some common elements  
        if ar1[i] <= ar2[j]: 
            m1 = m2 # Store the prev median 
            m2 = ar1[i] 
            i += 1
        else: 
            m1 = m2 # Store the prev median 
            m2 = ar2[j] 
            j += 1
    print(m1,m2)
    return (m1 + m2)/2
  
# Driver code to test above function 
ar1 = [1, 12, 15, 26, 38] 
ar2 = [2, 13, 17, 30, 45] 
n1 = len(ar1) 
n2 = len(ar2) 
if n1 == n2: 
    print("Median is ", getMedian(ar1, ar2, n1)) 
else: 
    print("Doesn't work for arrays of unequal size") 


"""
1. You need find a number of Path from start to dest
2. It cannot go diagnol
3. I can do a reccurssive call by moving up or right until I reach the destination
Anoher way is Dynamic Programming, I have a dict to track the visiting cells

Time = O(NxN)
Space = O(NxN)

"""

def num_of_paths_to_dest(n):
    memo = defaultdict(int)
    def helper(i,j):
        if i<0 or j < 0:
            return 0
        if i < j:
            memo[(i,j)] = 0
        if i == 0 and j == 0:
            return 1
        if (i,j) in memo:
            return memo[(i,j)]
        else:
            memo[(i,j)] += helper(i-1,j) + helper(i,j-1)
        return memo[(i,j)]
    return helper(n-1,n-1)

      
  
"""
numPaths = [[0 for i in range(n)] for j in range(n)]
  
  i = 0
  j = 0
  
  if n == 1:
    return 1
  
  numPaths[0][0] = 1
  
  for j in range(0, n):
    
    for i in range(j, n):
      
      if j-1 >= 0:
        numPaths[j][i] += numPaths[j-1][i]
        
      if i-1 >= 0:
        numPaths[j][i] += numPaths[j][i-1]
      
  
  return numPaths[n-1][n-1]
"""
import string

def wordCount(doc):
    result = []
    myDict = {}
    words = doc.strip(string.punctuation)
    words = words.split(" ")
    count = 0
    for w in words:
        word = w.strip(string.punctuation).lower()
        print(word)
        if word not in myDict:
            count += 1
            myDict[word] = [1, count]
        else:
            myDict[word][0] += 1   
    for values in myDict.keys():
        result.append([values,myDict[values][0],myDict[values][1]])
    result.sort(key=lambda i: (i[1],i[2]))
    print(result)
    #result.sort(key=lambda i: i[1])
    for i in range(len(result)):
        result[i].pop()
    result.sort(key=lambda i: i[1], reverse=True)
    print(result)
    
def toeplitzMatrix(matrix):
    row = len(matrix)
    col = len(matrix[0])
    
    for i in range(row-1):
        for j in range(col-1):
            if matrix[i][j] != matrix[i+1][j+1]:
                return False
    
    return True
        
def find_grants_cap(grantsArray, newBudget):
    sortedBudget = sorted(grantsArray)
    length = len(grantsArray)
    mean = newBudget / length
    for index, cap in enumerate(sortedBudget):
        if cap <= mean:
            overallocated = mean - cap
            remainderSum = (length -1) - index
            mean += overallocated / float(remainderSum)
    return mean

from collections import Counter

def uncommonFromSentences(A,B):
    count = {}
    for word in A.split():
        count[word] = count.get(word,0) + 1
    for word in B.split():
        count[word] = count.get(word,0) + 1
    
    return [word for word in count if count[word] == 1]

"""
162. Find Peak Element

A peak element is an element that is greater than its neighbors.

Given an input array nums, where nums[i] ≠ nums[i+1], find a peak element and return its index.

The array may contain multiple peaks, in that case return the index to any one of the peaks is fine.

You may imagine that nums[-1] = nums[n] = -∞.

Example 1:

Input: nums = [1,2,3,1]
Output: 2
Explanation: 3 is a peak element and your function should return the index number 2.
"""
def findPeakElement(nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    # always remove the smaller slope side
    
    left,right = 0,len(nums)-1
    while left+1 < right:
        mid = left + (right-left)//2
        if nums[mid]<nums[mid+1]:
            left = mid
        elif nums[mid]<nums[mid-1]:
            right = mid
        else:
            return mid
    
    if nums[left]>nums[right]:return left
    else: return right


def maxKilledEnemies(grid):
    if not grid:
        return 0
    ans = 0
    m,n = len(grid) , len(grid[0])
    kills = [[0 for c in range(n)] for r in range(m)]
    print(kills)
    hepler(grid, [0,m], [0,n], +1, kills, False)
    print(kills)
    hepler(grid, [0,m], [n-1,-1], -1, kills, False)
    print(kills)
    hepler(grid, [0,n], [0,m], +1, kills,True)
    print(kills)
    hepler(grid, [0,n], [m-1,-1], -1, kills,True)
    print(kills)
    for r in range(m):
        for c in range(n):
            ans = max(ans, kills[r][c]) 
    return ans
    
def hepler(grid, fixed, var, increment, kills, switch):
    [i,j] = fixed
    [p,q] = var
    for r in range(i,j):
        count = 0
        for c in range(p,q, increment):
            print(c)
            x,y = r,c
            if switch:
                x,y = c,r
            if grid[x][y] == 'E':
                count += 1
            if grid[x][y] == 'W':
                count = 0
            if grid[x][y] == '0':
                kills[x][y] += count
                
                

def addSum(a,b):
    return int(a)+int(b)

def addStrings(num1,num2):
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