# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 18:05:24 2020

@author: prchandr
"""

#Interview

"""
Question 1: From FB Phone Interview
	987. Vertical Order Traversal of a Binary Tree - FB
	Given a binary tree, return the vertical order traversal of its nodes values.
	For each node at position (X, Y), its left and right children respectively will be at positions (X-1, Y-1) and (X+1, Y-1).
	Running a vertical line from X = -infinity to X = +infinity, whenever the vertical line touches some nodes, we report the values of the nodes in order from top to bottom (decreasing Y coordinates).
	If two nodes have the same position, then the value of the node that is reported first is the value that is smaller.
	Return an list of non-empty reports in order of X coordinate.  Every report will have a list of values of nodes.
	 
	Example 1:
	    3
    /  \
   9   20
       / \
      15  7
Input: [3,9,20,null,null,15,7]
Output: [[9],[3,15],[20],[7]]
Explanation: 
Without loss of generality, we can assume the root node is at position (0, 0):
Then, the node with value 9 occurs at position (-1, -1);
The nodes with values 3 and 15 occur at positions (0, 0) and (0, -2);
The node with value 20 occurs at position (1, -1);
The node with value 7 occurs at position (2, -2).
"""

import collections
class Solution:
    def verticalTraversal(self, root):
        seen = collections.defaultdict(
                  lambda: collections.defaultdict(list))

        def dfs(node, x=0, y=0):
            if node:
                seen[x][y].append(node)
                print("seen", seen)
                dfs(node.left, x-1, y+1)
                dfs(node.right, x+1, y+1)

        dfs(root)
        ans = []

        for x in sorted(seen):
            report = []
            for y in sorted(seen[x]):
                report.extend(sorted(node.val for node in seen[x][y]))
            ans.append(report)

        return ans
"""
if __name__ == '__main__':
    VT = Solution()
    VT.verticalTraversal([3,9,20,15,7])

"""


####################### APPLE ############################################   
    
"""
Write a Python script that prints all lines containing "Apple" from a file called “file.txt". Please make a sample file of reasonable length to use with your program.
"""

def printApple():
    try:
        fileopen = open('file.txt', 'r') 
        Lines = fileopen.readlines() 
        for line in Lines:
            if 'apple' in line.lower():
                print(line)
    except FileNotFoundError:
        print("File not accessible")
            
"""

	• The file "results.txt" contains the following lines
			<PastedGraphic-2.png>

	Write a Python script that calculates the average time of all of the values.
"""

def avgTime():
    result = []
    try:
        fileopen = open('result.txt', 'r')
        lines = fileopen.readlines()
        for line in lines:
            if 'time' in line.lower():
                x = line.split()
                result.append(float(x[2]))
        average = sum(result)/len(result)
        return average
    except FileNotFoundError:
        print("File not accessible")
"""

• Write a Python script that creates a file called "Numbers.txt" and 
prints numbers from 1 to 100 in columns of ten separated by tabs. 
However, for any number ending in 5 or 0, print an asterisk instead of the number.

"""
import os
    
def printNumbers():
    if os.path.exists("Numbers.txt"):
        os.remove("Numbers.txt")
    with open('Numbers.txt','w') as file:
        for i in range(1,11):
            count = 0
            while count <= 99:
                if (i+count)%5==0 or (i+count)%10==0:
                    file.write('*\t')
                else:
                    file.write(f'{i+count}\t')
                count += 10
            file.write('\n')
    file.close()

"""
• There is a directory called "TestLogs". In that directory there are many different files. 
Files that end in .log contain information that we are interested in. 
There is a line in the file that begins with "DeviceInfo". That line contains 5 fields separated by tabs. 
Write a Python script that prints out the 3rd field in each file that ends in .log
"""
        
def testLogs():
    try:
        dirPath = "D:\Apple\TestLogs"
        for dirName in os.listdir(dirPath):
            if dirName.endswith('.log'):
                path = os.path.join(dirPath,dirName)
                with open(path, 'r') as file:
                    for line in file:
                        if line.startswith('DeviceInfo'):
                            x = line.split()
                            print("The 3rd Field contains: ", x[2])
    except FileNotFoundError:
        print("File not accessible")


"""
String S consisting of a and b
"""

def countidenticalString(S):
    count = 0
    runningCount = 1
    for i in range(len(S)-1):
        if S[i] != S[i+1]:
            runningCount = 1
        elif S[i] == S[i+1]:
            runningCount += 1
        
        if runningCount == 3:
            count += 1
            runningCount = 0
    
    return count

def minCostDelete(S,C):
    cost = 0
    i = 0
    while i < len(S)-1:
        print(S[i])
        if S[i] == S[i+1]:
            j = i+1
            temp = [] 
            while j > 0:
                if S[i] == S[i+1]:
                    temp.append(C[i])
                    i += 1
                else:
                    temp.append(C[i])
                    temp.sort()
                    cost += sum(temp[:i])
                    j = 0
            i = i + 1
            print("Forloop",temp,cost,i)
    return cost

def minCOst(S,C):
    cost = 0
    for i in range(len(S)):
        if S[i] == S[i+1]:
            temp = min(C[i],C[i+1])
            cost += temp
    return cost              


def result(T, R):
    group = 0
    score = 0
    tempScore = 0
    TotalScore = 0
    j = 0
    for i in range(len(T)):
        print("i",i)
        if j == T[i][-2]:
            continue
        group += 1
        if not T[i][-1].isdigit():
            if j != T[i][-2]:
                j = T[i][-2]
            tempCounter = i
            while j == T[i][-2]:
                print(j,T[i][-2],tempScore)
                if R[i] == "OK":
                    print("",tempScore)
                    tempScore += 1
                    tempCounter += 1
                    i += 1
                else:
                    tempScore -= 1    
                    i += 1
                if tempCounter != i:
                    tempScore = 0
        elif R[i] == "OK":
            print("score",i)
            score += 1
        print(T[i],score,tempScore, group)
        print(i)
    TotalScore = (score + tempScore) * 100 // group
    return TotalScore

"""
Find smallest number in unsorted array
"""
def getDifferentNumber(arr):
    i = 0
    while i < len(arr):
        print(i)
        if arr[i] != i:
            if arr[i] < len(arr):
                temp = arr[i]
                print(arr[i],arr[arr[i]])
                arr[i],arr[temp] = arr[temp], arr[i]
            else:
                i += 1
        else:
            i += 1
    
    for j in range(len(arr)):
        if j != arr[j]:
            return j
    
    return len(arr)      



"""

Text Justification

"""

def fullJustify(words,maxWidth):
    length = 0
    line = []
    ans = []
    for word in words:
        wLen = len(word)
        if length and length+wLen+1 > maxWidth:
            print(line)
            nw = len(line)
            total = maxWidth - (length - (nw-1))
            print(nw,total)
            if nw == 1:
                ans.append(line[0] + (' '*total))
            else:
                base, res = divmod(total, nw-1)
                for i in range(res):
                    line[i] += ' '
                    print("line",line)
                ans.append((' '*base).join(line))
                print("ans",ans)
            
            length = wLen
            line = [word]
        else:
            if line:
                length += 1 + wLen
            else:
                length += wLen
            line.append(word)
    
    if line:
        ans.append(' '.join(line) + ' '*(maxWidth - length))
    
    return ans


def maxNumberOfBalloons(text):
    balloon = 'BALLOON'
    myDict = {b:0 for b in balloon}
    
    for t in text:
        if t in balloon:
            myDict[t] += 1
    print(myDict)
    myDict['O'] //= 2
    myDict['L'] //= 2
    print(myDict)
    return min(myDict[d] for d in myDict)

"""
Maximum Width of Binary Tree
Given a binary tree, write a function to get the maximum width of the given tree. The maximum width of a tree is the maximum width among all levels.

The width of one level is defined as the length between the end-nodes (the leftmost and right most non-null nodes in the level, where the null nodes between the end-nodes are also counted into the length calculation.

It is guaranteed that the answer will in the range of 32-bit signed integer.

Example 1:

Input: 

           1
         /   \
        3     2
       / \     \  
      5   3     9 

Output: 4
Explanation: The maximum width existing in the third level with the length 4 (5,3,null,9).

"""

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def widthOfBinaryTree(self, root):
        maxWidth = 0
        first_Col = {}
        def helper(root,depth,col_index):
            nonlocal maxWidth
            if root is None:
                return
            if depth not in first_Col:
                first_Col[depth] = col_index
            maxWidth = max(maxWidth,col_index-first_Col[depth]+1)
            helper(root.left,depth+1,2*col_index)
            helper(root.right,depth+1,2*col_index+1)
        
        helper(root,0,0)
        return maxWidth
    
    
    def widthBinaryBFS(root):
        if not root:
            return 0
        queue = deque()
        queue.append((root,0))
        maxWidth = 1
        while queue:
            minCode, maxCode = queue[0][1], queue[-1][1]
            maxWidth = max(maxWidth, maxCode-minCode + 1)
            
            for _ in range(len(queue)):
                node,code = queue.popleft()
                if node.left:
                    queue.append((node.left,2*code))
                if node.right:
                    queue.append((node.right, 2*code+1))
                
        return maxWidth
    
"""
1071. Greatest Common Divisor of Strings

For two strings s and t, we say "t divides s" if and only if s = t + ... + t  (t concatenated with itself 1 or more times)

Given two strings str1 and str2, return the largest string x such that x divides both str1 and str2.

 

Example 1:

Input: str1 = "ABCABC", str2 = "ABC"
Output: "ABC"
"""
import math
def gcdOfStrings(str1,str2):
    n = len(str1)
    m = len(str2)
    
    gcd = math.gcd(n,m)
    print(gcd)
    astring = str1[:gcd] * (n//gcd)
    bstring = str2[:gcd] * (m//gcd)
    
    if str1==astring and str2 == bstring:
        return (str1[:gcd])
    
    else:
        return ""
"""
1520. Maximum Number of Non-Overlapping Substrings

Given a string s of lowercase letters, you need to find the maximum number of non-empty substrings of s that meet the following conditions:

The substrings do not overlap, that is for any two substrings s[i..j] and s[k..l], either j < k or i > l is true.
A substring that contains a certain character c must also contain all occurrences of c.
Find the maximum number of substrings that meet the above conditions. If there are multiple solutions with the same number of substrings, return the one with minimum total length. It can be shown that there exists a unique solution of minimum total length.

Notice that you can return the substrings in any order.

 

Example 1:

Input: s = "adefaddaccc"
Output: ["e","f","ccc"]
Explanation: The following are all the possible substrings that meet the conditions:
[
  "adefaddaccc"
  "adefadda",
  "ef",
  "e",
  "f",
  "ccc",
]
If we choose the first string, we cannot choose anything else and we'd get only 1. If we choose "adefadda", we are left with "ccc" which is the only one that doesn't overlap, thus obtaining 2 substrings. Notice also, that it's not optimal to choose "ef" since it can be split into two. Therefore, the optimal way is to choose ["e","f","ccc"] which gives us 3 substrings. No other solution of the same number of substrings exist.
"""  

def maxNumOfSubstrings(s):
    def get_substring_right(s, left_bound, right_bound, bound):
        right_most = right_bound
        i = left_bound
        while i <= right_most:
            letter = s[i]
            left, right = bound[letter]
            
            if left < left_bound:
                return -1
            
            right_most = max(right_most, right)
            
            i += 1
        
        return right_most
    bound = {}
    for i in range(len(s)):
        letter = s[i]
        if letter not in bound:
            bound[letter] = [i, i]
        bound[letter][1] = i
    
    substrings = []
    
    last_recorded_right = -1
    for i in range(len(s)):
        letter = s[i]
        left, right = bound[letter]
    
        if i == left:
            substring_right = get_substring_right(s, left, right, bound)
            
				# self contained substring
            if substring_right != -1: 
                if last_recorded_right < left:
                    substrings.append("")
                
                last_recorded_right = substring_right
                substrings[-1] = s[i:last_recorded_right + 1]
                
    return substrings
import unittest

def find_available_times(schedules):
    ret = []
    
    intervals = [list(x) for personal in schedules for x in personal]
    intervals.sort(key=lambda x:x[0], reverse = True)
    print(intervals)
    tem = []
    
    while intervals:
        pair = intervals.pop()
        if tem and tem[-1][1] >= pair[0]:
            tem[-1][1] = max(tem[-1][1], pair[1])
        else:
            tem.append(pair)
    
    for i in range(len(tem)-1):
        ret.append([tem[i][1], tem[i+1][0]])
    
    return ret

class CalendarTests(unittest.TestCase):

    def test_find_available_times(self):
        p1_meetings = [
            ( 845,  900),
            (1230, 1300),
            (1300, 1500),
        ]

        p2_meetings = [
            ( 0,    844),
            ( 845, 1200), 
            (1515, 1546),
            (1600, 2400),
        ]

        p3_meetings = [
            ( 845, 915),
            (1235, 1245),
            (1515, 1545),
        ]

        schedules = [p1_meetings, p2_meetings, p3_meetings]
       
        availability = [[844, 845], [1200, 1230], [1500, 1515], [1546, 1600]]

        self.assertEqual(
            find_available_times(schedules), 
            availability
            )


def main():
    unittest.main()


if __name__ == '__main__':
    main()
    
"""
361. Bomb Enemy

Given a 2D grid, each cell is either a wall 'W', an enemy 'E' or empty '0' (the number zero), return the maximum enemies you can kill using one bomb.
The bomb kills all the enemies in the same row and column from the planted point until it hits the wall since the wall is too strong to be destroyed.
Note: You can only put the bomb at an empty cell.

Example:

Input: [["0","E","0","0"],["E","0","W","E"],["0","E","0","0"]]
Output: 3 
Explanation: For the given grid,

0 E 0 0 
E 0 W E 
0 E 0 0

Placing a bomb at (1,1) kills 3 enemies.
"""

class DestroyEnemy:
    def maxKilledEnemy(self,grid):
        if not grid:
            return 0
        ans = 0
        m,n = len(grid),len(grid[0])
        kills =[[0 for c in range(n)] for _ in range(m)]
        self.helper(grid, [0,m], [0,n], +1,kills, False)
        self.helper(grid, [0,m], [n-1,-1], -1,kills, False )
        self.helper(grid, [0,n], [0,m], +1,kills, True)
        self.helper(grid, [0,n], [m-1,-1], -1,kills, True )


        for i in range(m):
            for j in range(n):
                ans = max(ans,kills[i][j])
        
        return ans
    
    def helper(self,grid,fixed,variable,increment,kills,switch):
        [i,j] = fixed
        [p,q] = variable
        
        for r in range(i,j):
            count = 0
            for c in range(p,q,increment):
                x,y = r,c
                if switch:
                    x,y = c,r
                if grid[x][y] == 'E':
                    count += 1
                if grid[x][y] =='W':
                    count = 0
                if grid[x][y] == '0':
                    kills[x][y] += count
        
s= DestroyEnemy()
print("Enemy",s.maxKilledEnemy([["0","E","0","0"],["E","0","W","E"],["0","E","0","0"]]))


def mutateTheArray(n,a):
    b = []
    for i in range(n):
        if i == 0:
            b.append((0+a[i]+a[i+1]))
        elif i == n-1:
            b.append((a[i-1]+a[i]+0))
        else:
            b.append((a[i-1]+a[i]+a[i+1]))
    
    return b


def hashMap(queryType, query):
    h = HashMap()
    for i in range(len(queryType)):
        if queryType[i] == "insert":
            h.insert(query[i][0],query[i][1])
        elif queryType[i] == "addToValue":
            h.addToValue(query[i])
        elif queryType[i] == "addToKey":
            h.addToKey(query[i])
        elif queryType[i] == "get":
            return h.get(query[i])
class HashMap:
    def __init__(self):
        self.myDict = {}
    def insert(self,x,y):
        self.myDict[x] = y
    def addToValue(self,x):
        for key,values in self.myDict.items():
            self.myDict[key] = values+x
        print(self.myDict)
    def addToKey(self,x):
        kv = list(self.myDict.items())
        self.myDict.clear()
        for key,values in kv:
            nKey = key + x
            self.myDict[nKey] = values
            print(self.myDict) 
    def get(self,x):
        return self.myDict[x]

"""
You are given numbers, a 3 × n matrix which contains only digits from 1 to 9. Consider a sliding window of size 3 × 3, which is sliding from left to right through the matrix numbers. The sliding window has n - 2 positions when sliding through the initial matrix.

Your task is to find whether or not each sliding window position contains all the numbers from 1 to 9 (inclusive). Return an array of length n - 2, where the ith element is true if the ith state of the sliding window contains all the numbers from 1 to 9, and false otherwise.

Example

numbers = [[1, 2, 3, 2, 5, 7],
           [4, 5, 6, 1, 7, 6],
           [7, 8, 9, 4, 8, 3]]

the output should be isSubmatrixFull(numbers) = [true, false, true, false].
"""

def isSubMatriFull(mat):
    n = len(mat[0])
    ans = [False]*(n-2)
    kernel = getCol(mat,0) + getCol(mat,1) + getCol(mat,2)
    print(kernel)
    for i in range(n-2):
        print(kernel)
        if len(set(kernel)) == 9:
            ans[i] = True
        if i < n-3:
            kernel = kernel[3:]+getCol(mat,i+3)
    return ans

def getCol(mat,col):
    return [mat[i][col] for i in range(3)]


"""
Snake and Ladder
"""

def snakesAndLadders(board):
    
    if not board: return None
        
    N = len(board)
    b_dict = {}
    for r in range(N):
        for c in range(N):
            if board[r][c] != -1:
                
                #if size is even: r is even: 2 flips
                #if size is even: r is odd: 1 flip
                #if size is odd: r is odd: 2 flips
                #if size is odd: r is even: 1 flip
                
                if N%2 == r%2:
                    num = (N-r-1)*N+(N-c-1)+1
                else:
                    num = (N-r-1)*N + c+1
                    
                b_dict[num] = board[r][c]
                    
    # # print(b_dict)
    # print(board)
    
    #bfs
    start = 1
    finish = N*N
    no_turns = 0
    l = [(start, no_turns)]
    queue = collections.deque(l)
    visited = set()
    while queue:
        num, no_turns = queue.popleft()
        
        #1...6
        for next_move in range(num+1, num+7):
        
            if next_move in b_dict:
                next_move = b_dict[next_move]
                
            if next_move >= finish:
                return no_turns + 1
                
            if next_move not in visited:
                visited.add(next_move)
 
                queue.append((next_move,no_turns+1))
    
    return -1


"""
["Bob","Alice","Ema","Alice","Charlie"]
["Bob","Alice","Ema"] or ["Ema","Alice","Charlie"]

["Mike","Bob","Mike","Ema","Alice","Charlie"]
["Mike","Bob","Alice","Ema"]
"""

def longest_Sub_Array(arr):
    d = {}
    start = 0
    length = 0
    lengt= 0
    for window in range(len(arr)):
        print(arr[window])
        if arr[window] in d:
            if lengt < length - (window-start):
                lengt = window -1
            start = max(start, d[arr[window]]+1)
            print(start,length)
        d[arr[window]] = window
        length = max(length, window-start + 1)
        print(start,d,length)
    return arr[lengt:length]
            
def longestSubArray(arr):
    d = {}
    ans = []
    start = 0
    end = 0
    for i,char in enumerate(arr):
        if char in d:
            start = d[min(d,key=d.get)]
            end = d[max(d,key=d.get)]+1
            temp = arr[start:end]
            if len(temp)>len(ans) or len(ans)==0:
                ans = temp
        d[char] = i
    if len(ans) < i-end:
        ans = arr[end:i+1]
    return ans

def longest_distinct_array(arr):
    last_seen_hash = {}
    max_length_so_far = 0 
    start_index = 0 
    length = 0
    for i in range(len(arr)):
        if arr[i] in last_seen_hash:
            last_seen = last_seen_hash[arr[i]]
            if last_seen >= start_index:
                length = i - 1
            if (i-start_index)>length:
                start_index = last_seen + 1
            if length > max_length_so_far:
                max_length_so_far = length
            length = 0                
        else:
            last_seen_hash[arr[i]] = i
            length += 1
        
        if length > max_length_so_far:
            max_length_so_far = length
        
    return arr[start_index:(start_index+max_length_so_far)+1] if max_length_so_far>0 else arr

"""
Smallest Substring of All Characters
Given an array of unique characters arr and a string str, Implement a function getShortestUniqueSubstring that finds the smallest substring of str containing all the characters in arr. Return "" (empty string) if such a substring doesn’t exist.

Come up with an asymptotically optimal solution and analyze the time and space complexities.

Example:

input:  arr = ['x','y','z'], str = "xyyzyzyx"

output: "zyx"
"""
def get_shortest_unique_substring(arr, str):
  mychar = set(arr)
  substr = ""
  current = {}
  for i,char in enumerate(str):
    if char in mychar:
      current[char] = i
      if len(current) == len(mychar):
        temp = str[current[min(current,key=current.get)]:current[max(current,key=current.get)]+1]
        if len(temp) < len(substr) or len(substr) == 0:
          substr = temp
  return substr


def sortedArray(arr):
    ans = set()
    arr.sort()
    for i in range(len(arr)-1):
        if arr[i]== arr[i+1]:
            ans.add(arr[i])
    
    return sorted(tuple(ans))


def solution(A, B, C):
    
    rt = ""
    while (A>0 or B>0 or C>0):
        if A > B and B != 0:
            if len(rt) > 0 and rt[-1]=='a':
                continue
            print("B<A",A,B,C)
            if (A > 0): 
                rt += 'a'
                A -= 1
            if (A > 0): 
                rt += 'a'
                A -= 1
            if (B > 0): 
                rt += 'b'
                B -= 1
            
        if (B > A) and A != 0:
            if len(rt) > 0 and rt[-1]=='b':
                continue
            print("A<B",A,B,C)
            if (B > 0):
                rt = rt+'b'
                B -= 1
            if (B > 0):
                rt+='b'
                B -= 1
            if (A > 0):
                rt+='a'
                A -= 1
        if (C > B) and B != 0:
            if len(rt) > 0 and rt[-1]=='c':
                continue
            print("B<C",A,B,C)
            if (C > 0): 
                rt += 'c'
                C -= 1
            if (C > 0): 
                rt += 'c'
                C -= 1
            if (B > 0): 
                rt += 'b'
                B -= 1
        if (A > C) and C!=0:
            if len(rt) > 0 and rt[-1]=='c':
                continue
            print("C<A",A,B,C)
            if (A > 0):
                rt += 'a'
                A -= 1
            if (A > 0):
                rt += 'a'
                A -= 1
            if (C > 0): 
                rt += 'c'
                C -= 1
        if (C > A) and A != 0:
            if len(rt) > 0 and rt[-1]=='c':
                continue
            print("A<C",A,B,C)
            if (C > 0):
                rt = rt+'c'
                C -= 1
            if (C > 0):
                rt+='c'
                C -= 1
            if (A > 0):
                rt+='a'
                A -= 1
        if (B > C) and C != 0:
            if len(rt) > 0 and rt[-1]=='b':
                continue
            print("C<B",A,B,C)
            if (B > 0):
                rt = rt+'b'
                B -= 1
            if (B > 0):
                rt+='b'
                B -= 1
            if (C > 0):
                rt+='c'
                C -= 1
        else :
            print("Else",A,B,C)
            if (0 < A): 
                rt += 'a'
                A -= 1
            if (0 < B):   
                rt += 'b'
                B -= 1
            if (0 < C):
                rt += 'c'
                C -= 1
    return rt

"""
76. Minimum Window Substring

Given two strings s and t, return the minimum window in s which will contain all the characters in t. If there is no such window in s that covers all characters in t, return the empty string "".

Note that If there is such a window, it is guaranteed that there will always be only one unique minimum window in s.

Example 1:

Input: s = "ADOBECODEBANC", t = "ABC"
Output: "BANC"
"""
from collections import Counter

def minWindow(s, t):
    trunning = Counter(t)
    left_uncovered = set(t)
    left = 0
    mins = ''
    for right, ch in enumerate(s):
        print(left,right,trunning,left_uncovered)
        if ch not in trunning: 
            continue
        trunning[ch] -= 1
        if not trunning[ch]:
            left_uncovered.remove(ch)
        while s[left] not in trunning or trunning[s[left]] < 0:
            if s[left] in trunning: 
                trunning[s[left]] += 1
            left += 1
        print("Hi",trunning,left_uncovered,mins,right,left)
        if left_uncovered:
            print("Hi")
            continue
        elif not mins or right+1-left < len(mins):
            print("mins",left,right,s[left:right+1])
            mins = s[left:right+1]
    
    return mins
"""
266. Palindrome Permutation

Given a string, determine if a permutation of the string could form a palindrome.

Example 1:

Input: "code"
Output: false
"""
def canPermutePalindrome(s):
    letters = defaultdict(int)
    for c in s:
        """
        count the number of appearances of each letter
        """
        letters[c] += 1
    ones = 0
    for letter in letters:
        occurrences = letters[letter]
        
        """
        ensure there is at most one letter that
        occurs an odd number of times
        """
        ones += 1 if occurrences % 2 != 0 else 0
        if ones > 1: return False
    
    return True