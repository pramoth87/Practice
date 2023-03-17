# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 18:55:27 2020

@author: prchandr
"""

"""
57. Insert Interval

Given a set of non-overlapping intervals, insert a new interval into the intervals (merge if necessary).

You may assume that the intervals were initially sorted according to their start times.

Example 1:

Input: intervals = [[1,3],[6,9]], newInterval = [2,5]
Output: [[1,5],[6,9]]
"""

from collections import defaultdict

def insertInterval(intervals, newInterval):
    newset = []
    go = True
    for v in intervals:
        if go:
            if v[0] <= newInterval[0] <= v[1] or newInterval[0] <= v[0]:
                newset.append([min(v[0],newInterval[0]), max(v[1],newInterval[1])])
                go = False
            else:
                newset.append(v)
        else:
            #newset = insertInterval(newset,v)
            newset.append(v)
    if go:
        newset.append(newInterval)
    return newset

def insert(intervals, newInterval):
    res = []
    for idx, interval in enumerate(intervals):
        if interval[1] < newInterval[0]:
            print(interval)
            res.append(interval)
        elif newInterval[1] < interval[0]:
            print("elif",interval)
            res.append(newInterval)
            return res + intervals[idx:]
        else:
            newInterval[0] = min(newInterval[0], interval[0])
            newInterval[1] = max(newInterval[1], interval[1])
        print(newInterval)
    res.append(newInterval)
    return res

def largestTimeFromDigits(A):
    A.sort()
    for h in range(23,-1,-1):
        for m in range(59,-1,-1):
            t = [h//10, h%10, m//10, m%10]
            td = sorted(t)
            if td == A:
                return str(t[0])+str(t[1]) + ':' + str(t[2]) + str(t[3])
    
    return ""

def largestTimeFromDigitsBT(A):
    limits = [2, 9, 5, 9]
    nums = sorted(A, reverse=True)
    def backtrack(ans, i):
        print("Ans",ans)
        for j in range(4):
            t = nums[j]
		# Get valid range for current index
            lm = 3 if i == 1 and ans[0] == 2 else limits[i]
            print("limit",lm,nums,i)
            if 0 <= nums[j] <= lm:
                ans[i] = nums[j]
                print("Inside If", tuple(ans))
                # Mark value at current index as unavailable
                nums[j] = -1
                if i == 3:
                    return '%s%s:%s%s' % tuple(ans)
                else:
                    ret = backtrack(ans, i+1)
                    if ret: # If found answer then return
                        return ret
				# Revert value at current index as available
                nums[j] = t
        return ''

    ans = [-1] * 4
    return backtrack(ans, 0)
"""
To find the maximum Rectangle

Matrix = [["1","0","1","0","0"],
          ["1","0","1","1","1"],
          ["1","1","1","1","1"],
          ["1","0","0","1","0"]]

Output = 6
"""
def maximalRectangle(matrix):
    maxarea = 0
    dp = [[0 for i in range(len(matrix[0]))] for _ in range(len(matrix))]
    for i in range(len(matrix)):
        print(i)
        for j in range(len(matrix[0])):
            if matrix[i][j] == '0': continue
            # compute the maximum width and update dp with it
            width = dp[i][j] = dp[i][j-1] + 1 if j else 1
            print(dp)

            # compute the maximum area rectangle with a lower right corner at [i, j]
            for k in range(i, -1, -1):
                
                width = min(width, dp[k][j])
                maxarea = max(maxarea, width * (i-k+1))
                print(width,k,i,maxarea)
                
    return maxarea

"""
To Find the maximum Square

Input: 

1 0 1 0 0
1 0 1 1 1
1 1 1 1 1
1 0 0 1 0

Output: 4

"""

def maximalSquare(matrix):
    if not matrix:
        return 0
    nrows = len(matrix)
    ncols = len(matrix[0])
    max_square_len = 0
    dp = [[0] * (ncols + 1) for i in range(nrows + 1)]
    
    for i in range(1, nrows + 1):
        for j in range(1, ncols + 1):
            if (matrix[i - 1][j - 1] == '1'):
                dp[i][j] = min(dp[i][j - 1], dp[i - 1][j], dp[i - 1][j - 1]) + 1
                max_square_len = max(max_square_len, dp[i][j])
                
    return max_square_len ** 2


def removeOuterParentheses(S):
    output = ""
    count = 0
    for i in range(len(S)):
        if count > 0 and not (count ==1 and S[i] == ")"):
            output += S[i]
        if S[i] == "(":
            count += 1
        else:
            count -= 1
    return output


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


def networkDelayTime(times,N,K):
    travelDict = defaultdict(list)
    for edges in times:
        travelDict[edges[0]].append((edges[2],edges[1]))
    dist = {node: float('inf') for node in range(1,N+1)}
    def dfs(node,elapsed):
        if elapsed >= dist[node]: return
        dist[node] = elapsed
        for time,neighbour in sorted(travelDict[node]):
            print(time,neigh)
            dfs(neighbour,elapsed+time)
    dfs(K,0)
    ans = max(dist.values())
    return ans if ans < float('inf') else -1


def isLongPressedName(name, typed):
    i = 0
    j = 0
    stack = []
    while i < len(name) and j < len(typed):
        if name[i] == typed[j]:
            stack.append(name[i])
            i += 1
            j += 1
        else:
            if len(stack) == 0 or typed[j] != stack[-1]:
                return False
            j += 1
    
    if len(stack) != len(name):
        return False
    
    for k in range(j,len(typed)):
        if stack[-1] != typed[k]:
            return False
    return True


"""

299. Bulls and Cows

You are playing the following Bulls and Cows game with your friend: 
You write down a number and ask your friend to guess what the number is. 
Each time your friend makes a guess, you provide a hint that indicates 
how many digits in said guess match your secret number exactly in both digit and position (called "bulls") 
and how many digits match the secret number but locate in the wrong position (called "cows"). 
Your friend will use successive guesses and hints to eventually derive the secret number.

Write a function to return a hint according to the secret number and friend's guess, 
use A to indicate the bulls and B to indicate the cows.
Example 1:

Input: secret = "1807", guess = "7810"

Output: "1A3B"

Explanation: 1 bull and 3 cows. The bull is 8, the cows are 0, 1 and 7.
Example 2:

Input: secret = "1123", guess = "0111"

Output: "1A1B"

Explanation: The 1st 1 in friend's guess is a bull, the 2nd or 3rd 1 is a cow.

"""
def getHint(secret,guess):
    h = defaultdict(int)
    Bulls = 0
    Cows = 0
    for idx, st in enumerate(secret):
        g = guess[idx]
        if st == g:
            Bulls += 1
        else:
            print(st,g,Cows, int(h[st] <0), int(h[st] > 0))
            Cows += int(h[st] < 0) + int(h[g] > 0)
            #print(st,g,Cows, int(h[st] <0), int(h[st] > 0))
            h[st] += 1
            h[g] -= 1
        print(h)
    return str(Bulls) + "A" + str(Cows) + "B"


def calc_drone_min_energy(route):
  ans = 0
  Tank = 0
  for i in range(len(route)-1):#check boundary for last
    if route[i+1][2] <= route[i][2]:#next step is goind down, need not worry about fuel
      Tank += route[i][2] - route[i+1][2]
    else:
      fuel_needed = route[i+1][2] - route[i][2]
      
      if Tank >= fuel_needed: # dont worry and fly
        Tank = Tank - fuel_needed
      else:
        ans += fuel_needed - Tank
        Tank = 0
        
  return ans


"""
315. Count of Smaller Numbers After Self

You are given an integer array nums and you have to return a new counts array. 
The counts array has the property where counts[i] is the number of smaller elements to the right of nums[i].

Example 1:

Input: nums = [5,2,6,1]
Output: [2,1,1,0]
Explanation:
To the right of 5 there are 2 smaller elements (2 and 1).
To the right of 2 there is only 1 smaller element (1).
To the right of 6 there is 1 smaller element (1).
To the right of 1 there is 0 smaller element.
"""

def countSmaller(nums):
    n = len(nums)
    if not n:
        return []
    # Create a sorted list with last element of given arr
    sorted_list = [nums[-1]]
    # Last number has 0 smaller elements to its right
    output = [0]

    def search(arr,target):
        na = len(arr)
        left, right = 0, na - 1

        # Quick checks of sorted list beg and end index values
        if arr[0] >= target:
            return 0
        if arr[-1] < target:
            return na

        while left <= right:
            mid = left + (right - left) // 2

            if arr[mid] == target:
                # If you have found the number, great!
                # but you want the number smaller to it
                # Linear probe backwards from mid, unless
                # you find different number (since sorted)
                # you will find the next smaller number and
                # return its index
                while mid > 0 and arr[mid] == target:
                    mid -= 1
                return mid + 1
            
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return left

    # Since we have already processed last element, start from n - 2
    for i in range(n - 2, -1, -1):
        # Binary search the index
        index = search(sorted_list, nums[i])
        output.append(index)
        print("ForLoop",sorted_list,index,nums[i])
        # Grow the sorted list after every iteration
        # By inserting the element in correct index position
        sorted_list.insert(index, nums[i])
        
    print(sorted_list,output)

    return output[::-1]

"""
329. Longest Increasing Path in a Matrix

Given an integer matrix, find the length of the longest increasing path.

From each cell, you can either move to four directions: left, right, up or down. 
You may NOT move diagonally or move outside of the boundary (i.e. wrap-around is not allowed).

Example 1:
Input: nums = 
[ [9,9,4],
  [6,6,8],
  [2,1,1]] 
Output: 4 
Explanation: The longest increasing path is [1, 2, 6, 9].
"""

def longestIncreasingPath(matrix):
    if not matrix or not matrix[0]:
        return 0
    memo = {}
    res = 0
    
    m = len(matrix)
    n = len(matrix[0])
    def dfs(i,j):
        if (i,j) in memo:
            return memo[(i,j)]
        step = 1
        for ni,nj in ((0,1),(1,0),(0,-1),(-1,0)):
            x = i + ni
            y = j + nj
            if 0 <= x<m and 0<=y<n and matrix[x][y] > matrix[i][j]:
                tem = 1+dfs(x,y)
                step = max(step,tem)
        memo[(i,j)] = step
        return step
    
    for i in range(m):
        for j in range(n):
            temp = dfs(i,j)
            res = max(res,temp)
    return res

"""

337. House Robber III

The thief has found himself a new place for his thievery again. 
There is only one entrance to this area, called the "root." 
Besides the root, each house has one and only one parent house. 
After a tour, the smart thief realized that "all houses in this place forms a binary tree". 
It will automatically contact the police if two directly-linked houses were broken into on the same night.

Determine the maximum amount of money the thief can rob tonight without alerting the police.

"""
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

def rob(root):
    def roberMind(root):
        if not root:
            return (0, 0)
        robbedLeft, skippedLeft = roberMind(root.left)
        robbedRight, skippedRight = roberMind(root.right)
        #print("Values",robbedLeft,skippedLeft,robbedRight,skippedRight )
        return (root.val+skippedLeft+skippedRight, max(robbedLeft+robbedRight, skippedLeft+skippedRight, skippedLeft+robbedRight, robbedLeft+skippedRight))
    return max(roberMind(root))

"""
You have n  tiles, where each tile has one letter tiles[i] printed on it.

Return the number of possible non-empty sequences of letters you can make using the letters printed on those tiles.

 

Example 1:

Input: tiles = "AAB"
Output: 8
Explanation: The possible sequences are "A", "B", "AA", "AB", "BA", "AAB", "ABA", "BAA".
"""

def numTilePossibilities(tiles):    

    list_of_tiles = list(tiles)
    print("List",list_of_tiles)
    paths = []
    def handle_tile(path, used):
        if path not in paths:
            print(path)
            paths.append(path[:])
            print("Inside if Loop",path,paths)
        for idx, tile in enumerate(list_of_tiles):
            if not used[idx]:
                path.append(tile)
                used[idx] = True
                handle_tile(path, used)
                used[idx] = False
                path.pop()
    handle_tile([], [False] * len(list_of_tiles))    
    return len(paths) - 1

"""
1042. Flower Planting With No Adjacent

You have N gardens, labelled 1 to N.  In each garden, you want to plant one of 4 types of flowers.

paths[i] = [x, y] describes the existence of a bidirectional path from garden x to garden y.

Also, there is no garden that has more than 3 paths coming into or leaving it.

Your task is to choose a flower type for each garden such that, for any two gardens connected by a path, they have different types of flowers.

Return any such a choice as an array answer, where answer[i] is the type of flower planted in the (i+1)-th garden.  The flower types are denoted 1, 2, 3, or 4.  It is guaranteed an answer exists.


Example 1:

Input: N = 3, paths = [[1,2],[2,3],[3,1]]
Output: [1,2,3]
"""

def gardenNoAdj(N, paths):
    # four color style question
    # traversal the graph, color each node
    color = [0] * (N+1) # 0 for not colored yet, and real color takes from 1 to 4
    
    graph = defaultdict(list)
    for u, v in paths:
        graph[u].append(v)
        graph[v].append(u)
    
    def dfs(node):
        if color[node] > 0:
            return True
        useless = set([color[v] for v in graph[node] if color[v] > 0])
        print("Useless",color,useless)
        for i in range(1, 5):
            if i not in useless: # take one color (try)
                print(graph,color,node)
                color[node] = i
                print(graph,color,node)
                isok = True
                for v in graph[node]:
                    print(v)
                    if not dfs(v):
                        print("if",v,node)
                        isok = False
                        break
                if isok:
                    return True
                color[node] = 0
        
        return False
        
    for u in range(1, N+1):
        print("u",u)
        dfs(u)
    return color[1:]

"""
346. Moving Average from Data Stream

Given a stream of integers and a window size, calculate the moving average of all integers in the sliding window.

Example:

MovingAverage m = new MovingAverage(3);
m.next(1) = 1
m.next(10) = (1 + 10) / 2
m.next(3) = (1 + 10 + 3) / 3
m.next(5) = (10 + 3 + 5) / 3
"""

class MovingAverage:
    def __init__(self,size):
        self.size = size
        self.queue = []
        
    def next(self,val):
        self.queue.append(val)
        moving_sum = sum(self.queue[-self.size:])
        return moving_sum/min(len(self.queue),self.size)    

m = MovingAverage(3);
print(m.next(1))
print(m.next(10))
print(m.next(3))
print(m.next(5))

"""
359. Logger Rate Limiter

Design a logger system that receives a stream of messages along with their timestamps. Each unique message should only be printed at most every 10 seconds (i.e. a message printed at timestamp t will prevent other identical messages from being printed until timestamp t + 10).

All messages will come in chronological order. Several messages may arrive at the same timestamp.

Implement the Logger class:

Logger() Initializes the logger object.
bool shouldPrintMessage(int timestamp, string message) Returns true if the message should be printed in the given timestamp, otherwise returns false.
 

Example 1:

Input
["Logger", "shouldPrintMessage", "shouldPrintMessage", "shouldPrintMessage", "shouldPrintMessage", "shouldPrintMessage", "shouldPrintMessage"]
[[], [1, "foo"], [2, "bar"], [3, "foo"], [8, "bar"], [10, "foo"], [11, "foo"]]
Output
[null, true, true, false, false, false, true]

Explanation
Logger logger = new Logger();
logger.shouldPrintMessage(1, "foo");  // return true, next allowed timestamp for "foo" is 1 + 10 = 11
logger.shouldPrintMessage(2, "bar");  // return true, next allowed timestamp for "bar" is 2 + 10 = 12
logger.shouldPrintMessage(3, "foo");  // 3 < 11, return false
logger.shouldPrintMessage(8, "bar");  // 8 < 12, return false
logger.shouldPrintMessage(10, "foo"); // 10 < 11, return false
logger.shouldPrintMessage(11, "foo"); // 11 >= 11, return true, next allowed timestamp for "foo" is 11 + 10 = 21
"""
from collections import deque

class Logger(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self._msg_set = set()
        self._msg_queue = deque()
    
    def shouldPrintMessage(self, timestamp, message):
        """
        Returns true if the message should be printed in the given timestamp, otherwise returns false.
        """
        while self._msg_queue:
            msg, ts = self._msg_queue[0]
            if timestamp - ts >= 10:
                self._msg_queue.popleft()
                self._msg_set.remove(msg)
            else:
                break
        
        if message not in self._msg_set:
            self._msg_set.add(message)
            self._msg_queue.append((message, timestamp))
            return True
        else:
            return False
"""

375. Guess Number Higher or Lower II

We are playing the Guess Game. The game is as follows:

I pick a number from 1 to n. You have to guess which number I picked.

Every time you guess wrong, I'll tell you whether the number I picked is higher or lower.

However, when you guess a particular number x, and you guess wrong, you pay $x. You win the game when you guess the number I picked.

Example:

n = 10, I pick 8.

First round:  You guess 5, I tell you that it's higher. You pay $5.
Second round: You guess 7, I tell you that it's higher. You pay $7.
Third round:  You guess 9, I tell you that it's lower. You pay $9.

Game over. 8 is the number I picked.

You end up paying $5 + $7 + $9 = $21.
"""

def getmoneyAmount(n):
    dp = [[0] * (n+1) for i in range(n+1) ]
    return dfsDP(1,n,dp)
def dfsDP(start,end,dp):
    if start >= end:
        return 0
    if dp[start][end] > 0:
        print(dp)
        return dp[start][end]
    
    res = float('inf')
    for amt in range(start, end + 1):
        tmp = amt + max(dfsDP(amt+1, end , dp), dfsDP(start, amt-1, dp))
        res = min(res,tmp)
    dp[start][end] = res
    
    return res

"""
394. Decode String

Given an encoded string, return its decoded string.

The encoding rule is: k[encoded_string], where the encoded_string inside the square brackets is being repeated exactly k times. Note that k is guaranteed to be a positive integer.

You may assume that the input string is always valid; No extra white spaces, square brackets are well-formed, etc.

Furthermore, you may assume that the original data does not contain any digits and that digits are only for those repeat numbers, k. For example, there won't be input like 3a or 2[4].


Example 1:

Input: s = "3[a]2[bc]"
Output: "aaabcbc"
"""

def decodeString(s):
    res = ''
    stack = []
    for char in s:
        print("char",char)
        if char != ']':
            stack.append(char)
            print(stack)
        else:
            
            currStr, multiplier = '', ''
            while stack[-1] != '[':
                currStr = stack.pop() + currStr
                print(currStr)
            stack.pop()
            while len(stack) and stack[-1].isdigit():
                multiplier = stack.pop() + multiplier
            curr= int(multiplier) * currStr
            print("curr",curr, currStr,stack)
            for c in curr:
                stack.append(c)
                print("Stack",stack)
    return res + "".join(stack)

"""

X is a good number if after rotating each digit individually by 180 degrees, we get a valid number that is different from X.  Each digit must be rotated - we cannot choose to leave it alone.

A number is valid if each digit remains a digit after rotation. 0, 1, and 8 rotate to themselves; 2 and 5 rotate to each other (on this case they are rotated in a different direction, in other words 2 or 5 gets mirrored); 6 and 9 rotate to each other, and the rest of the numbers do not rotate to any other number and become invalid.

Now given a positive number N, how many numbers X from 1 to N are good?

Example:
Input: 10
Output: 4
Explanation: 
There are four good numbers in the range [1, 10] : 2, 5, 6, 9.
Note that 1 and 10 are not good numbers, since they remain unchanged after rotating.
"""

def rotatedDigits(N):
    myDict = {'2':'5','5':'2','6':'9','9':'6','0':'0','1':'1','8':'8'}
    invalid = ['3','4','7']
    dp = [0,0,1,1,1,2,3,3,3,4,4]
    if N<= 10:
        return dp[N]
    else:
        for i in range(10, N+1):
            s = ''
            inv = 0
            for j in str(i):
                if j in invalid:
                    inv = 1
                else:
                    s += myDict[j]
            
            if inv:
                dp.append(dp[-1])
            else:
                if str(i) == s:
                    dp.append(dp[-1])
                else:
                    dp.append(dp[-1]+1)
    return dp[N+1]

"""
In a row of seats, 1 represents a person sitting in that seat, and 0 represents that the seat is empty. 

There is at least one empty seat, and at least one person sitting.

Alex wants to sit in the seat such that the distance between him and the closest person to him is maximized. 

Return that maximum distance to closest person.

Example 1:

Input: [1,0,0,0,1,0,1]
Output: 2
Explanation: 
If Alex sits in the second open seat (seats[2]), then the closest person has distance 2.
If Alex sits in any other open seat, the closest person has distance 1.
Thus, the maximum distance to the closest person is 2.
"""

def maxDistToClosest(seats):
    N = len(seats)
    left, right = [N] * N, [N] * N
    for i in range(N):
        if seats[i] == 1:
            left[i] = 0
        elif i > 0:
            left[i] = left[i-1] + 1
    
    for i in range(N-1,-1,-1):
        if seats[i] == 1:
            right[i] = 0
        elif i < N-1:
            right[i] = right[i+1] + 1
        
    print(left, right)
    return max(min(left[i], right[i]) for i, seat in enumerate(seats) if not seat)

"""
399. Evaluate Division

Equations are given in the format A / B = k, where A and B are variables represented as strings, and k is a real number (floating point number). Given some queries, return the answers. If the answer does not exist, return -1.0.

Example:
Given a / b = 2.0, b / c = 3.0.
queries are: a / c = ?, b / a = ?, a / e = ?, a / a = ?, x / x = ? .
return [6.0, 0.5, -1.0, 1.0, -1.0 ].

The input is: vector<pair<string, string>> equations, vector<double>& values, vector<pair<string, string>> queries , where equations.size() == values.size(), and the values are positive. This represents the equations. Return vector<double>.

According to the example above:

equations = [ ["a", "b"], ["b", "c"] ],
values = [2.0, 3.0],
queries = [ ["a", "c"], ["b", "a"], ["a", "e"], ["a", "a"], ["x", "x"] ]. 

"""

class Solution:
    def calcEquation(self, equations,values, queries):
        # make graph
        g = defaultdict()
        for i, (a, b) in enumerate(equations):
            if a not in g: g[a] = defaultdict()
            if b not in g: g[b] = defaultdict()
            g[a][b] =   values[i]
            g[b][a] = 1/values[i]
        print("graph",g)
        # solve equation: start -> next_node -> end node
        def dfs(s, e, curr_val, visited):
            if s in visited: return
            visited.add(s)
            if s == e: self.res = curr_val; return
            for next_node, next_val in g[s].items():
                print("For",next_node,next_val)
                dfs(next_node, e, curr_val * next_val, visited)
        # calculate result
        res = []
        for s, e in queries:  # s:start, e:end
            if s not in g or e not in g: res.append(-1.0   ); continue
            elif s == e                : res.append( 1.0   ); continue
            elif e in g[s]             : res.append(g[s][e]); continue
            self.res = -1.0            
            dfs(s, e, 1.0, set())
            res.append(self.res)
        return res    
s = Solution()
print(s.calcEquation([["a", "b"],["b", "c"]],[2.0,3.0],[["a", "c"],["b", "a"],["a", "e"],["a", "a"],["x", "x"]]))


"""
489. Robot Room Cleaner

Given a robot cleaner in a room modeled as a grid.

Each cell in the grid can be empty or blocked.

The robot cleaner with 4 given APIs can move forward, turn left or turn right. Each turn it made is 90 degrees.

When it tries to move into a blocked cell, its bumper sensor detects the obstacle and it stays on the current cell.

Design an algorithm to clean the entire room using only the 4 given APIs shown below.

"""

def cleanRoom(self, robot):
		"""
		:type robot: Robot
		:rtype: None
		"""
		direction = [(-1, 0), (0, 1), (1, 0), (0, -1)]

		def dfs(robot, i, j, di, done):
			robot.clean()
			done.add((i, j))
			for _ in range(4):
				robot.turnRight()
				di = (di + 1) % 4
				x, y = direction[di]
				if (i + x, j + y) not in done and robot.move():
					dfs(robot, i + x, j + y, di, done)
			robot.turnRight()
			robot.turnRight()
			robot.move()
			robot.turnRight()
			robot.turnRight()                

		dfs(robot, 0, 0, 0, set())

"""
You are given a license key represented as a string S which consists only alphanumeric character and dashes. The string is separated into N+1 groups by N dashes.

Given a number K, we would want to reformat the strings such that each group contains exactly K characters, except for the first group which could be shorter than K, but still must contain at least one character. Furthermore, there must be a dash inserted between two groups and all lowercase letters should be converted to uppercase.

Given a non-empty string S and a number K, format the string according to the rules described above.

Example 1:
Input: S = "5F3Z-2e-9-w", K = 4

Output: "5F3Z-2E9W"

Explanation: The string S has been split into two parts, each part has 4 characters.
Note that the two extra dashes are not needed and can be removed.
"""     

def licenseKeyFormatting(S,K):
    S = S.replace('-','').upper()
    res = []
    if not S:
        return ''
    if len(S) % K:
        res.append(S[:len(S)%K])
        S = S[len(S)%K:]
    while S:
        res.append(S[len(S) % K:K])
        S = S[K:]
    
    if not res[0]:
        res.pop(0)
    
    return '-'.join(res)

"""
Suppose we abstract our file system by a string in the following manner:

The string "dir\n\tsubdir1\n\tsubdir2\n\t\tfile.ext" represents:

dir
    subdir1
    subdir2
        file.ext
The directory dir contains an empty sub-directory subdir1 and a sub-directory subdir2 containing a file file.ext.

The string "dir\n\tsubdir1\n\t\tfile1.ext\n\t\tsubsubdir1\n\tsubdir2\n\t\tsubsubdir2\n\t\t\tfile2.ext" represents:

dir
    subdir1
        file1.ext
        subsubdir1
    subdir2
        subsubdir2
            file2.ext
"""

def lengthLongestPath(input):
    if not input or '.' not in input:
        return 0
    p = input.split('\n')
    st = []
    m = 0
    for d in p:
        level = d.count('\t') + 1
        if level:
            d = d.replace('\t', '')
        
        while len(st) >= level:
            st.pop()
        st.append(d)
        
        if '.' in d:
            ret = '\\'.join(st)
            m = max(m,len(ret))
    return m

"""
528. Random Pick with Weight

Given an array of positive integers w. where w[i] describes the weight of ith index (0-indexed).

We need to call the function pickIndex() which randomly returns an integer in the range [0, w.length - 1]. pickIndex() 
should return the integer proportional to its weight in the w array. 
For example, for w = [1, 3], the probability of picking index 0 is 1 / (1 + 3) = 0.25 (i.e 25%) 
while the probability of picking index 1 is 3 / (1 + 3) = 0.75 (i.e 75%).

More formally, the probability of picking index i is w[i] / sum(w).


Example 1:

Input
["Solution","pickIndex"]
[[[1]],[]]
Output
[null,0]

Explanation
Solution solution = new Solution([1]);
solution.pickIndex(); // return 0. 
Since there is only one single element on the array the only option is to return the first element.
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

    def pickIndex(self) -> int:
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
659. Split Array into Consecutive Subsequences

Given an array nums sorted in ascending order, return true if and only if you can split it into 1 or more subsequences such that each subsequence consists of consecutive integers and has length at least 3.


Example 1:

Input: [1,2,3,3,4,5]
Output: True
Explanation:
You can split them into two consecutive subsequences : 
1, 2, 3
3, 4, 5
"""

def isPossible(nums):
    count = Counter(nums)
    tails = Counter()
    print(count,tails)
    for x in nums:
        if count[x] == 0:
            continue
        elif tails[x] > 0:
            tails[x] -= 1
            tails[x+1] += 1
        elif count[x+1] > 0 and count[x+2] > 0:
            count[x+1] -= 1
            count[x+2] -= 1
            tails[x+3] += 1
        else:
            return False
        count[x] -= 1
        print(count,tails)
    print(count,tails)
    return True

"""
683. K Empty Slots

You have N bulbs in a row numbered from 1 to N. Initially, all the bulbs are turned off. We turn on exactly one bulb everyday until all bulbs are on after N days.

You are given an array bulbs of length N where bulbs[i] = x means that on the (i+1)th day, we will turn on the bulb at position x where i is 0-indexed and x is 1-indexed.

Given an integer K, find out the minimum day number such that there exists two turned on bulbs that have exactly K bulbs between them that are all turned off.

If there isn't such day, return -1.

Example 1:

Input: 
bulbs: [1,3,2]
K: 1
Output: 2
Explanation:
On the first day: bulbs[0] = 1, first bulb is turned on: [1,0,0]
On the second day: bulbs[1] = 3, third bulb is turned on: [1,0,1]
On the third day: bulbs[2] = 2, second bulb is turned on: [1,1,1]
We return 2 because on the second day, there were two on bulbs with one off bulb between them.
"""

def kEmptySlots(flowers, k):
    days = [0] * len(flowers)
    print(days)
    for day, position in enumerate(flowers, 1):
        days[position - 1] = day
    print(days)
    ans = float('inf')
    left, right = 0, k+1
    while right < len(days):
        print(right,left)
        for i in range(left + 1, right):
            print(i,right,left)
            if days[i] < days[left] or days[i] < days[right]:
                left, right = i, i+k+1
                print("intro",left,right)
                break
        ans = min(ans, max(days[left], days[right]))
        left, right = right, right+k+1

    return ans if ans < float('inf') else -1


def judgePoint24(nums):
    
    def deep(cur):
        if len(cur) == 1:
            return set(cur)
        res = set()
        for i in range(1, len(cur)):
            left = deep(cur[:i])
            right = deep(cur[i:])
            
            for l in left:
                for r in right:
                    res.add(l + r)
                    res.add(l - r)
                    res.add(l * r)
                    if r != 0:
                        res.add(l / r)
        return res
    
    def dfs(cur, remain):
        if not remain:
            for res in deep(cur):
                if round(res, 4) == 24:
                    print("round",res)
                    return True
            return False
        else:
            for i in range(len(remain)):
                if dfs(cur + [remain[i]], remain[:i] + remain[i + 1:]):
                    return True
    return dfs([], nums)

"""
722. Remove Comments

Given a C++ program, remove comments from it. 
The program source is an array where source[i] is the i-th line of the source code. 
This represents the result of splitting the original source code string by the 
newline character \n.

"""

def removeComments(source):
    output, com = [],False
    for line in source:
        if not com:
            temp = []
        i = 0
        while i < len(line):
            if com:
                if line[i:i+2] == "*/":
                    com = False
                    i += 1
            else:
                if line[i:i+2] == "//":
                    break
                elif line[i:i+2] == "/*":
                    com = True
                    i += 1
                else:
                    temp.append(line[i])
            i += 1
        if temp and not com:
            output.append("".join(temp))
    
    return output

"""
727. Minimum Window Subsequence

Given strings S and T, find the minimum (contiguous) substring W of S, so that T is a subsequence of W.

If there is no such window in S that covers all characters in T, return the empty string "". 
If there are multiple such minimum-length windows, return the one with the left-most starting index.

Example 1:

Input: 
S = "abcdebdde", T = "bde"
Output: "bcde"
Explanation: 
"bcde" is the answer because it occurs before "bdde" which has the same length.
"deb" is not a smaller window because the elements of T in the window must occur in order.
"""
import collections
def minWindow(s, t):
    targets = collections.Counter(t)
    counts = collections.Counter()
    diffs = len(t)
    j = 0
    result = s + ' '
    for i, c in enumerate(s):
        #print(counts,diffs)
        counts[c] += 1
        if counts[c] <= targets[c]:
            diffs -= 1
        print(i,counts,targets)
        while diffs == 0:
            print("diff",diffs,counts,targets)
            temp = s[j:i+1]
            print("t",temp)
            if len(temp) < len(result):
                result = temp
            n = s[j]
            j += 1
            counts[n] -= 1
            print("Counts",counts,targets)
            if counts[n] < targets[n]:
                diffs += 1

    return result if len(result) <= len(s) else ''

"""
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
    size = len(nums)
    if size == 1:
        return 0
    if nums[0] > nums[1]:
        return 0
    if size == 2:
        return nums.index(max(nums))
    l,m,r = 0,1,2
    while r < size:
        if nums[m] > nums[l] and nums[m] > nums[r]:
            return m
        l += 1
        m += 1
        r += 1
    
    return m

"""
Given a non-empty string check if it can be constructed by taking a substring of it and appending multiple copies of the substring together. 
You may assume the given string consists of lowercase English letters only and its length will not exceed 10000.

 

Example 1:

Input: "abab"
Output: True
Explanation: It's the substring "ab" twice.
"""

def repeatedSubstringPattern(s): #Knuth Morris Pratt Algorithem
    pattern, left, right = [None] * len(s), 0 , 1
    while right < len(s):
        if s[left] == s[right]:
            pattern[right] = left
            left, right = left+1 , right+1
        
        elif left > 0 and pattern[left-1] is not None:
            left = pattern[left -1] + 1
        elif left > 0:
            left = 0
        else:
            right += 1
    
    max_repeated = pattern[-1]
    if max_repeated is None:
        return False
    cycle = len(s) - max_repeated - 1
    return len(s) % cycle == 0

"""
729. My Calendar I

Implement a MyCalendar class to store your events. 
A new event can be added if adding the event will not cause a double booking.
Your class will have the method, book(int start, int end). 
Formally, this represents a booking on the half open interval [start, end), 
the range of real numbers x such that start <= x < end.
A double booking happens when two events have some non-empty intersection (ie., there is some time that is common to both events.)
For each call to the method MyCalendar.book, return true if the event can be added to the calendar successfully without causing a double booking. 
Otherwise, return false and do not add the event to the calendar.
Your class will be called like this: MyCalendar cal = new MyCalendar(); MyCalendar.book(start, end)

Example 1:

MyCalendar();
MyCalendar.book(10, 20); // returns true
MyCalendar.book(15, 25); // returns false
MyCalendar.book(20, 30); // returns true
Explanation: 
The first event can be booked.  The second can't because time 15 is already booked by another event.
The third event can be booked, as the first event takes every time less than 20, but not including 20.

Time Complexity (Java): O(N \log N)O(NlogN), where NN is the number of events booked. 
For each new event, we search that the event is legal in O(\log N)O(logN) time, then insert it in O(1)O(1) time.

"""
class MyCalendar(object): #O(N**2), O(N)- Space
    def __init__(self):
        self.calendar = []

    def book(self, start, end):
        for s, e in self.calendar:
            if s < end and start < e:
                return False
        self.calendar.append((start, end))
        return True
    
class Node: #Balanced Tree to sort the event. O(NLogn)
    __slots__ = 'start', 'end', 'left', 'right'
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.left = self.right = None

    def insert(self, node):
        if node.start >= self.end:
            if not self.right:
                self.right = node
                return True
            return self.right.insert(node)
        elif node.end <= self.start:
            if not self.left:
                self.left = node
                return True
            return self.left.insert(node)
        else:
            return False

class MyCalendar(object):
    def __init__(self):
        self.root = None

    def book(self, start, end):
        if self.root is None:
            self.root = Node(start, end)
            return True
        return self.root.insert(Node(start, end))
    
    
"""
732. My Calendar III

A k-booking happens when k events have some non-empty intersection (i.e., there is some time that is common to all k events.)

You are given some events [start, end), after each given event, return an integer k representing the maximum k-booking between all the previous events.

Implement the MyCalendarThree class:

MyCalendarThree() Initializes the object.
int book(int start, int end) Returns an integer k representing the largest integer such that there exists a k-booking in the calendar.

Example 1:

Input
["MyCalendarThree", "book", "book", "book", "book", "book", "book"]
[[], [10, 20], [50, 60], [10, 40], [5, 15], [5, 10], [25, 55]]
Output
[null, 1, 1, 2, 3, 3, 3]

Explanation
MyCalendarThree myCalendarThree = new MyCalendarThree();
myCalendarThree.book(10, 20); // return 1, The first event can be booked and is disjoint, so the maximum k-booking is a 1-booking.
myCalendarThree.book(50, 60); // return 1, The second event can be booked and is disjoint, so the maximum k-booking is a 1-booking.
myCalendarThree.book(10, 40); // return 2, The third event [10, 40) intersects the first event, and the maximum k-booking is a 2-booking.
myCalendarThree.book(5, 15); // return 3, The remaining events cause the maximum K-booking to be only a 3-booking.
myCalendarThree.book(5, 10); // return 3
myCalendarThree.book(25, 55); // return 3
"""
    
class MyCalendarThree(object):

    def __init__(self):
        self.delta = collections.Counter()

    def book(self, start, end):
        self.delta[start] += 1
        self.delta[end] -= 1
        active = ans = 0
        for x in sorted(self.delta):
            active += self.delta[x]
            if active > ans: ans = active

        return ans

"""
752. Open the Lock

You have a lock in front of you with 4 circular wheels. Each wheel has 10 slots: '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'. The wheels can rotate freely and wrap around: for example we can turn '9' to be '0', or '0' to be '9'. Each move consists of turning one wheel one slot.

The lock initially starts at '0000', a string representing the state of the 4 wheels.

You are given a list of deadends dead ends, meaning if the lock displays any of these codes, the wheels of the lock will stop turning and you will be unable to open it.

Given a target representing the value of the wheels that will unlock the lock, return the minimum total number of turns required to open the lock, or -1 if it is impossible.


Example 1:

Input: deadends = ["0201","0101","0102","1212","2002"], target = "0202"
Output: 6
Explanation:
A sequence of valid moves would be "0000" -> "1000" -> "1100" -> "1200" -> "1201" -> "1202" -> "0202".
Note that a sequence like "0000" -> "0001" -> "0002" -> "0102" -> "0202" would be invalid,
because the wheels of the lock become stuck after the display becomes the dead end "0102".

"""
from collections import deque
class Solution:    

    def getNextCombinations(self,combination):
        nextCombinations = []        
        for i, value in enumerate(combination):
            for valueDifference in (-1, 1):
                rotatedValue = (int(value) + valueDifference) % 10
                nextCombinations.append(combination[:i] + str(rotatedValue) + combination[i + 1:])            
        return nextCombinations
    
    def openLock(self,deadends,target):
        deadEnds = set(deadends)
        queue = deque([('0000', 0)])
        seen = {'0000'}        
		# Keep searching while there are still valid paths.
        while queue:
            combination, depth = queue.popleft()
			# We've found our target.
            if combination == target:
                return depth
			# We've hit a dead end and terminate searching from this node.
            if combination in deadEnds:
                continue
                
			# Get the nodes connected to the current node.
            for nextCombination in self.getNextCombinations(combination):
				# If we've not visited the node before, then add it to the queue to perform BFS on again.
                if nextCombination not in seen:
                    seen.add(nextCombination)
                    queue.append((nextCombination, depth + 1))
                    
		# We searched all valid routes but could not get to the target.
        return -1
    
"""
809. Expressive Words

Sometimes people repeat letters to represent extra feeling. For example:

"hello" -> "heeellooo"
"hi" -> "hiiii"
In these strings like "heeellooo", we have groups of adjacent letters that are all the same: "h", "eee", "ll", "ooo".

You are given a string s and an array of query strings words. A query word is stretchy if it can be made to be equal to s by any number of applications of the following extension operation: choose a group consisting of characters c, and add some number of characters c to the group so that the size of the group is three or more.

For example, starting with "hello", we could do an extension on the group "o" to get "hellooo", but we cannot get "helloo" since the group "oo" has a size less than three. Also, we could do another extension like "ll" -> "lllll" to get "helllllooo". If s = "helllllooo", then the query word "hello" would be stretchy because of these two extension operations: query = "hello" -> "hellooo" -> "helllllooo" = s.
Return the number of query strings that are stretchy.

 

Example 1:

Input: s = "heeellooo", words = ["hello", "hi", "helo"]
Output: 1
Explanation: 
We can extend "e" and "o" in the word "hello" to get "heeellooo".
We can't extend "helo" to get "heeellooo" because the group "ll" is not size 3 or more.
"""
def expressiveWords(s,words):
    numStretchyWords = 0
    wordA = s
    for wordB in words:
        i = 0
        j = 0
        while i < len(wordA) and j < len(wordB):
            if wordA[i] != wordB[j]:
                break
            countA = 1
            countB = 1
            while (i+1 < len(wordA))and (wordA[i+1]==wordA[i]):
                countA += 1
                i += 1
            while (j+1 < len(wordB)) and (wordB[j+1] == wordB[j]):
                countB += 1
                j += 1
            
            if countA < countB or (countA > countB and countA < 3):
                break
            i += 1
            j += 1
            
            if i == len(wordA) and j == len(wordB):
                numStretchyWords += 1
            
    return numStretchyWords


"""
833. Find And Replace in String

To some string S, we will perform some replacement operations that replace groups of letters with new ones (not necessarily the same size).

Each replacement operation has 3 parameters: a starting index i, a source word x and a target word y.  The rule is that if x starts at position i in the original string S, then we will replace that occurrence of x with y.  If not, we do nothing.

For example, if we have S = "abcd" and we have some replacement operation i = 2, x = "cd", y = "ffff", then because "cd" starts at position 2 in the original string S, we will replace it with "ffff".

Using another example on S = "abcd", if we have both the replacement operation i = 0, x = "ab", y = "eee", as well as another replacement operation i = 2, x = "ec", y = "ffff", this second operation does nothing because in the original string S[2] = 'c', which doesn't match x[0] = 'e'.

All these operations occur simultaneously.  It's guaranteed that there won't be any overlap in replacement: for example, S = "abc", indexes = [0, 1], sources = ["ab","bc"] is not a valid test case.

Example 1:

Input: S = "abcd", indexes = [0,2], sources = ["a","cd"], targets = ["eee","ffff"]
Output: "eeebffff"
Explanation: "a" starts at index 0 in S, so it's replaced by "eee".
"cd" starts at index 2 in S, so it's replaced by "ffff".

"""

def findReplaceString(s, indexes,sources,target):
    ans = []
    i = 0
    zipped = sorted(zip(indexes,sources,target))
    for index,source,target in zipped:
        while i < index:
            ans.append(s[i])
            i += 1
        
        if source == s[i:i+len(source)]:
            ans += target.split()
        else:
            ans += s[i:i+len(source)].split()
        
        i += len(source)
    
    ans += s[i:].split()
    return ''.join(ans)

"""
846. Hand of Straights

Alice has a hand of cards, given as an array of integers.

Now she wants to rearrange the cards into groups so that each group is size W, and consists of W consecutive cards.

Return true if and only if she can.

Example 1:

Input: hand = [1,2,3,6,2,3,4,7,8], W = 3
Output: true
Explanation: Alice's hand can be rearranged as [1,2,3],[2,3,4],[6,7,8].
"""
from collections import OrderedDict 
def isNStraightHand(hand, W):
    hand.sort()
    d=OrderedDict()
    for i in hand:
        if i not in d:
            d[i]=1
        else:
            d[i]+=1
    while True:
        for i in d:
            x=i
            d[x]=d[x]-1
            if d[x]==0:
                del d[x]
            for j in range(W-1):
                if x+1 not in d:
                    return False
                else:
                    d[x+1]=d[x+1]-1
                    if d[x+1]==0:
                        del d[x+1]
                x+=1            
            if d=={}:
                return True
            break
    return True
"""
946. Validate Stack Sequences

Given two sequences pushed and popped with distinct values, return true if and only if this could have been the result of a sequence of push and pop operations on an initially empty stack.

Example 1:

Input: pushed = [1,2,3,4,5], popped = [4,5,3,2,1]
Output: true
Explanation: We might do the following sequence:
push(1), push(2), push(3), push(4), pop() -> 4,
push(5), pop() -> 5, pop() -> 3, pop() -> 2, pop() -> 1
"""
def validateStackSequences(pushed, popped):
    j = 0
    stack = []
    for x in pushed:
        stack.append(x)
        while stack and j < len(popped) and stack[-1] == popped[j]:
            stack.pop()
            j += 1

    return j == len(popped)
"""
1060. Missing Element in Sorted Array

Given a sorted array A of unique numbers, find the K-th missing number starting from the leftmost number of the array.

Example 1:

Input: A = [4,7,9,10], K = 1
Output: 5
Explanation: 
The first missing number is 5.
"""
def missingElement(nums, k):
    # think about when no missing numbers, then would understand why "-idx" works. 
    #  # Return how many numbers are missing until nums[idx]
    missing_cnt = lambda x: nums[x] - nums[0] - x
    print(missing_cnt(len(nums)-1))
    if k > missing_cnt(len(nums)-1):
        return nums[-1] + k - missing_cnt(len(nums)-1)
    # find idx where missing count is equal or more than k
    l, h = 0, len(nums)
    while l < h:
        m = (l+h)//2
        cnt = missing_cnt(m)
        if cnt < k:
            l = m+1
        else:
            h = m      
    return nums[h-1]+k-missing_cnt(h-1)

"""
Count Submatrices With All Ones
Given a rows * columns matrix mat of ones and zeros, return how many submatrices have all ones.

 

Example 1:

Input: mat = [[1,0,1],
              [1,1,0],
              [1,1,0]]
Output: 13
Explanation:
There are 6 rectangles of side 1x1.
There are 2 rectangles of side 1x2.
There are 3 rectangles of side 2x1.
There is 1 rectangle of side 2x2. 
There is 1 rectangle of side 3x1.
Total number of rectangles = 6 + 2 + 3 + 1 + 1 = 13.
"""
def numSubmat(mat):
    R = len(mat)
    C = len(mat[0])
    count = 0
    if not R or not C:
        return 0
    dp = [[0]* (C+1) for _ in range(R+1)]
    print(dp)
    for i in range(1, R+1):
        for j in range(1,C+1):
            if mat[i-1][j-1]:
                print(i,j)
                dp[i][j] = dp[i][j-1]+1
                count += dp[i][j]
                print(dp)
                curMin = dp[i][j]
                for k in range(i-1,-1,-1):
                    print("k",k,j)
                    if dp[k][j] == 0:
                        break
                    curMin = min(curMin, dp[k][j])
                    count += curMin
    return count
"""
Given a time represented in the format "HH:MM", form the next closest time by reusing the current digits. There is no limit on how many times a digit can be reused.

You may assume the given input string is always valid. For example, "01:34", "12:09" are all valid. "1:34", "12:9" are all invalid.

 

Example 1:

Input: time = "19:34"
Output: "19:39"
Explanation: The next closest time choosing from digits 1, 9, 3, 4, is 19:39, which occurs 5 minutes later.
It is not 19:33, because this occurs 23 hours and 59 minutes later.
"""
def nextClosestTime(self, time: str) -> str:
    x = set(time)
    t= int(time[0:2])*60 
    t+=int(time[3:])
    r = set()
    print(x,t)
    for i in x:
        try:
            r.add(int(i))
        except:
            f=0
    print(r)
    while True:
        t = (t+1)%(24*60)
        temp =  [int(t/60//10),int(t//60%10),int(t%60//10),int(t%60%10)]
        print(t,temp)
        flag =True
        for i in temp:
            if i not in r:
                print("i",i)
                flag=False
        if flag:
            return "".join(str(p) for p in temp[0:2])+":"+"".join(str(p) for p in temp[2:])
"""
300. Longest Increasing Subsequence

Given an integer array nums, return the length of the longest strictly increasing subsequence.

A subsequence is a sequence that can be derived from an array by deleting some or no elements without changing the order of the remaining elements. For example, [3,6,2,7] is a subsequence of the array [0,3,1,6,2,2,7].

Example 1:

Input: nums = [10,9,2,5,3,7,101,18]
Output: 4
Explanation: The longest increasing subsequence is [2,3,7,101], therefore the length is 4.
"""
import math
    
def lengthOfLIS(nums):
    M = [nums[0]]*len(nums)
    hi = 0
    for i in range(1, len(nums)):
        print(M,i,hi)
        if nums[i] <= M[0]:
            M[0] = nums[i]
        elif nums[i] > M[hi]:
            hi += 1
            M[hi] = nums[i]
        else:
            print(i,nums[i],M,hi)
            lo, hii = 0, hi
            while hii > lo:
                mid = math.ceil((hii+lo)/2)
                print(lo,hii,mid)
                if M[mid] >= nums[i]:
                    hii = mid -1
                else:
                    lo = mid
            M[lo+1] = nums[i]
    print(M)
    return hi+1
"""
410. Split Array Largest Sum

Given an array nums which consists of non-negative integers and an integer m, you can split the array into m non-empty continuous subarrays.

Write an algorithm to minimize the largest sum among these m subarrays.

Example 1:

Input: nums = [7,2,5,10,8], m = 2
Output: 18
Explanation:
There are four ways to split nums into two subarrays.
The best way is to split it into [7,2,5] and [10,8],
where the largest sum among the two subarrays is only 18.
"""
def splitArray(nums, m):
    """
    :type nums: List[int]
    :type m: int
    :rtype: int
    """
    # Edge case
    if m == len(nums):
        return max(nums)
    
    def satisfy_condition(cap):
        arrCnt = 1
        subSum = 0
        for n in nums:
            subSum += n
            if subSum > cap:
                subSum = n
                arrCnt += 1
        return arrCnt <= m
    
    l, r = max(nums), sum(nums)
    while l < r:
        mid = l + (r-l)//2
        if satisfy_condition(mid):
            r = mid
        else:
            l = mid + 1
    return l

"""
418. Sentence Screen Fitting

Given a rows x cols screen and a sentence represented as a list of strings, return the number of times the given sentence can be fitted on the screen.

The order of words in the sentence must remain unchanged, and a word cannot be split into two lines. A single space must separate two consecutive words in a line.

Example 1:

Input: sentence = ["hello","world"], rows = 2, cols = 8
Output: 1
Explanation:
hello---
world---
The character '-' signifies an empty space on the screen.

Solution 1: Brute Force (Time Limit Exceeded)

Let's fill words row by row.
"""
def wordsTyping(sentence, rows, cols):
    n = len(sentence)
    ans = 0
    wordIdx = 0
    for _ in range(rows):
        c = 0  # Reset for new row
        while c + len(sentence[wordIdx]) <= cols:
            c += len(sentence[wordIdx]) + 1
            wordIdx += 1
            if wordIdx == n:
                ans += 1
                wordIdx = 0
    return ans
"""
Complexity:

Time: O(rows * cols), where rows <= 2*10^4 is number of rows, cols <= 2*10^4 is number of columns.
Space: O(1)
✔️ Solution 2: Dynamic Programming

We can see that the while loop is re-calculated each time -> We can cache that result by using dp.
Let dp[i] is (nextIdx, times) when the word ith at the beginning of the row.
"""
def wordsTyping(sentence,rows,cols):
    n = len(sentence)
    def dp(i):  # Return (nextIndex, times) if the word at ith is the beginning of the row
        c = 0
        times = 0
        while c + len(sentence[i]) <= cols:
            c += len(sentence[i]) + 1
            i += 1
            if i == n:
                times += 1
                i = 0
        return i, times
    ans = 0
    wordIdx = 0
    for _ in range(rows):
        ans += dp(wordIdx)[1]
        wordIdx = dp(wordIdx)[0]
    return ans
"""
Complexity:

Time: O(N * cols + rows), where N <= 100 is number of words in the sentence, rows <= 2*10^4 is number of rows, cols <= 2*10^4 is number of columns.
Space: O(N)"""
"""
465. Optimal Account Balancing

You are given an array of transactions transactions where transactions[i] = [fromi, toi, amounti] indicates that the person with ID = fromi gave amounti $ to the person with ID = toi.

Return the minimum number of transactions required to settle the debt.

Example 1:

Input: transactions = [[0,1,10],[2,0,5]]
Output: 2
Explanation:
Person #0 gave person #1 $10.
Person #2 gave person #0 $5.
Two transactions are needed. One way to settle the debt is person #1 pays person #0 and #2 $5 each.
"""

def minTransfers(transactions):
    
    balances = collections.defaultdict(int)
    for sender, receiver, amount in transactions: 
        balances[sender] -= amount 
        balances[receiver] += amount 
    print(balances)
    balances = balances.values()
    
    givers = [b for b in balances if b > 0]
    receivers = [b for b in balances if b < 0]
    print((givers, receivers))
    res = float('inf')
    def dfs(givers, receivers, si, sofar):
        nonlocal res
        if si == len(givers): 
            res = min(res, sofar)
        elif sofar >= res: 
            return
        for j in range(len(receivers)): 
            if receivers[j] < 0: 
                amount = min(abs(givers[si]), abs(receivers[j])) 
                givers[si] -= amount 
                receivers[j] += amount 
                nsi = si 
                if givers[si] == 0: 
                    nsi += 1
                dfs(givers, receivers, nsi, sofar+1)
                givers[si] += amount 
                receivers[j] -= amount
                
        
    dfs(givers, receivers, 0, 0)
    return res
"""
562. Longest Line of Consecutive One in Matrix

Given an m x n binary matrix mat, return the length of the longest line of consecutive one in the matrix.

The line could be horizontal, vertical, diagonal, or anti-diagonal.

Example 1:
Input: mat = [[0,1,1,0],[0,1,1,0],[0,0,0,1]]
Output: 3
"""

def longestLine(mat):
    nrow, ncol = len(mat), len(mat[0])
    # dp[i][j] is the max length of lines that start from i, j and span in the 
    # horizontal, vertival, diagonal, and anti-diagonal directions
    dp = [[[0, 0, 0, 0] for _ in range(ncol)] for _ in range(nrow)]
    if mat[nrow-1][ncol-1] == 1:
        res = 1
        dp[nrow-1][ncol-1] = [1, 1, 1, 1]
    else:
        res = 0
    for i in range(nrow-1, -1, -1):
        for j in range(ncol-1, -1, -1):
            if i == nrow-1 and j == ncol-1:
                continue
            if mat[i][j] == 1:
                dp[i][j][0] = dp[i][j+1][0] + 1 if j+1<ncol else 1
                dp[i][j][1] = dp[i+1][j][1] + 1 if i+1<nrow else 1
                dp[i][j][2] = dp[i+1][j+1][2] + 1 if i+1<nrow and j+1<ncol else 1
                dp[i][j][3] = dp[i+1][j-1][3] + 1 if i+1<nrow and j-1>=0 else 1
                res = max([res]+dp[i][j])
    return res

"""
593. Valid Square

Given the coordinates of four points in 2D space p1, p2, p3 and p4, return true if the four points construct a square.

The coordinate of a point pi is represented as [xi, yi]. The input is not given in any order.

A valid square has four equal sides with positive length and four equal angles (90-degree angles).

 

Example 1:

Input: p1 = [0,0], p2 = [1,1], p3 = [1,0], p4 = [0,1]
Output: true
Pretty straightforward, use a stack to compute all dists (tho you could also manually write it out). Then check that sides are equal, diagonals are equal, and that pythagorean identity is satisfied.

"""
def validSquare(p1, p2, p3, p4):
    stack = [p1,p2,p3,p4]
    d = []
    while stack:
        x1 = stack.pop()
        for x2 in stack:
            d.append(getDist(x1,x2))
    
    d.sort()
    equalSides = d[0] == d[1] == d[2] == d[3]
    equalDiags = d[4] == d[5]
    pythagorean = d[0]+d[1] == d[4] and d[4] > d[0]
 
    return equalSides and equalDiags and pythagorean
        
        
def getDist(x1,x2):
    return (x1[0]-x2[0])**2 + (x1[1]-x2[1])**2 

"""
652. Find Duplicate Subtrees

Given the root of a binary tree, return all duplicate subtrees.

For each kind of duplicate subtrees, you only need to return the root node of any one of them.

Two trees are duplicate if they have the same structure with the same node values.

 

Example 1:


Input: root = [1,2,3,4,null,2,4,null,null,4]
Output: [[2,4],[4]]
"""
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def findDuplicateSubtrees(self, root):

        seen = defaultdict(int)
        output = []

        def dfs(node):
            if not node:
                return
            subtree = tuple([dfs(node.left), node.val, dfs(node.right)])
            if seen[subtree] == 1:
                output.append(node)
            seen[subtree] += 1
            return subtree

        dfs(root)
        return output
    
"""
695. Max Area of Island

You are given an m x n binary matrix grid. An island is a group of 1's (representing land) connected 4-directionally (horizontal or vertical.) You may assume all four edges of the grid are surrounded by water.

The area of an island is the number of cells with a value 1 in the island.

Return the maximum area of an island in grid. If there is no island, return 0.

 

Example 1:


Input: grid = [[0,0,1,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,0,1,1,1,0,0,0],[0,1,1,0,1,0,0,0,0,0,0,0,0],[0,1,0,0,1,1,0,0,1,0,1,0,0],[0,1,0,0,1,1,0,0,1,1,1,0,0],[0,0,0,0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,1,1,0,0,0],[0,0,0,0,0,0,0,1,1,0,0,0,0]]
Output: 6
Explanation: The answer is not 11, because the island must be connected 4-directionally.


"""
"""
Time Complexity: O(R*C)O(R∗C), where RR is the number of rows in the given grid, and CC is the number of columns. We visit every square once.

Space complexity: O(R*C), the space used by seen to keep track of visited squares, and the space used by the call stack during our recursion.
"""
def maxAreaOfIsland(self, grid):
    seen = set()
    def area(r, c):
        if not (0 <= r < len(grid) and 0 <= c < len(grid[0])
                and (r, c) not in seen and grid[r][c]):
            return 0
        seen.add((r, c))
        return (1 + area(r+1, c) + area(r-1, c) +
                area(r, c-1) + area(r, c+1))

    return max(area(r, c)
               for r in range(len(grid))
               for c in range(len(grid[0])))

#O(1))
def maxAreaOfIsland(grid):
    row=len(grid)
    column=len(grid[0])

    def dfs(x,y):
        
        if(x<0 or y<0 or x>=row or y>=column):
            return 0
        elif(grid[x][y]==0):
            return 0
        else:
            grid[x][y]=0
            return 1+dfs(x+1,y)+dfs(x,y+1)+dfs(x-1,y)+dfs(x,y-1)
    result=0
    for i in range(row):
        for k in range(column):
            if(grid[i][k]==1):
                result=max(result,dfs(i,k))
    return (result)
"""
715. Range Module

A Range Module is a module that tracks ranges of numbers. Design a data structure to track the ranges represented as half-open intervals and query about them.

A half-open interval [left, right) denotes all the real numbers x where left <= x < right.

Implement the RangeModule class:

RangeModule() Initializes the object of the data structure.
void addRange(int left, int right) Adds the half-open interval [left, right), tracking every real number in that interval. Adding an interval that partially overlaps with currently tracked numbers should add any numbers in the interval [left, right) that are not already tracked.
boolean queryRange(int left, int right) Returns true if every real number in the interval [left, right) is currently being tracked, and false otherwise.
void removeRange(int left, int right) Stops tracking every real number currently being tracked in the half-open interval [left, right).
 

Example 1:

Input
["RangeModule", "addRange", "removeRange", "queryRange", "queryRange", "queryRange"]
[[], [10, 20], [14, 16], [10, 14], [13, 15], [16, 17]]
Output
[null, null, null, true, false, true]

Explanation
RangeModule rangeModule = new RangeModule();
rangeModule.addRange(10, 20);
rangeModule.removeRange(14, 16);
rangeModule.queryRange(10, 14); // return True,(Every number in [10, 14) is being tracked)
rangeModule.queryRange(13, 15); // return False,(Numbers like 14, 14.03, 14.17 in [13, 15) are not being tracked)
rangeModule.queryRange(16, 17); // return True, (The number 16 in [16, 17) is still being tracked, despite the remove operation)

"""
from sortedcontainers import SortedDict

class RangeModule:#O(n)

    def __init__(self):
        self.track = SortedDict()

    def addRange(self, left: int, right: int) -> None:
        start, end = left, right
        drop = []
        # find overlap range
        for s, e in self.track.items():
            if e < start:
                continue
            
            if s > end:
                continue
            # update boundary
            start = min(s, start)
            end = max(e, end)
			# track overlap interval start time
            drop.append(s)
        
		# delete overlap interval
        while drop:
            del self.track[drop.pop()]
        
		#insert new one
        self.track[start] = end

    def queryRange(self, left: int, right: int) -> bool:
		#check if there is a interval cover left and right
        for s, e in self.track.items():
            if s <= left and right <= e:
                return True
        
        return False
    def removeRange(self, left: int, right: int) -> None:
		# find overlap intervals and delete them
        start, end = left, right
        drop = []
        for s, e in self.track.items():
            if e < start:
                continue
            
            if s > end:
                continue
                
            start = min(start, s)
            end = max(end, e)
            drop.append(s)
        
        while drop:
            del self.track[drop.pop()]
        # insert new interval
        if start < left:
            self.track[start] = left
        
        if end > right:
            self.track[right] = end
"""
735. Asteroid Collision

We are given an array asteroids of integers representing asteroids in a row.

For each asteroid, the absolute value represents its size, and the sign represents its direction (positive meaning right, negative meaning left). Each asteroid moves at the same speed.

Find out the state of the asteroids after all collisions. If two asteroids meet, the smaller one will explode. If both are the same size, both will explode. Two asteroids moving in the same direction will never meet.

Example 1:

Input: asteroids = [5,10,-5]
Output: [5,10]
Explanation: The 10 and -5 collide resulting in 10. The 5 and 10 never collide.
"""
def asteroidCollision(asteroids):
    stack = []
    for a in asteroids:
        while stack and a < 0 and stack[-1] > 0:
            diff = a + stack[-1]
            if diff > 0:
                a = 0
            elif diff < 0:
                stack.pop()
            else:
                a = 0
                stack.pop()
        if a :
            stack.append(a)
    return stack
"""
792. Number of Matching Subsequences

Given a string s and an array of strings words, return the number of words[i] that is a subsequence of s.

A subsequence of a string is a new string generated from the original string with some characters (can be none) deleted without changing the relative order of the remaining characters.

For example, "ace" is a subsequence of "abcde".
 
Example 1:

Input: s = "abcde", words = ["a","bb","acd","ace"]
Output: 3
Explanation: There are three strings in words that are a subsequence of s: "a", "acd", "ace".
"""
def numMatchingSubseq(s, words):
    # speed: O(W * N), where W = len(words), N = len(s)
    # space: O(1)

    result = 0
    for word in words:
        pos = -1
        for ch in word:  # the worst number of iterations is len(s), it includes internal operations in s.find()
            pos = s.find(ch, pos+1)
            if pos < 0:
					# word failed matching as a subsequence
                break
        else:
				# word matched as a subsequence
            result += 1

    return result
"""
853. Car Fleet

There are n cars going to the same destination along a one-lane road. The destination is target miles away.

You are given two integer array position and speed, both of length n, where position[i] is the position of the ith car and speed[i] is the speed of the ith car (in miles per hour).

A car can never pass another car ahead of it, but it can catch up to it, and drive bumper to bumper at the same speed.

The distance between these two cars is ignored (i.e., they are assumed to have the same position).

A car fleet is some non-empty set of cars driving at the same position and same speed. Note that a single car is also a car fleet.

If a car catches up to a car fleet right at the destination point, it will still be considered as one car fleet.

Return the number of car fleets that will arrive at the destination.

 

Example 1:

Input: target = 12, position = [10,8,0,5,3], speed = [2,4,1,1,3]
Output: 3
Explanation: 
The cars starting at 10 and 8 become a fleet, meeting each other at 12.
The car starting at 0 doesn't catch up to any other car, so it is a fleet by itself.
The cars starting at 5 and 3 become a fleet, meeting each other at 6.
Note that no other cars meet these fleets before the destination, so the answer is 3.
"""
def carFleet(target, position, speed):
 
        cars = []
        for i in range(len(position)):
            cars.append((position[i], speed[i]))
        
        cars.sort(key=lambda x: (x[0], x[1]), reverse=True) # desc order : closer to traget comes first
        stack = []
        for x, v in cars:
            dist = target - x # remaning distance to tagret
            if not stack:
                stack.append(dist/v) # arrivalTime = dist/v
            elif dist/v > stack[-1]: # car arrives late -> thus does not join previous fleet and forms its own fleet
                stack.append(dist/v)
            # if curr arrivalTime is <= prev arrivalTime -> then curr car joins prev fleet and gets discolved into it (aka we don't need to do anything)
        return len(stack)
"""
875. Koko Eating Bananas

Koko loves to eat bananas. There are n piles of bananas, the ith pile has piles[i] bananas. The guards have gone and will come back in h hours.

Koko can decide her bananas-per-hour eating speed of k. Each hour, she chooses some pile of bananas and eats k bananas from that pile. If the pile has less than k bananas, she eats all of them instead and will not eat any more bananas during this hour.

Koko likes to eat slowly but still wants to finish eating all the bananas before the guards return.

Return the minimum integer k such that she can eat all the bananas within h hours.

Example 1:

Input: piles = [3,6,7,11], h = 8
Output: 4
"""

def minEatingSpeed(piles, h):
    l=1
    result=r=max(piles)

    while l<=r:
        mid=(l+r)//2
        count=0
        for ele in piles:
            count+=math.ceil(ele/mid)
        print(count,mid,l,r,h)
        if count<=h:
            result=min(result,mid)
            r=mid-1
        else:
            l=mid+1
    return result

"""
900. RLE Iterator

We can use run-length encoding (i.e., RLE) to encode a sequence of integers. In a run-length encoded array of even length encoding (0-indexed), for all even i, encoding[i] tells us the number of times that the non-negative integer value encoding[i + 1] is repeated in the sequence.

For example, the sequence arr = [8,8,8,5,5] can be encoded to be encoding = [3,8,2,5]. encoding = [3,8,0,9,2,5] and encoding = [2,8,1,8,2,5] are also valid RLE of arr.
Given a run-length encoded array, design an iterator that iterates through it.

Implement the RLEIterator class:

RLEIterator(int[] encoded) Initializes the object with the encoded array encoded.
int next(int n) Exhausts the next n elements and returns the last element exhausted in this way. If there is no element left to exhaust, return -1 instead.

Example 1:

Input
["RLEIterator", "next", "next", "next", "next"]
[[[3, 8, 0, 9, 2, 5]], [2], [1], [1], [2]]
Output
[null, 8, 8, 5, -1]

Explanation
RLEIterator rLEIterator = new RLEIterator([3, 8, 0, 9, 2, 5]); // This maps to the sequence [8,8,8,5,5].
rLEIterator.next(2); // exhausts 2 terms of the sequence, returning 8. The remaining sequence is now [8, 5, 5].
rLEIterator.next(1); // exhausts 1 term of the sequence, returning 8. The remaining sequence is now [5, 5].
rLEIterator.next(1); // exhausts 1 term of the sequence, returning 5. The remaining sequence is now [5].
rLEIterator.next(2); // exhausts 2 terms, returning -1. This is because the first term exhausted was 5,
but the second term did not exist. Since the last term exhausted does not exist, we return -1.
"""

class RLEIterator:

    def __init__(self, encoding):
        self.encoding = encoding
        self.cur_pointer = 0

    def next(self, n):
        while self.cur_pointer < len(self.encoding) and n > 0:
            if self.encoding[self.cur_pointer] >= n:
                self.encoding[self.cur_pointer] -= n
                return self.encoding[self.cur_pointer + 1]
            else:
                n -= self.encoding[self.cur_pointer]
                self.cur_pointer += 2
        return -1
    
"""
951. Flip Equivalent Binary Trees

For a binary tree T, we can define a flip operation as follows: choose any node, and swap the left and right child subtrees.

A binary tree X is flip equivalent to a binary tree Y if and only if we can make X equal to Y after some number of flip operations.

Given the roots of two binary trees root1 and root2, return true if the two trees are flip equivelent or false otherwise.

Example 1:

Flipped Trees Diagram
Input: root1 = [1,2,3,4,5,6,null,null,null,7,8], root2 = [1,3,2,null,6,4,5,null,null,null,null,8,7]
Output: true
Explanation: We flipped at nodes with values 1, 3, and 5.
"""
class Solution(object):
    def flipEquiv(self, root1, root2):
        if root1 is root2:
            return True
        if not root1 or not root2 or root1.val != root2.val:
            return False

        return (self.flipEquiv(root1.left, root2.left) and
                self.flipEquiv(root1.right, root2.right) or
                self.flipEquiv(root1.left, root2.right) and
                self.flipEquiv(root1.right, root2.left))
"""
1048. Longest String Chain

You are given an array of words where each word consists of lowercase English letters.

wordA is a predecessor of wordB if and only if we can insert exactly one letter anywhere in wordA without changing the order of the other characters to make it equal to wordB.

For example, "abc" is a predecessor of "abac", while "cba" is not a predecessor of "bcad".
A word chain is a sequence of words [word1, word2, ..., wordk] with k >= 1, where word1 is a predecessor of word2, word2 is a predecessor of word3, and so on. A single word is trivially a word chain with k == 1.

Return the length of the longest possible word chain with words chosen from the given list of words.

 

Example 1:

Input: words = ["a","b","ba","bca","bda","bdca"]
Output: 4
Explanation: One of the longest word chains is ["a","ba","bda","bdca"].
O(N*L^2)
"""
def longestStrChain(words):
    wordset = set(words)
    memo = {}
    def dfs(current_word):
        if current_word not in wordset: return 0
        max_length = 1
        for i in range(len(current_word)+1):
            for c in "abcdefghijklmnopqrstuvwxyz":
                new_word = current_word[:i] + c + current_word[i:]
                if new_word in wordset:
                    if new_word in memo: 
                        max_length = max(max_length, 1 + memo[new_word])
                    else:
                        max_length = max(max_length, 1 + dfs(new_word))
        memo[current_word] = max_length
        return max_length
    max_chain = 0
    for word in words:
        if word not in memo:
            max_chain = max(max_chain, dfs(word))
            
    return max_chain

"""
1110. Delete Nodes And Return Forest

Given the root of a binary tree, each node in the tree has a distinct value.

After deleting all nodes with a value in to_delete, we are left with a forest (a disjoint union of trees).

Return the roots of the trees in the remaining forest. You may return the result in any order.

Example 1:

Input: root = [1,2,3,4,5,6,7], to_delete = [3,5]
Output: [[1,2,null,4],[6],[7]]

Convert to_delete list to a set because we need to check whether current node has to be deleted or not. So, better to keep avg O(1) time. Change values of nodes to be deleted to '#' while going from top to bottom.
While coming back, check whether the current node or it's children are '#'. If children are '#', remove parent child connection. If curr node is '#', then append children to the final list.
"""
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def delNodes(self, root, to_delete):
        ans = []
        deleted = set(to_delete)
        
        def DFS(node):
            if node.val in deleted:
                node.val = '#'
                
            for child in [node.left, node.right]:
                if child:
                    DFS(child)
            
            if node.left and node.left.val == '#':
                node.left = None
            if node.right and node.right.val == '#':
                node.right = None
                
            if node.val == '#':
                if node.left:
                    ans.append(node.left)
                if node.right:
                    ans.append(node.right)
        
        DFS(root)
        if root.val != '#':
            ans.append(root)
        return ans
"""
1146. Snapshot Array

Implement a SnapshotArray that supports the following interface:

SnapshotArray(int length) initializes an array-like data structure with the given length.  Initially, each element equals 0.
void set(index, val) sets the element at the given index to be equal to val.
int snap() takes a snapshot of the array and returns the snap_id: the total number of times we called snap() minus 1.
int get(index, snap_id) returns the value at the given index, at the time we took the snapshot with the given snap_id
 
Example 1:

Input: ["SnapshotArray","set","snap","set","get"]
[[3],[0,5],[],[0,6],[0,0]]
Output: [null,null,0,null,5]
Explanation: 
SnapshotArray snapshotArr = new SnapshotArray(3); // set the length to be 3
snapshotArr.set(0,5);  // Set array[0] = 5
snapshotArr.snap();  // Take a snapshot, return snap_id = 0
snapshotArr.set(0,6);
snapshotArr.get(0,0);  // Get the value of array[0] with snap_id = 0, return 5
"""  

"""
Complexity:

Time:
Constructor, set, snap: O(1)
get: O(logN)
Space: O(N)
"""
class SnapshotArray:
    def __init__(self, length):
        self.map = defaultdict(list)
        self.snapId = 0

    def set(self, index: int, val: int) -> None:
        if self.map[index] and self.map[index][-1][0] == self.snapId:
            self.map[index][-1][1] = val
            return
        self.map[index].append([self.snapId, val])

    def snap(self) -> int:
        self.snapId += 1
        return self.snapId - 1

    def get(self, index: int, snap_id: int) -> int:
        arr = self.map[index]
        left, right, ans = 0, len(arr) - 1, -1
        while left <= right:
            mid = (left + right) // 2
            if arr[mid][0] <= snap_id:
                ans = mid
                left = mid + 1
            else:
                right = mid - 1
        if ans == -1: return 0
        return arr[ans][1]
    

"""
1277. Count Square Submatrices with All Ones

Given a m * n matrix of ones and zeros, return how many square submatrices have all ones.

Example 1:

Input: matrix =
[
  [0,1,1,1],
  [1,1,1,1],
  [0,1,1,1]
]
Output: 15
Explanation: 
There are 10 squares of side 1.
There are 4 squares of side 2.
There is  1 square of side 3.
Total number of squares = 10 + 4 + 1 = 15.
"""
def countSquares(matrix):
    r= len(matrix) #Height
    c = len(matrix[0]) # Width
    result = 0
    for i in range(r):
        for j in range(c):
            if matrix[i][j]:
                if i > 0 and j > 0:
                    matrix[i][j] += min(matrix[i][j-1],matrix[i-1][j-1],matrix[i-1][j])
                result += matrix[i][j]
    
    return result

"""
1293. Shortest Path in a Grid with Obstacles Elimination

You are given an m x n integer matrix grid where each cell is either 0 (empty) or 1 (obstacle). You can move up, down, left, or right from and to an empty cell in one step.

Return the minimum number of steps to walk from the upper left corner (0, 0) to the lower right corner (m - 1, n - 1) given that you can eliminate at most k obstacles. If it is not possible to find such walk return -1.

Example 1:

Input: 
grid = 
[[0,0,0],
 [1,1,0],
 [0,0,0],
 [0,1,1],
 [0,0,0]], 
k = 1
Output: 6
Explanation: 
The shortest path without eliminating any obstacle is 10. 
The shortest path with one obstacle elimination at position (3,2) is 6. Such path is (0,0) -> (0,1) -> (0,2) -> (1,2) -> (2,2) -> (3,2) -> (4,2).
"""
from queue import Queue


def shortestPath(grid, k):
    m, n = len(grid), len(grid[0])
    if m == 1 and n == 1:
        return 0
    di = [-1, 0, 0, 1]
    dj = [0, -1, 1, 0]        
    step = 0
    q = Queue()
    q.put((0, 0, k))
    memo = set()
    
    while len(q.queue) > 0:
        size = len(q.queue)
        for _ in range(size):
            i, j, k = q.get()
            for t in range(4):
                ni = i + di[t]
                nj = j + dj[t]
                if ni in range(m) and nj in range(n):
                    if ni == m-1 and nj == n-1:
                        return step + 1
                    if grid[ni][nj] == 1 and k > 0 and (ni, nj, k-1) not in memo:
                        q.put((ni, nj, k-1))
                        memo.add((ni, nj, k-1))
                    elif grid[ni][nj] == 0 and (ni, nj, k) not in memo:
                        q.put((ni, nj, k))
                        memo.add((ni, nj, k))
        step += 1
    return -1
"""
1376. Time Needed to Inform All Employees

A company has n employees with a unique ID for each employee from 0 to n - 1. The head of the company is the one with headID.

Each employee has one direct manager given in the manager array where manager[i] is the direct manager of the i-th employee, manager[headID] = -1. Also, it is guaranteed that the subordination relationships have a tree structure.

The head of the company wants to inform all the company employees of an urgent piece of news. He will inform his direct subordinates, and they will inform their subordinates, and so on until all employees know about the urgent news.

The i-th employee needs informTime[i] minutes to inform all of his direct subordinates (i.e., After informTime[i] minutes, all his direct subordinates can start spreading the news).

Return the number of minutes needed to inform all the employees about the urgent news.

Example 1:

Input: n = 1, headID = 0, manager = [-1], informTime = [0]
Output: 0
Explanation: The head of the company is the only employee in the company.
"""
def numOfMinutes(n, headID, manager, informTime):
        """
        This is a very easy question, we need dict/graph mapping managers to employees and then
        we can do dfs to find time for each leaf node to be informed. The max of such leaf node
        times is the answer as that will be the maximum time taken to inform all employees
        """
        rel_map = collections.defaultdict(list)
        ans = 0
        
        # initalizing rel_map for building graph
        for emp,manager in enumerate(manager):
            rel_map[manager].append(emp)
            
        # now calculate time for each leaf node and find max
        def dfs(eid):
            # no further subordinate
            if not rel_map.get(eid):
                return 0
            
            time = informTime[eid] 
            time += max([ dfs(cid) for cid in rel_map.get(eid)])
            
            return time
        
        return dfs(headID)
    
"""
1406. Stone Game III

Alice and Bob continue their games with piles of stones. There are several stones arranged in a row, and each stone has an associated value which is an integer given in the array stoneValue.
Alice and Bob take turns, with Alice starting first. On each player's turn, that player can take 1, 2 or 3 stones from the first remaining stones in the row.
The score of each player is the sum of values of the stones taken. The score of each player is 0 initially.
The objective of the game is to end with the highest score, and the winner is the player with the highest score and there could be a tie. The game continues until all the stones have been taken.
Assume Alice and Bob play optimally.
Return "Alice" if Alice will win, "Bob" if Bob will win or "Tie" if they end the game with the same score.

Example 1:

Input: values = [1,2,3,7]
Output: "Bob"
Explanation: Alice will always lose. Her best move will be to take three piles and the score become 6. Now the score of Bob is 7 and Bob wins.
"""
def stoneGameIII(stoneValue):
    length=len(stoneValue)
    mem=[None]*length
    
    def Helper(i):
        if i>=length:
            return 0
        if mem[i] is not None:
            return mem[i]
        sums=stoneValue[i]
        maximum=sums-Helper(i+1)
        if i+1<length:
            sums+=stoneValue[i+1]
            maximum=max(maximum,sums-Helper(i+2))
        if i+2<length:
            sums+=stoneValue[i+2]
            maximum=max(maximum,sums-Helper(i+3))
        mem[i]=maximum
        return maximum
    
    result=Helper(0)
    if result>0:
        return "Alice"
    elif result<0:
        return "Bob"
    else:
        return "Tie"
"""
1423. Maximum Points You Can Obtain from Cards

There are several cards arranged in a row, and each card has an associated number of points. The points are given in the integer array cardPoints.

In one step, you can take one card from the beginning or from the end of the row. You have to take exactly k cards.

Your score is the sum of the points of the cards you have taken.

Given the integer array cardPoints and the integer k, return the maximum score you can obtain.

Example 1:

Input: cardPoints = [1,2,3,4,5,6,1], k = 3
Output: 12
Explanation: After the first step, your score will always be 1. However, choosing the rightmost card first will maximize your total score. The optimal strategy is to take the three cards on the right, giving a final score of 1 + 6 + 5 = 12.


Compute sum of k points from one end. Loop over k and each time subtract one element from end and add one element from front. Keep track of max.
"""
def maxScore(points, k):
    n = len(points)
    if k >= n:
        return sum(points)

    max_seen = sum(points[n-k:])
    prev_sum = max_seen
    
    for i in range(1, k+1):
        # max_seen = max(sum(points[:i]) + sum(points[n-k+i:]), max_seen) # O(k)
        new_sum = prev_sum - points[n-k+i-1] + points[i-1]  # O(1)
        max_seen = max(max_seen, new_sum)
        prev_sum = new_sum
    return max_seen
"""
1499. Max Value of Equation

You are given an array points containing the coordinates of points on a 2D plane, sorted by the x-values, where points[i] = [xi, yi] such that xi < xj for all 1 <= i < j <= points.length. You are also given an integer k.

Return the maximum value of the equation yi + yj + |xi - xj| where |xi - xj| <= k and 1 <= i < j <= points.length.

It is guaranteed that there exists at least one pair of points that satisfy the constraint |xi - xj| <= k.

Example 1:

Input: points = [[1,3],[2,0],[5,10],[6,-10]], k = 1
Output: 4
Explanation: The first two points satisfy the condition |xi - xj| <= 1 and if we calculate the equation we get 3 + 0 + |1 - 2| = 4. Third and fourth points also satisfy the condition and give a value of 10 + -10 + |5 - 6| = 1.
No other pairs satisfy the condition, so we return the max of 4 and 1.
"""
def findMaxValueOfEquation(points, k):
    
    left, right = 0, 1
    max_value = float('-inf')
    
    while right < len(points):
    	xl, yl = points[left]
    	xr, yr = points[right]
    
    	diff = abs(xr - xl)
    	if left == right:
    		right += 1
    	elif diff <= k:
    		m = yl + yr + diff
    		max_value = max(max_value, m)
    		if yl >= yr - diff:
    			right += 1
    		else:
    			left += 1
    	else:
    		left += 1
    return max_value

"""
1509. Minimum Difference Between Largest and Smallest Value in Three Moves

Given an array nums, you are allowed to choose one element of nums and change it by any value in one move.

Return the minimum difference between the largest and smallest value of nums after perfoming at most 3 moves.

Example 1:

Input: nums = [5,3,2,4]
Output: 0
Explanation: Change the array [5,3,2,4] to [2,2,2,2].
The difference between the maximum and minimum is 2-2 = 0.
"""
def minDifference(nums):
    n=len(nums)
    if n<=4:
        return 0
    nums.sort()
    minimum=float("inf")
    for i in range(4):
        minimum=min(minimum,nums[n-4+i]-nums[i])
    return minimum 

"""
1525. Number of Good Ways to Split a String

You are given a string s, a split is called good if you can split s into 2 non-empty strings p and q where its concatenation is equal to s and the number of distinct letters in p and q are the same.

Return the number of good splits you can make in s.

Example 1:

Input: s = "aacaba"
Output: 2
Explanation: There are 5 ways to split "aacaba" and 2 of them are good. 
("a", "acaba") Left string and right string contains 1 and 3 different letters respectively.
("aa", "caba") Left string and right string contains 1 and 3 different letters respectively.
("aac", "aba") Left string and right string contains 2 and 2 different letters respectively (good split).
("aaca", "ba") Left string and right string contains 2 and 2 different letters respectively (good split).
("aacab", "a") Left string and right string contains 3 and 1 different letters respectively.
"""
# TimeComplexity O(n)
# MemoryComplexity O(n)

def numSplits(s):
    if not s: return 0
    total = 0
    # prefix unique count
    prefix_count = [0]*len(s)
    unique = set()
    for i in range(len(s)):
        unique.add(s[i])
        prefix_count[i] = len(unique)
    # checking suffix
    unique.clear()
    for j in range(len(s) - 1, 0, -1):
        unique.add(s[j])
        if prefix_count[j-1] == len(unique):
            total += 1
    return total  
"""
1526. Minimum Number of Increments on Subarrays to Form a Target Array

Given an array of positive integers target and an array initial of same size with all zeros.

Return the minimum number of operations to form a target array from initial if you are allowed to do the following operation:

Choose any subarray from initial and increment each value by one.
The answer is guaranteed to fit within the range of a 32-bit signed integer.

Example 1:

Input: target = [1,2,3,2,1]
Output: 3
Explanation: We need at least 3 operations to form the target array from the initial array.
[0,0,0,0,0] increment 1 from index 0 to 4 (inclusive).
[1,1,1,1,1] increment 1 from index 1 to 3 (inclusive).
[1,2,2,2,1] increment 1 at index 2.
[1,2,3,2,1] target array is formed.
"""
def minNumberOperations(target):
    pre = count = 0
    for num in target + [0]:
        print(num,pre)
        if num < pre: # goes down
            count += pre - num
        pre = num
    return count

"""
1548. The Most Similar Path in a Graph

We have n cities and m bi-directional roads where roads[i] = [ai, bi] connects city ai with city bi. Each city has a name consisting of exactly 3 upper-case English letters given in the string array names. Starting at any city x, you can reach any city y where y != x (i.e. the cities and the roads are forming an undirected connected graph).

You will be given a string array targetPath. You should find a path in the graph of the same length and with the minimum edit distance to targetPath.

You need to return the order of the nodes in the path with the minimum edit distance, The path should be of the same length of targetPath and should be valid (i.e. there should be a direct road between ans[i] and ans[i + 1]). If there are multiple answers return any one of them.

The edit distance is defined as follows:


Follow-up: If each node can be visited only once in the path, What should you change in your solution?
 
Example 1:

Input: n = 5, roads = [[0,2],[0,3],[1,2],[1,3],[1,4],[2,4]], names = ["ATL","PEK","LAX","DXB","HND"], targetPath = ["ATL","DXB","HND","LAX"]
Output: [0,2,4,2]
Explanation: [0,2,4,2], [0,3,0,2] and [0,3,1,2] are accepted answers.
[0,2,4,2] is equivalent to ["ATL","LAX","HND","LAX"] which has edit distance = 1 with targetPath.
[0,3,0,2] is equivalent to ["ATL","DXB","ATL","LAX"] which has edit distance = 1 with targetPath.
[0,3,1,2] is equivalent to ["ATL","DXB","PEK","LAX"] which has edit distance = 1 with targetPath.
 Create a graph do BFS,  find the destinations that ends target destination.
 Use priority queue to get the one with minimum edit first
"""

def mostSimilar(n,roads,names,targetPath):
    if len(roads) == 0 or len(targetPath) == 0 : return []
            
    graph= defaultdict(list)
    for s, d in roads:
        graph[s].append(d)
        graph[d].append(s)
    name2id = {name:id  for id,name in enumerate(names)}
    if len(targetPath) == 1: return [name2id[targetPath[0]]]
    target_ids = [name2id[i] for i in targetPath]
    print(f'target {target_ids} ')
    ans= []
    level = 0
    queue = []
    # Entries in priorityQueue has following format
    #(edits, -stops, src) to make sure the ones with minimum number of edits and  
    # with longest length is processed first
    queue.append((0,-1,[target_ids[0]], target_ids[0]))
    
    while queue:
        #print (queue)
        edits, stops, path, cur = queue.pop(0)
        if -stops == len(target_ids) and path[-1] == target_ids[-1] :
            return "".join([str(p) for p in path] )
        elif  -stops < len(target_ids):
            for city in graph[cur]:
                new_edits = edits + (1 if target_ids[-stops] != city else 0)    
                heapq.heappush(queue, (new_edits, stops-1,path + [city], city))
    return ""
"""
1706. Where Will the Ball Fall

You have a 2-D grid of size m x n representing a box, and you have n balls. The box is open on the top and bottom sides.

Each cell in the box has a diagonal board spanning two corners of the cell that can redirect a ball to the right or to the left.

A board that redirects the ball to the right spans the top-left corner to the bottom-right corner and is represented in the grid as 1.
A board that redirects the ball to the left spans the top-right corner to the bottom-left corner and is represented in the grid as -1.
We drop one ball at the top of each column of the box. Each ball can get stuck in the box or fall out of the bottom. A ball gets stuck if it hits a "V" shaped pattern between two boards or if a board redirects the ball into either wall of the box.

Return an array answer of size n where answer[i] is the column that the ball falls out of at the bottom after dropping the ball from the ith column at the top, or -1 if the ball gets stuck in the box.

Example 1:
Input: grid = [[1,1,1,-1,-1],[1,1,1,-1,-1],[-1,-1,-1,1,1],[1,1,1,1,-1],[-1,-1,-1,-1,-1]]
Output: [1,-1,-1,-1,-1]
Explanation: This example is shown in the photo.
Ball b0 is dropped at column 0 and falls out of the box at column 1.
Ball b1 is dropped at column 1 and will get stuck in the box between column 2 and 3 and row 1.
Ball b2 is dropped at column 2 and will get stuck on the box between column 2 and 3 and row 0.
Ball b3 is dropped at column 3 and will get stuck on the box between column 2 and 3 and row 0.
Ball b4 is dropped at column 4 and will get stuck on the box between column 2 and 3 and row 1.
"""
def findBall(grid):
    def DFS(i,j,r,c):
        if i==r:#We reached the bottom row
            return j
        curr=grid[i][j]
        if curr==1 and (j+1==c or grid[i][j+1]==-1):#If adjacent number is diff than curr or doesn't exist
            return -1
        elif curr==-1 and (j-1<0 or grid[i][j-1]==1):#If adjacent number is diff than curr or doesn't exist
            return -1
        if curr==1:
            return DFS(i+1,j+1,r,c)
        else:
            return DFS(i+1,j-1,r,c)
    row,col=len(grid),len(grid[0])
    arr=[-1]*col
    for c in range(col):
        ans=DFS(0,c,row,col)
        arr[c]=ans
    return arr

"""
1776. Car Fleet II

There are n cars traveling at different speeds in the same direction along a one-lane road. You are given an array cars of length n, where cars[i] = [positioni, speedi] represents:

positioni is the distance between the ith car and the beginning of the road in meters. It is guaranteed that positioni < positioni+1.
speedi is the initial speed of the ith car in meters per second.
For simplicity, cars can be considered as points moving along the number line. Two cars collide when they occupy the same position. Once a car collides with another car, they unite and form a single car fleet. The cars in the formed fleet will have the same position and the same speed, which is the initial speed of the slowest car in the fleet.

Return an array answer, where answer[i] is the time, in seconds, at which the ith car collides with the next car, or -1 if the car does not collide with the next car. Answers within 10-5 of the actual answers are accepted.

Example 1:

Input: cars = [[1,2],[2,1],[4,3],[7,2]]
Output: [1.00000,-1.00000,3.00000,-1.00000]
Explanation: After exactly one second, the first car will collide with the second car, and form a car fleet with speed 1 m/s. After exactly 3 seconds, the third car will collide with the fourth car, and form a car fleet with speed 2 m/s.
"""
def getCollisionTimes(cars):
    """
    Your car can collid with a car ahead of your car
        Only and Only if your car speed is greater
        
        Time to collide = diff_of_position/diff_of_speed
        
        But If the Slower car ahead of your car, collide with other car 
        ahead of it in lesser time than yours collison time 
        than your car cant collide.    
    """
    #stack holds cars ahead of your car.
    stack = []  
    res = [-1]*len(cars)  
    for i in range(len(cars)-1, -1, -1):      
        ccP, ccS = cars[i]      
        while stack:          
            top = cars[stack[-1]]
            if top[1] >= ccS or (top[0] - ccP) / (ccS-top[1]) >= res[stack[-1]] > 0:
                stack.pop()
            else:
                break
        if stack:
            top = cars[stack[-1]]
            collisonT = (top[0] - ccP) / (ccS-top[1])
            res[i] = collisonT      
        stack.append(i)  
    return res

"""
1834. Single-Threaded CPU

You are given n​​​​​​ tasks labeled from 0 to n - 1 represented by a 2D integer array tasks, where tasks[i] = [enqueueTimei, processingTimei] means that the i​​​​​​th​​​​ task will be available to process at enqueueTimei and will take processingTimei to finish processing.

You have a single-threaded CPU that can process at most one task at a time and will act in the following way:

If the CPU is idle and there are no available tasks to process, the CPU remains idle.
If the CPU is idle and there are available tasks, the CPU will choose the one with the shortest processing time. If multiple tasks have the same shortest processing time, it will choose the task with the smallest index.
Once a task is started, the CPU will process the entire task without stopping.
The CPU can finish a task then start a new one instantly.
Return the order in which the CPU will process the tasks.

Example 1:

Input: tasks = [[1,2],[2,4],[3,2],[4,1]]
Output: [0,2,3,1]
Explanation: The events go as follows: 
- At time = 1, task 0 is available to process. Available tasks = {0}.
- Also at time = 1, the idle CPU starts processing task 0. Available tasks = {}.
- At time = 2, task 1 is available to process. Available tasks = {1}.
- At time = 3, task 2 is available to process. Available tasks = {1, 2}.
- Also at time = 3, the CPU finishes task 0 and starts processing task 2 as it is the shortest. Available tasks = {1}.
- At time = 4, task 3 is available to process. Available tasks = {1, 3}.
- At time = 5, the CPU finishes task 2 and starts processing task 3 as it is the shortest. Available tasks = {1}.
- At time = 6, the CPU finishes task 3 and starts processing task 1. Available tasks = {}.
- At time = 10, the CPU finishes task 1 and becomes idle.
"""
import heapq
def getOrder(tasks):
    sorted_tasks = sorted([(enqueue_time, process_time, index) for index, [enqueue_time, process_time] in enumerate(tasks)])
    res, heap = [], []
    print(sorted_tasks)
    enqueued, current_time = 0, sorted_tasks[0][0]
    while len(res) < len(sorted_tasks):
        while enqueued < len(sorted_tasks):
            enqueue_time, process_time, index = sorted_tasks[enqueued]
            print(enqueued,enqueue_time,index,heap,res)
            if enqueue_time > current_time and heap:
                break
            else:
                enqueued += 1
                heapq.heappush(heap, (process_time, index, enqueue_time))
        process_time, index, enqueue_time = heapq.heappop(heap)
        current_time = max(current_time, enqueue_time) + process_time
        res.append(index)
    return res
"""
1937. Maximum Number of Points with Cost

You are given an m x n integer matrix points (0-indexed). Starting with 0 points, you want to maximize the number of points you can get from the matrix.

To gain points, you must pick one cell in each row. Picking the cell at coordinates (r, c) will add points[r][c] to your score.

However, you will lose points if you pick a cell too far from the cell that you picked in the previous row. For every two adjacent rows r and r + 1 (where 0 <= r < m - 1), picking cells at coordinates (r, c1) and (r + 1, c2) will subtract abs(c1 - c2) from your score.

Return the maximum number of points you can achieve.

abs(x) is defined as:

x for x >= 0.
-x for x < 0.
 
Example 1:


Input: points = [[1,2,3],[1,5,1],[3,1,1]]
Output: 9
Explanation:
The blue cells denote the optimal cells to pick, which have coordinates (0, 2), (1, 1), and (2, 0).
You add 3 + 5 + 3 = 11 to your score.
However, you must subtract abs(2 - 1) + abs(1 - 0) = 2 from your score.
Your final score is 11 - 2 = 9.

"""
def maxPoints(points):
    m = len(points)
    n = len(points[0])
    v_max = [0] * n
    
    for r in range(m):
        p = points[r]
        for c in range(n):
            v_max[c] += p[c]
        
        # left to right
        for cl in range(n - 1):
            cr = cl + 1
            v_max[cr] = max(v_max[cl] - 1, v_max[cr])

        # right to left
        for cr in range(n - 1):
            cr = n - 1 - cr
            cl = cr - 1
            v_max[cl] = max(v_max[cr] - 1, v_max[cl])
    
    return max(v_max)

"""
1466. Reorder Routes to Make All Paths Lead to the City Zero

There are n cities numbered from 0 to n - 1 and n - 1 roads such that there is only one way to travel between two different cities (this network form a tree). Last year, The ministry of transport decided to orient the roads in one direction because they are too narrow.

Roads are represented by connections where connections[i] = [ai, bi] represents a road from city ai to city bi.

This year, there will be a big event in the capital (city 0), and many people want to travel to this city.

Your task consists of reorienting some roads such that each city can visit the city 0. Return the minimum number of edges changed.

It's guaranteed that each city can reach city 0 after reorder.

Example 1:

Input: n = 6, connections = [[0,1],[1,3],[2,3],[4,0],[4,5]]
Output: 3
Explanation: Change the direction of edges show in red such that each node can reach the node 0 (capital).
"""
def minReorder(n, connections):
    needChange = 0
    fromcity = collections.defaultdict(set)
    neigh = collections.defaultdict(set)
    for fr,to in connections:
        fromcity[fr].add(to)
        neigh[fr].add(to)
        neigh[to].add(fr)
    seen = set()
    def dfs(city):
        nonlocal needChange
        for n in neigh[city]:
            if n not in seen:
                seen.add(city)
                if city not in fromcity[n]:
                    needChange += 1
                dfs(n)
    dfs(0)
    return needChange




"""
Given two sorted arrays nums1 and nums2 of size m and n respectively, return the median of the two sorted arrays.

The overall run time complexity should be O(log (m+n)).

 

Example 1:

Input: nums1 = [1,3], nums2 = [2]
Output: 2.00000
Explanation: merged array = [1,2,3] and median is 2.
Example 2:

Input: nums1 = [1,2], nums2 = [3,4]
Output: 2.50000
Explanation: merged array = [1,2,3,4] and median is (2 + 3) / 2 = 2.5.

First one is O{log(m+n)}
Second is O((m+n)Log(m+n)).
"""

def findMedianSortedArrays(nums1, nums2):
    n=(len(nums1)+len(nums2))
    
    if(n%2!=0):
        return float(self.helper(n//2, nums1, nums2))
    else:
        return ((self.helper(n//2, nums1, nums2))+(self.helper(n//2-1, nums1, nums2)))/2
   
def helper(self, k, nums1, nums2):
    left=min(nums1[0], nums2[0])
    
    right=max(nums1[-1], nums2[-1])
    
    ans=0
    
    while(left<=right):
        mid=left+(right-left)//2
        
        count=self.lesserCount(nums1, nums2, mid)
        
        if(count<=k):
            ans=mid
            left=mid+1
        else:
            right=mid-1
    return ans  

def lesserCount(self, nums1, nums2, target):
    return self.binarySearch(nums1, target)+self.binarySearch(nums2, target)
    
def binarySearch(self, arr, target):
    left=0
    right=len(arr)-1
    ans=0
    while(left<=right):
        mid=left+(right-left)//2
        
        if(arr[mid]<target):
            ans=mid+1
            left=mid+1
        else:
            right=mid-1
            
    return ans

"""
def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
    merged_list = sorted(nums1 + nums2)
    list_len = len(merged_list)
    half = int(list_len / 2)
    if list_len % 2 == 0:
        return (merged_list[half - 1] + merged_list[half]) / 2
    else:
        return merged_list[half]
"""
    
"""
9. Palindrome Number
Given an integer x, return true if x is a 
palindrome
, and false otherwise.

 

Example 1:

Input: x = 121
Output: true
Explanation: 121 reads as 121 from left to right and from right to left.
"""
def isPalindrome(x):
    initial = x
    if x < 0:
        return False
    ans = 0
    while x:
        ans = x%10 + ans*10
        x //= 10
    
    return ans == initial

def isPalindrome(x):
    x = str(x)
    if x == x[::-1]:
        return True
    return False

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
For example, 2 is written as II in Roman numeral, just two ones added together. 12 is written as XII, which is simply X + II. The number 27 is written as XXVII, which is XX + V + II.

Roman numerals are usually written largest to smallest from left to right. However, the numeral for four is not IIII. Instead, the number four is written as IV. Because the one is before the five we subtract it making four. The same principle applies to the number nine, which is written as IX. There are six instances where subtraction is used:

I can be placed before V (5) and X (10) to make 4 and 9. 
X can be placed before L (50) and C (100) to make 40 and 90. 
C can be placed before D (500) and M (1000) to make 400 and 900.
Given a roman numeral, convert it to an integer.

 

Example 1:

Input: s = "III"
Output: 3
Explanation: III = 3.
"""
def romanToInt(s):
    values = {"I":1,"V":5,"X":10,"L":50,"C":100,"D":500,"M":1000}
    total = 0
    i = 0
    while i < len(s):
        if i < len(s)-1 and values[s[i]] < values[s[i+1]]:
            total += values[s[i+1]] - values[s[i]]
            i += 2
        else:
            total += values[s[i]]
            i += 1
    return total
"""
14. Longest Common Prefix

Write a function to find the longest common prefix string amongst an array of strings.

If there is no common prefix, return an empty string "".

 

Example 1:

Input: strs = ["flower","flow","flight"]
Output: "fl"
"""
def longestCommonPrefix(string):
    res = ""
    for z in zip(*string):
        if len(set(z)) == 1:
            res += z[0]
        else:
            break
    return res

"""
20. Valid Parentheses

Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

An input string is valid if:

Open brackets must be closed by the same type of brackets.
Open brackets must be closed in the correct order.
 

Example 1:

Input: s = "()"
Output: true
"""

def isValid(s):
    bracket_map = {"(": ")", "[": "]",  "{": "}"}
    open_par = set(["(", "[", "{"])
    stack = []
    for i in s:
        if i in open_par:
            stack.append(i)
        elif stack and i == bracket_map[stack[-1]]:
                stack.pop()
        else:
            return False
    return stack == []
"""
42. Trapping Rain Water

Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it can trap after raining.


Example 1:

Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6
Explanation: The above elevation map (black section) is represented by array [0,1,0,2,1,0,1,3,2,1,2,1]. In this case, 6 units of rain water (blue section) are being trapped.

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
56. Merge Intervals

Given an array of intervals where intervals[i] = [starti, endi], merge all overlapping intervals, and return an array of the non-overlapping intervals that cover all the intervals in the input.

 

Example 1:

Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
Output: [[1,6],[8,10],[15,18]]
Explanation: Since intervals [1,3] and [2,6] overlap, merge them into [1,6].
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
68. Text Justification

Given an array of strings words and a width maxWidth, format the text such that each line has exactly maxWidth characters and is fully (left and right) justified.

You should pack your words in a greedy approach; that is, pack as many words as you can in each line. Pad extra spaces ' ' when necessary so that each line has exactly maxWidth characters.

Extra spaces between words should be distributed as evenly as possible. If the number of spaces on a line does not divide evenly between words, the empty slots on the left will be assigned more spaces than the slots on the right.

For the last line of text, it should be left-justified, and no extra space is inserted between words.

Note:

A word is defined as a character sequence consisting of non-space characters only.
Each word's length is guaranteed to be greater than 0 and not exceed maxWidth.
The input array words contains at least one word.
 

Example 1:

Input: words = ["This", "is", "an", "example", "of", "text", "justification."], maxWidth = 16
Output:
[
   "This    is    an",
   "example  of text",
   "justification.  "
]
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
                print("base", base,res)
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
"""
79. Word Search

Given an m x n grid of characters board and a string word, return true if word exists in the grid.

The word can be constructed from letters of sequentially adjacent cells, where adjacent cells are horizontally or vertically neighboring. The same letter cell may not be used more than once.

 

Example 1:


Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
Output: true
"""
def exist(board, word):
    
    directions = [[0,1],[1,0],[0,-1],[-1,0]]
    m,n = len(board),len(board[0])
    isfound=[0]
    
    def dfs(i, j, s, visited, t):
        # print(s, visited)
        if s==word:
            isfound[0]=1
            return True
        else:
            flag=False
            for x in directions:
                if(i+x[0]>=0 and i+x[0]<m and j+x[1]<n and j+x[1]>=0):
                    if(board[i+x[0]][j+x[1]]==word[t] and (i+x[0], j+x[1]) not in visited):
                        tvisited=visited.copy()
                        tvisited[(i+x[0], j+x[1])]=1
                        flag=True
                        dfs(i+x[0], j+x[1],s+word[t], tvisited, t+1)
            if(not flag):
                return False

    for i in range(m):
        for j in range(n):
            if(board[i][j]==word[0]):
                td={}
                td[(i, j)]=1
                dfs(i, j, word[0], td , 1)
    
    if(isfound[0]==1):
        return True
    return False
"""
128. Longest Consecutive Sequence

Given an unsorted array of integers nums, return the length of the longest consecutive elements sequence.

You must write an algorithm that runs in O(n) time.

 

Example 1:

Input: nums = [100,4,200,1,3,2]
Output: 4
Explanation: The longest consecutive elements sequence is [1, 2, 3, 4]. Therefore its length is 4.
"""

def longestConsecutive(nums):
    numSets=set(nums)
    longest=0
    for n in numSets:
        if (n-1) not in numSets:
            length=0
            while (n+length) in numSets:
                length+=1
            longest=max(length,longest)
    return longest

"""
146. LRU Cache

Design a data structure that follows the constraints of a Least Recently Used (LRU) cache.

Implement the LRUCache class:

LRUCache(int capacity) Initialize the LRU cache with positive size capacity.
int get(int key) Return the value of the key if the key exists, otherwise return -1.
void put(int key, int value) Update the value of the key if the key exists. Otherwise, add the key-value pair to the cache. If the number of keys exceeds the capacity from this operation, evict the least recently used key.
Follow up:
Could you do get and put in O(1) time complexity?

 

Example 1:

Input
["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
[[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]
Output
[null, null, null, 1, null, -1, null, -1, 3, 4]

Explanation
LRUCache lRUCache = new LRUCache(2);
lRUCache.put(1, 1); // cache is {1=1}
lRUCache.put(2, 2); // cache is {1=1, 2=2}
lRUCache.get(1);    // return 1
lRUCache.put(3, 3); // LRU key was 2, evicts key 2, cache is {1=1, 3=3}
lRUCache.get(2);    // returns -1 (not found)
lRUCache.put(4, 4); // LRU key was 1, evicts key 1, cache is {4=4, 3=3}
lRUCache.get(1);    // return -1 (not found)
lRUCache.get(3);    // return 3
lRUCache.get(4);    // return 4
"""

class LRUCache:

    def __init__(self, capacity: int):
        self.size = capacity
        self.cache = collections.OrderedDict()

    def get(self, key: int) -> int:
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        else:
            return -1        

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)  
        elif len(self.cache) == self.size:
            self.cache.popitem(last=False)
        self.cache[key] = value
        
"""
334. Increasing Triplet Subsequence

Given an integer array nums, return true if there exists a triple of indices (i, j, k) such that i < j < k and nums[i] < nums[j] < nums[k]. If no such indices exists, return false.

Example 1:

Input: nums = [1,2,3,4,5]
Output: true
Explanation: Any triplet where i < j < k is valid.
"""        
        
def increasingTriplet(nums):
    prev=nums[0]
    minel=float('inf')
    
    for i in nums[1:]:
        # print(minel,'hehe',prev,i)

        if i>prev:
            
            if i>minel:
                return True
            # cnt+=1
            minel=min(i,minel)
        prev=min(prev,i)
    return False

"""
366. Find Leaves of Binary Tree

Given the root of a binary tree, collect a tree's nodes as if you were doing this:

Collect all the leaf nodes.
Remove all the leaf nodes.
Repeat until the tree is empty.

Example 1:

Input: root = [1,2,3,4,5]
Output: [[4,5,3],[2],[1]]
Explanation:
[[3,5,4],[2],[1]] and [[3,4,5],[2],[1]] are also considered correct 
answers since per each level it does not matter the order on which elements are returned.

Here's a simple python DFS implementation used to populate a dictionary (key = index, values = list of nodes), 
O(N) space, O(N) time, beats 99%

The DFS recursively calculates the layer index by getting the maximum depth from the left and right subtrees of a given node, 
which is then used to populate the dictionary.

"""
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

def findLeaves(root):
    output = collections.defaultdict(list)
    
    def dfs(node):
        if not node: 
            return 0 
        left = dfs(node.left)
        right = dfs(node.right)
        layer = max(left, right)
        output[layer].append(node.val)
        return layer + 1
    
    dfs(root)
    return output.values() 
"""
424. Longest Repeating Character Replacement

You are given a string s and an integer k. You can choose any character of the string and change it to any other uppercase English character. You can perform this operation at most k times.

Return the length of the longest substring containing the same letter you can get after performing the above operations.

 

Example 1:

Input: s = "ABAB", k = 2
Output: 4
Explanation: Replace the two 'A's with two 'B's or vice versa.
"""
def characterReplacement(s, k):        
    cnts = defaultdict(int)
    
    lo, hi, n = 0, 0, len(s)
    ans = 1
    while hi < n:
        cnts[s[hi]] += 1
        while (hi - lo + 1) - max(cnts.values()) > k:
            cnts[s[lo]] -= 1
            lo += 1
        
        hi += 1
        ans = max(ans, hi - lo)
    return ans

"""
818. Race Car


Your car starts at position 0 and speed +1 on an infinite number line. Your car can go into negative positions. Your car drives automatically according to a sequence of instructions 'A' (accelerate) and 'R' (reverse):

When you get an instruction 'A', your car does the following:
position += speed
speed *= 2
When you get an instruction 'R', your car does the following:
If your speed is positive then speed = -1
otherwise speed = 1
Your position stays the same.
For example, after commands "AAR", your car goes to positions 0 --> 1 --> 3 --> 3, and your speed goes to 1 --> 2 --> 4 --> -1.

Given a target position target, return the length of the shortest sequence of instructions to get there.


Example 1:

Input: target = 3
Output: 2
Explanation: 
The shortest instruction sequence is "AA".
Your position goes from 0 --> 1 --> 3.
"""
def racecar(target):
    q = [(0, 1)]
    steps = 0
    
    while q:
        num = len(q)
        for i in range(num):
            pos, speed = q.pop(0)
            if pos == target:
                return steps
            q.append((pos+speed, speed*2))
            rev_speed = -1 if speed > 0 else 1
            if (pos+speed) < target and speed < 0 or (pos+speed) > target and speed > 0:
                q.append((pos, rev_speed))
        steps += 1
"""
1101. The Earliest Moment When Everyone Become Friends

There are n people in a social group labeled from 0 to n - 1. You are given an array logs where logs[i] = [timestampi, xi, yi] indicates that xi and yi will be friends at the time timestampi.

Friendship is symmetric. That means if a is friends with b, then b is friends with a. Also, person a is acquainted with a person b if a is friends with b, or a is a friend of someone acquainted with b.

Return the earliest time for which every person became acquainted with every other person. If there is no such earliest time, return -1.

 

Example 1:

Input: logs = [[20190101,0,1],[20190104,3,4],[20190107,2,3],[20190211,1,5],[20190224,2,4],[20190301,0,3],[20190312,1,2],[20190322,4,5]], n = 6
Output: 20190301
Explanation: 
The first event occurs at timestamp = 20190101, and after 0 and 1 become friends, we have the following friendship groups [0,1], [2], [3], [4], [5].
The second event occurs at timestamp = 20190104, and after 3 and 4 become friends, we have the following friendship groups [0,1], [2], [3,4], [5].
The third event occurs at timestamp = 20190107, and after 2 and 3 become friends, we have the following friendship groups [0,1], [2,3,4], [5].
The fourth event occurs at timestamp = 20190211, and after 1 and 5 become friends, we have the following friendship groups [0,1,5], [2,3,4].
The fifth event occurs at timestamp = 20190224, and as 2 and 4 are already friends, nothing happens.
The sixth event occurs at timestamp = 20190301, and after 0 and 3 become friends, we all become friends.
"""        
class Solution:  
    def earliestAcq(self,A, n):
        A=sorted(A,key=lambda x:x[0])
        
        def recur(i):
            self.visited.add(i)
            for a in dic[i]:
                if a not in self.visited:
                    recur(a)
            return
        
        dic=collections.defaultdict(list)
        
        for a,b,c in A:
            self.visited=set()
            dic[b].append(c)
            dic[c].append(b)
            recur(b)
            #print(self.visited)
            if len(self.visited)==n:
                return a
        return -1
A = [[20190101,0,1],[20190104,3,4],[20190107,2,3],[20190211,1,5],[20190224,2,4],[20190301,0,3],[20190312,1,2],[20190322,4,5]]
n = 6 
acq = Solution().earliestAcq(A,n)

#acq = s.earliestAcq(A,n)
print("Earliest Acquaintance:", acq)

"""
1105. Filling Bookcase Shelves

You are given an array books where books[i] = [thicknessi, heighti] indicates the thickness and height of the ith book. You are also given an integer shelfWidth.

We want to place these books in order onto bookcase shelves that have a total width shelfWidth.

We choose some of the books to place on this shelf such that the sum of their thickness is less than or equal to shelfWidth, then build another level of the shelf of the bookcase so that the total height of the bookcase has increased by the maximum height of the books we just put down. We repeat this process until there are no more books to place.

Note that at each step of the above process, the order of the books we place is the same order as the given sequence of books.

For example, if we have an ordered list of 5 books, we might place the first and second book onto the first shelf, the third book on the second shelf, and the fourth and fifth book on the last shelf.
Return the minimum possible height that the total bookshelf can be after placing shelves in this manner.

Input: books = [[1,1],[2,3],[2,3],[1,1],[1,1],[1,1],[1,2]], shelf_width = 4
Output: 6
Explanation:
The sum of the heights of the 3 shelves is 1 + 3 + 2 = 6.
Notice that book number 2 does not have to be on the first shelf.
"""

def minHeightShelves(books, shelfWidth):
    n = len(books)
    dp = [0] * (n + 1)
    for i in range(1, n + 1):
        width, height = books[i - 1]
        dp[i] = dp[i - 1] + height
        j = i - 1
        while j > 0 and width + books[j - 1][0] <= shelfWidth:
            width += books[j - 1][0]
            height = max(height, books[j - 1][1])
            dp[i] = min(dp[i], dp[j - 1] + height)
            j -= 1
    return dp[n]


"""
1136. Parallel Courses

You are given an integer n, which indicates that there are n courses labeled from 1 to n. You are also given an array relations where relations[i] = [prevCoursei, nextCoursei], representing a prerequisite relationship between course prevCoursei and course nextCoursei: course prevCoursei has to be taken before course nextCoursei.

In one semester, you can take any number of courses as long as you have taken all the prerequisites in the previous semester for the courses you are taking.

Return the minimum number of semesters needed to take all courses. If there is no way to take all the courses, return -1.

 

Example 1:


Input: n = 3, relations = [[1,3],[2,3]]
Output: 2
Explanation: The figure above represents the given graph.
In the first semester, you can take courses 1 and 2.
In the second semester, you can take course 3.
"""



def minimumSemesters(n, relations):
    graph = {i:[] for i in range(1,n+1)}
    for start,end in relations:
        graph[start].append(end)
    
    visited = {}
    def dfs(node):
        if node in visited:
            return visited[node]
        else:
            visited[node] = -1
        maxLen = 1
        for end in graph[node]:
            length = dfs(end)
            if length == -1:
                return -1
            else:
                maxLen = max(length+1, maxLen)
        visited[node] = maxLen
        return maxLen
    maxLength = -1
    for node in graph.keys():
        length = dfs(node)
        if length == -1:
            return -1
        else:
            maxLength = max(length, maxLength)
    return maxLength
"""
2013. Detect Squares

You are given a stream of points on the X-Y plane. Design an algorithm that:

Adds new points from the stream into a data structure. Duplicate points are allowed and should be treated as different points.
Given a query point, counts the number of ways to choose three points from the data structure such that the three points and the query point form an axis-aligned square with positive area.
An axis-aligned square is a square whose edges are all the same length and are either parallel or perpendicular to the x-axis and y-axis.

Implement the DetectSquares class:

DetectSquares() Initializes the object with an empty data structure.
void add(int[] point) Adds a new point point = [x, y] to the data structure.
int count(int[] point) Counts the number of ways to form axis-aligned squares with point point = [x, y] as described above.
 

Example 1:


Input
["DetectSquares", "add", "add", "add", "count", "count", "add", "count"]
[[], [[3, 10]], [[11, 2]], [[3, 2]], [[11, 10]], [[14, 8]], [[11, 2]], [[11, 10]]]
Output
[null, null, null, null, 1, 0, null, 2]

Explanation
DetectSquares detectSquares = new DetectSquares();
detectSquares.add([3, 10]);
detectSquares.add([11, 2]);
detectSquares.add([3, 2]);
detectSquares.count([11, 10]); // return 1. You can choose:
                               //   - The first, second, and third points
detectSquares.count([14, 8]);  // return 0. The query point cannot form a square with any points in the data structure.
detectSquares.add([11, 2]);    // Adding duplicate points is allowed.
detectSquares.count([11, 10]); // return 2. You can choose:
                               //   - The first, second, and third points
                               //   - The first, third, and fourth points
"""

from collections import Counter
class DetectSquares:

    def __init__(self):
        self.counter = Counter()
        

    def add(self, point: List[int]) -> None:
        self.counter[tuple(point)] += 1
        

    def count(self, point: List[int]) -> int:
        res = 0
        x, y = point
        for x0, y0 in self.counter.keys():
            if abs(x-x0) == abs(y-y0) and abs(x-x0):
                res += self.counter[(x0,y0)]*self.counter[(x0,y)]*self.counter[(x,y0)]
        return res
    
"""
2034. Stock Price Fluctuation 

You are given a stream of records about a particular stock. Each record contains a timestamp and the corresponding price of the stock at that timestamp.

Unfortunately due to the volatile nature of the stock market, the records do not come in order. Even worse, some records may be incorrect. Another record with the same timestamp may appear later in the stream correcting the price of the previous wrong record.

Design an algorithm that:

Updates the price of the stock at a particular timestamp, correcting the price from any previous records at the timestamp.
Finds the latest price of the stock based on the current records. The latest price is the price at the latest timestamp recorded.
Finds the maximum price the stock has been based on the current records.
Finds the minimum price the stock has been based on the current records.
Implement the StockPrice class:

StockPrice() Initializes the object with no price records.
void update(int timestamp, int price) Updates the price of the stock at the given timestamp.
int current() Returns the latest price of the stock.
int maximum() Returns the maximum price of the stock.
int minimum() Returns the minimum price of the stock.
 

Example 1:

Input
["StockPrice", "update", "update", "current", "maximum", "update", "maximum", "update", "minimum"]
[[], [1, 10], [2, 5], [], [], [1, 3], [], [4, 2], []]
Output
[null, null, null, 5, 10, null, 5, null, 2]

Explanation
StockPrice stockPrice = new StockPrice();
stockPrice.update(1, 10); // Timestamps are [1] with corresponding prices [10].
stockPrice.update(2, 5);  // Timestamps are [1,2] with corresponding prices [10,5].
stockPrice.current();     // return 5, the latest timestamp is 2 with the price being 5.
stockPrice.maximum();     // return 10, the maximum price is 10 at timestamp 1.
stockPrice.update(1, 3);  // The previous timestamp 1 had the wrong price, so it is updated to 3.
                          // Timestamps are [1,2] with corresponding prices [3,5].
stockPrice.maximum();     // return 5, the maximum price is 5 after the correction.
stockPrice.update(4, 2);  // Timestamps are [1,2,4] with corresponding prices [3,5,2].
stockPrice.minimum();     // return 2, the minimum price is 2 at timestamp 4.
"""   
class StockPrice:

    def __init__(self):
        self.dic=defaultdict(int)
        self.maxts=-inf
        self.arr=[]
        
        

    def update(self, timestamp, price):
        if timestamp in self.dic:
            self.arr.remove(self.dic[timestamp])
        self.dic[timestamp]=price
        self.maxts=max(self.maxts,timestamp)
        #Binary Search
        lo,hi=0,len(self.arr)-1
        arr=self.arr
        while lo<=hi:
            mid=lo+(hi-lo)//2
            if arr[mid]>price:
                hi=mid-1
            else:
                lo=mid+1
        arr.insert(lo,price)
        
        

    def current(self):
        return self.dic[self.maxts]
        

    def maximum(self):
        return self.arr[-1]

    def minimum(self):
        return self.arr[0]
    

"""
2101. Detonate the Maximum Bombs

You are given a list of bombs. The range of a bomb is defined as the area where its effect can be felt. This area is in the shape of a circle with the center as the location of the bomb.

The bombs are represented by a 0-indexed 2D integer array bombs where bombs[i] = [xi, yi, ri]. xi and yi denote the X-coordinate and Y-coordinate of the location of the ith bomb, whereas ri denotes the radius of its range.

You may choose to detonate a single bomb. When a bomb is detonated, it will detonate all bombs that lie in its range. These bombs will further detonate the bombs that lie in their ranges.

Given the list of bombs, return the maximum number of bombs that can be detonated if you are allowed to detonate only one bomb.

 

Example 1:


Input: bombs = [[2,1,3],[6,1,4]]
Output: 2
Explanation:
The above figure shows the positions and ranges of the 2 bombs.
If we detonate the left bomb, the right bomb will not be affected.
But if we detonate the right bomb, both bombs will be detonated.
So the maximum bombs that can be detonated is max(1, 2) = 2.
"""
def maximumDetonation(bombs):

    # Build adjacent graph
    # Use dfs

    # 660 ms, 85 %

    n = len(bombs)
    # Build a graph for adjacent bombs. 
    graph = defaultdict(list)
    for i in range(n):
        for j in range(i+1, n):
            distance = sqrt((bombs[i][0] - bombs[j][0])**2 + (bombs[i][1] - bombs[j][1])**2)
            if distance <= bombs[i][2]:
                graph[i].append(j)
            if distance <= bombs[j][2]:
                graph[j].append(i)

    def dfs(node):
        for neigh in graph[node]:
            if neigh not in seen:
                seen.add(neigh)
                dfs(neigh)
                
    max_count = 0
    for i in range(n):
        seen = set()
        seen.add(i)
        dfs(i)
        # print(f'seen:{seen}')
        max_count = max(max_count, len(seen))

    return max_count

"""
2402. Meeting Rooms III

You are given an integer n. There are n rooms numbered from 0 to n - 1.

You are given a 2D integer array meetings where meetings[i] = [starti, endi] means that a meeting will be held during the half-closed time interval [starti, endi). All the values of starti are unique.

Meetings are allocated to rooms in the following manner:

Each meeting will take place in the unused room with the lowest number.
If there are no available rooms, the meeting will be delayed until a room becomes free. The delayed meeting should have the same duration as the original meeting.
When a room becomes unused, meetings that have an earlier original start time should be given the room.
Return the number of the room that held the most meetings. If there are multiple rooms, return the room with the lowest number.

A half-closed interval [a, b) is the interval between a and b including a and not including b.

 

Example 1:

Input: n = 2, meetings = [[0,10],[1,5],[2,7],[3,4]]
Output: 0
Explanation:
- At time 0, both rooms are not being used. The first meeting starts in room 0.
- At time 1, only room 1 is not being used. The second meeting starts in room 1.
- At time 2, both rooms are being used. The third meeting is delayed.
- At time 3, both rooms are being used. The fourth meeting is delayed.
- At time 5, the meeting in room 1 finishes. The third meeting starts in room 1 for the time period [5,10).
- At time 10, the meetings in both rooms finish. The fourth meeting starts in room 0 for the time period [10,11).
Both rooms 0 and 1 held 2 meetings, so we return 0. 


Complexity
Time O(nlogn)
Space O(n)
"""


def mostBooked(n, meetings):
    ready = [r for r in range(n)]
    rooms = []
    heapify(ready)
    res = [0] * n
    for s,e in sorted(meetings):
        while rooms and rooms[0][0] <= s:
            t,r = heappop(rooms)
            heappush(ready, r)
        if ready:
            r = heappop(ready)
            heappush(rooms, [e, r])
        else:
            t,r = heappop(rooms)
            heappush(rooms, [t + e - s, r])
        res[r] += 1
    return res.index(max(res))


"""
2416. Sum of Prefix Scores of Strings

You are given an array words of size n consisting of non-empty strings.

We define the score of a string word as the number of strings words[i] such that word is a prefix of words[i].

For example, if words = ["a", "ab", "abc", "cab"], then the score of "ab" is 2, since "ab" is a prefix of both "ab" and "abc".
Return an array answer of size n where answer[i] is the sum of scores of every non-empty prefix of words[i].

Note that a string is considered as a prefix of itself.

 

Example 1:

Input: words = ["abc","ab","bc","b"]
Output: [5,4,3,2]
Explanation: The answer for each string is the following:
- "abc" has 3 prefixes: "a", "ab", and "abc".
- There are 2 strings with the prefix "a", 2 strings with the prefix "ab", and 1 string with the prefix "abc".
The total is answer[0] = 2 + 2 + 1 = 5.
- "ab" has 2 prefixes: "a" and "ab".
- There are 2 strings with the prefix "a", and 2 strings with the prefix "ab".
The total is answer[1] = 2 + 2 = 4.
- "bc" has 2 prefixes: "b" and "bc".
- There are 2 strings with the prefix "b", and 1 string with the prefix "bc".
The total is answer[2] = 2 + 1 = 3.
- "b" has 1 prefix: "b".
- There are 2 strings with the prefix "b".
The total is answer[3] = 2.
"""