# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 08:43:42 2020
@author: prchandr
"""
from collections import deque,defaultdict
"""
994. Rotting Oranges Time and Space - O(N) Size of the grid

In a given grid, each cell can have one of three values:

the value 0 representing an empty cell;
the value 1 representing a fresh orange;
the value 2 representing a rotten orange.
Every minute, any fresh orange that is adjacent (4-directionally) to a rotten orange becomes rotten.

Return the minimum number of minutes that must elapse until no cell has a fresh orange.  If this is impossible, return -1 instead.

Example 1:

Input: [[2,1,1],[1,1,0],[0,1,1]]
Output: 4
"""
def rottenOrange(grid):
    queue = deque()
    freshOranges = 0
    row,col = len(grid),len(grid[0])
    
    for r in range(row):
        for c in range(col):
            if grid[r][c] == 2:
                queue.append((r,c))
            elif grid[r][c] == 1:
                freshOranges += 1
    queue.append((-1,-1))
    minutes_elapsed = -1
    directions = [(1,0),(0,1),(-1,0),(0,-1)]
    
    while queue:
        ro, co = queue.popleft()
        if ro == -1:
            minutes_elapsed += 1
            if queue:
                queue.append((-1,-1))
        else:
            for d in directions:
                neighbor_row, neighbor_col = ro+d[0], co+d[1]
                if 0 <= neighbor_row < row and 0 <= neighbor_col < col:
                    if grid[neighbor_row][neighbor_col] == 1:
                        grid[neighbor_row][neighbor_col] = 2
                        freshOranges -= 1
                        queue.append((neighbor_row,neighbor_col))
    return minutes_elapsed if freshOranges == 0 else -1

"""
127. Word Ladder

Given two words (beginWord and endWord), and a dictionary's word list, find the length of shortest transformation sequence from beginWord to endWord, such that:

Only one letter can be changed at a time.
Each transformed word must exist in the word list.
Note:

Return 0 if there is no such transformation sequence.
All words have the same length.
All words contain only lowercase alphabetic characters.
You may assume no duplicates in the word list.
You may assume beginWord and endWord are non-empty and are not the same.
Example 1:

Input:
beginWord = "hit",
endWord = "cog",
wordList = ["hot","dot","dog","lot","log","cog"]
"""

def ladderLength(beginWord,endWord,wordList):
    graph = {}
    for char in wordList:
        for s in range(len(char)):
            key = char[:s] +'*'+ char[s+1:]
            if key not in graph:
                graph[key] = [char]
            else:
                graph[key].append(char)    
    queue = deque()
    queue.append([beginWord,0])
    visited = set()
    while queue:
        node,length = queue.popleft()
        if node == endWord:
            return length+1
        for i in range(len(node)):
            common = node[:i] +'*'+ node[i+1:]
            if common in graph:
                for key in graph[common]:
                    if key not in visited:
                        visited.add(key)
                        queue.append([key,length+1])
    
    return -1

"""
1091. Shortest Path in Binary Matrix

In an N by N square grid, each cell is either empty (0) or blocked (1).

A clear path from top-left to bottom-right has length k if and only if it is composed of cells C_1, C_2, ..., C_k such that:

Adjacent cells C_i and C_{i+1} are connected 8-directionally (ie., they are different and share an edge or corner)
C_1 is at location (0, 0) (ie. has value grid[0][0])
C_k is at location (N-1, N-1) (ie. has value grid[N-1][N-1])
If C_i is located at (r, c), then grid[r][c] is empty (ie. grid[r][c] == 0).
Return the length of the shortest such clear path from top-left to bottom-right.  If such a path does not exist, return -1.
Example 1:

Input: [[0,1],[1,0]]

Output: 2 --->BFS

"""

def shortestPathBinaryMatrix(grid):
    visited = set()
    pathLen = 1
    row = len(grid)
    col = len(grid[0])
    if grid[0][0]==1:
        return -1
    target = (row-1,col-1)
    q = []
    q.append((0,0))
    visited.add((0,0))
    while q:
        qLen = len(q)
        while qLen:
            x,y=node=q.pop(0)
            if node == target:
                return pathLen
            for g in [(x-1,y-1),(x+1,y+1),(x,y+1),(x,y-1),(x-1,y),(x+1,y),(x+1,y-1),(x-1,y+1)]:
                i,j = g
                if 0<=i<row and 0<=j<col and grid[i][j]==0 and g not in visited:
                    q.append(g)
                    visited.add(g)
            qLen -= 1
        pathLen += 1
    return -1

"""
1010. Pairs of Songs With Total Durations Divisible by 60

You are given a list of songs where the ith song has a duration of time[i] seconds.

Return the number of pairs of songs for which their total duration in seconds is divisible by 60. Formally, we want the number of indices i, j such that i < j with (time[i] + time[j]) % 60 == 0.

Example 1:

Input: time = [30,20,150,100,40]
Output: 3
Explanation: Three pairs have a total duration divisible by 60:
(time[0] = 30, time[2] = 150): total duration 180
(time[1] = 20, time[3] = 100): total duration 120
(time[1] = 20, time[4] = 40): total duration 60
"""
def numPairsDivisibleBy60(time):
    remainders = collections.defaultdict(int)
    ret = 0
    for t in time:
        if t % 60 == 0: # check if a%60==0 && b%60==0
            ret += remainders[0]
        else: # check if a%60+b%60==60
            ret += remainders[60-t%60]
        remainders[t % 60] += 1 # remember to update the remainders
    return ret
"""
1120. Maximum Average Subtree

Given the root of a binary tree, find the maximum average value of any subtree of that tree.

(A subtree of a tree is any node of that tree plus all its descendants. The average value of a tree is the sum of its values, divided by the number of nodes.)

Example 1:

Input: [5,6,1]
Output: 6.00000
Explanation: 
For the node with value = 5 we have an average of (5 + 6 + 1) / 3 = 4.
For the node with value = 6 we have an average of 6 / 1 = 6.
For the node with value = 1 we have an average of 1 / 1 = 1.
So the answer is 6 which is the maximum.
"""
def maxAverageSubtree(root):
    maxAve = float('-inf')
    dfs(root)
    return maxAve
    
    def dfs(root):
        nonlocal maxAve
        if not root:
            return (0,0)
        leftSum, leftSize = dfs(root.left)
        rightSum, rightSize = dfs(root.right)
        _sum = root.val + leftSum + rightSum
        size = 1 + rightSize + leftSize
        ave = _sum//size
        maxAve = max(maxAve,ave)
        return _sum,size

"""
123. Best Time to Buy and Sell Stock III

Say you have an array for which the ith element is the price of a given stock on day i.

Design an algorithm to find the maximum profit. You may complete at most two transactions.

Note: You may not engage in multiple transactions at the same time (i.e., you must sell the stock before you buy again).

Example 1:

Input: prices = [3,3,5,0,0,3,1,4]
Output: 6
Explanation: Buy on day 4 (price = 0) and sell on day 6 (price = 3), profit = 3-0 = 3.
Then buy on day 7 (price = 1) and sell on day 8 (price = 4), profit = 4-1 = 3.
Example 2:

Input: prices = [1,2,3,4,5]
Output: 4
Explanation: Buy on day 1 (price = 1) and sell on day 5 (price = 5), profit = 5-1 = 4.
Note that you cannot buy on day 1, buy on day 2 and sell them later, as you are engaging multiple transactions at the same time. You must sell before buying again.
"""
def maxProfit(prices):
    """
    :type prices: List[int]
    :rtype: int
    """
    if len(prices) <= 1:
        return 0

    left_min = prices[0]
    right_max = prices[-1]

    length = len(prices)
    left_profits = [0] * length
    # pad the right DP array with an additional zero for convenience.
    right_profits = [0] * (length + 1)

    # construct the bidirectional DP array
    for l in range(1, length):
        left_profits[l] = max(left_profits[l-1], prices[l] - left_min)
        left_min = min(left_min, prices[l])

        r = length - 1 - l
        right_profits[r] = max(right_profits[r+1], right_max - prices[r])
        right_max = max(right_max, prices[r])

    max_profit = 0
    for i in range(0, length):
        max_profit = max(max_profit, left_profits[i] + right_profits[i+1])

    return max_profit
"""
1152. Analyze User Website Visit Pattern
We are given some website visits: the user with name username[i] visited the website website[i] at time timestamp[i].
A 3-sequence is a list of websites of length 3 sorted in ascending order by the time of their visits.  
(The websites in a 3-sequence are not necessarily distinct.)
Find the 3-sequence visited by the largest number of users. 
If there is more than one solution, return the lexicographically smallest such 3-sequence.
Example 1:
Input: username = ["joe","joe","joe","james","james","james","james","mary","mary","mary"], timestamp = [1,2,3,4,5,6,7,8,9,10], 
website = ["home","about","career","home","cart","maps","home","home","about","career"]
Output: ["home","about","career"]
Explanation: 
The tuples in this example are:
["joe", 1, "home"]
["joe", 2, "about"]
["joe", 3, "career"]
["james", 4, "home"]
["james", 5, "cart"]
["james", 6, "maps"]
["james", 7, "home"]
["mary", 8, "home"]
["mary", 9, "about"]
["mary", 10, "career"]
The 3-sequence ("home", "about", "career") was visited at least once by 2 users.
The 3-sequence ("home", "cart", "maps") was visited at least once by 1 user.
The 3-sequence ("home", "cart", "home") was visited at least once by 1 user.
The 3-sequence ("home", "maps", "home") was visited at least once by 1 user.
The 3-sequence ("cart", "maps", "home") was visited at least once by 1 user.
"""
from heapq import heappop, heappush, heapify 
from itertools import combinations
def mostVisitedPattern(username, timestamp, website):
        '''
        1) The idea is to use min heap to sort the websites visited by each user in ascending order.
        2) then use a dict with users as keys and visited website as values
        3) traverse through each of these lists of websites and create a sequence of 3 for all possible combinations
        4) find the count for each of the sequence
        '''
        queue = []
        heapify(queue)
        #1) sort data based on timestamp - O(logn) where n = number of users
        for uname,tstamp,wsite in zip(username,timestamp,website):
            heappush(queue, (tstamp,wsite,uname))
        #print("queue", queue)
        user_dict = defaultdict(list)
        #2) categorize websites based on users - O(n)
        while queue:
            _,web,user = heappop(queue)
            user_dict[user].append(web)
        #print("Dict",user_dict)
        
        seq_count_dict = defaultdict(int)
        max_count = 0
        result = tuple()
        print("UserDict",user_dict)
        #3) traverse thriugh all websites to fins sequence of 3 - O(n*k)
        for websites in user_dict.values():
            seq_combinations = combinations(websites,3) #O(k^3) where k is max number of websites visted by a user
            for seq in set(seq_combinations): # since we want the count of a sequence visited by most number of users, if a user visits the same sequence multiple times, it is counted as 1
                seq_count_dict[seq] += 1
                if seq_count_dict[seq] > max_count:
                    max_count = seq_count_dict[seq]
                    result = seq
                elif seq_count_dict[seq] == max_count: #If the count is same and you find a sequence with a smaller lexographical order
                    if seq < result:
                        print("Seq",seq,result)
                        result = seq
        return list(result)
        #Time Complexity: O(n*k^3)
"""
1167. Minimum Cost to Connect Sticks

You have some number of sticks with positive integer lengths. These lengths are given as an array sticks, where sticks[i] is the length of the ith stick.

You can connect any two sticks of lengths x and y into one stick by paying a cost of x + y. You must connect all the sticks until there is only one stick remaining.

Return the minimum cost of connecting all the given sticks into one stick in this way.

Example 1:

Input: sticks = [2,4,3]
Output: 14
Explanation: You start with sticks = [2,4,3].
1. Combine sticks 2 and 3 for a cost of 2 + 3 = 5. Now you have sticks = [5,4].
2. Combine sticks 5 and 4 for a cost of 5 + 4 = 9. Now you have sticks = [9].
There is only one stick left, so you are done. The total cost is 5 + 9 = 14.
"""       
import heapq

def connectSticks(sticks):
    
    if not sticks or len(sticks) < 2:
        return 0
    
    heapq.heapify(sticks)
    total_cost = 0
    
    while len(sticks) > 1:
        current_cost = heapq.heappop(sticks) + heapq.heappop(sticks)
        total_cost += current_cost
        heapq.heappush(sticks, current_cost)
        
    return total_cost

"""
1268. Search Suggestions System

Given an array of strings products and a string searchWord. We want to design a system that suggests at most three product names from products after each character of searchWord is typed. Suggested products should have common prefix with the searchWord. If there are more than three products with a common prefix return the three lexicographically minimums products.

Return list of lists of the suggested products after each character of searchWord is typed. 

Example 1:

Input: products = ["mobile","mouse","moneypot","monitor","mousepad"], searchWord = "mouse"
Output: [
["mobile","moneypot","monitor"],
["mobile","moneypot","monitor"],
["mouse","mousepad"],
["mouse","mousepad"],
["mouse","mousepad"]
]
Explanation: products sorted lexicographically = ["mobile","moneypot","monitor","mouse","mousepad"]
After typing m and mo all products match and we show user ["mobile","moneypot","monitor"]
After typing mou, mous and mouse the system suggests ["mouse","mousepad"]
"""
def suggestedProducts(products,searchWord):
    products = sorted(products)
    slen = len(searchWord)
    output = []
    
    for i in range(slen):
        tem = []
        curr = searchWord[:i]
        count = 0
        for j in range(len(products)):
            if count < 3:
                if curr == products[j][:i]:
                    tem.append(products[j])
                    count += 1
        output.append(tem)
    
    return output

"""
139. Word Break O(n^3)

Given a non-empty string s and a dictionary wordDict containing a list of non-empty words, determine if s can be segmented into a space-separated sequence of one or more dictionary words.

Note:

The same word in the dictionary may be reused multiple times in the segmentation.
You may assume the dictionary does not contain duplicate words.
Example 1:

Input: s = "leetcode", wordDict = ["leet", "code"]
Output: true
Explanation: Return true because "leetcode" can be segmented as "leet code".
"""
def wordBreak1(s, wordDict):
    words = set(wordDict)
    words_length = set(len(word) for word in words)
    
    def _search(i):
        if i == len(s):
            return True
        result = False
        for length in words_length:
            if s[i:i+length] in words:
                 result = result or _search(i+length)
        return result
    
    return _search(0)

def wordBreak(s, wordDict):
    dp = [False]*(len(s)+1)
    dp[0] = True
    for i in range(len(s)):
        if dp[i]:
            for j in range(i+1,len(s)+1):
                if s[i:j] in wordDict:
                    dp[j] = True
               
    return dp[-1]
"""
140. Word Break II

Given a non-empty string s and a dictionary wordDict containing a list of non-empty words, 
add spaces in s to construct a sentence where each word is a valid dictionary word. 
Return all such possible sentences.

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

import collections
class LFUCache:

    def __init__(self, capacity: int):
        self.capacity = capacity                                         # stores the capacity
        self.di = {}                                                     # stores keys with their frequencies
        self.lfu = defaultdict(collections.OrderedDict)                             # stores frequencies and all keys that have a given frequency
        self.least = 0                                                   # stores the minimum frequence which exists in the cache. We need this for eviction
        
    def update(self, key, value = None):                                 # help function which update the frequency of a key
        poz = self.di[key] + 1                                           # if a key frequency was N then after we visit it, the frequency changes to N + 1
        v = self.lfu[poz - 1].pop(key)                                   # if we update the position of a key in the cache then we should maintain its last value
        if value is not None:                                            # we call the update function in 2 cases: 1. From get function. In this case we maintain its last value; 2. From put function. In this case we should change the key's value with a new one
            v = value                                                    # 2nd case
        self.lfu[poz][key], self.di[key] = v, poz                        # update the key in both dictionaries
        if not self.lfu[poz - 1] and self.least == poz - 1:              # if there a no more keys with the Nth frequence, and the Nth frequence was the minimal frequence then we need to increment the minimum frequence
            self.least += 1
        print("LFU DIct",self.lfu,poz,key)
        return self.lfu[poz][key]                                        # this line is used only when the updated function was called from the get function
    
    def get(self, key: int) -> int:
        return self.update(key) if key in self.di else -1                # if we find a key, then we should update its position and return its value, otherwise we return -1

    def put(self, key: int, value: int) -> None:
        if not self.capacity: return                                     # we need this line for the case when the capacity is 0 as we can't put anything
        if key in self.di: self.update(key, value)                       # if the key is already in our cache then we only update its value with a new one
        else:                                                            # the key isn't in our cache. Its frequence becomes 1
            if len(self.di) == self.capacity:                            # the cache reached its capacity
                del self.di[self.lfu[self.least].popitem(last=False)[0]] # firstly, we remove the key with the minimum frequence, and then we delete the key from the cache
            self.lfu[1][key] = value                                     # last 3 lines put the key in the cache
            self.di[key] = 1
            self.least = 1

lfu = LFUCache(2)
print(lfu.put(1,1))
print(lfu.put(2,2))
print("LFU",lfu.get(1))
print(lfu.put(3,3))
print("LFU",lfu.get(2))
"""
200. Number of Islands

Given an m x n 2d grid map of '1's (land) and '0's (water), return the number of islands.

An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. 
You may assume all four edges of the grid are all surrounded by water.

Example 1:

Input: grid = [
  ["1","1","1","1","0"],
  ["1","1","0","1","0"],
  ["1","1","0","0","0"],
  ["0","0","0","0","0"]
]
Output: 1
"""
def island(grid):
    if len(grid) ==0:
        return 0
    rows = len(grid)
    cols = len(grid[0])
    visited = [[False for i in range(cols)] for j in range(rows)]
    
    islands_Count = 0
    
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 1 and visited[i][j] == False:
                #visited[i][j] = True
                print(i,j)
                DFS(grid,i,j,visited,rows,cols)
                islands_Count += 1
    return islands_Count

def DFS(grid, i, j, visited, rows, cols):
    visited[i][j] = True
    print(i,j,visited)
    if i-1 >= 0 and visited[i-1][j]== False and grid[i-1][j]==1:
        DFS(grid,i-1,j,visited,rows,cols)
    if i+1 < rows and visited[i+1][j]== False and grid[i+1][j]==1:
        DFS(grid,i+1,j,visited,rows,cols)
    if j-1 >= 0 and visited[i][j-1]== False and grid[i][j-1]==1:
        DFS(grid,i,j-1,visited,rows,cols)
    if j+1 < cols and visited[i][j+1]==False and grid[i][j+1]==1:
        DFS(grid,i,j+1,visited,rows,cols)
        
    return    


"""
694. Number of Distinct Islands

Given a non-empty 2D array grid of 0's and 1's, an island is a group of 1's (representing land) connected 4-directionally (horizontal or vertical.) You may assume all four edges of the grid are surrounded by water.

Count the number of distinct islands. An island is considered to be the same as another if and only if one island can be translated (and not rotated or reflected) to equal the other.

Example 1:
11000
11000
00011
00011
Given the above grid map, return 1.
Example 2:
11011
10000
00001
11011
Given the above grid map, return 3.
"""
def numDistinctIslands(grid):
    M, N = len(grid),len(grid[0])
    # a queue for BFS
    q = collections.deque() 
    # a set for recording the pattern/ tranverse-path  / shape of island
    
    s = set()
    
    # iterate through each pixel: 
    for i in range(M):
        for j in range(N):
            
            #if it is 1: start BFS and use 2 to mark it as 'Searched'
            if grid[i][j] == 1: 
                grid[i][j] = 2
                
                # in current iteration state, (i,j) is the BFS starting point, and all operations below are within this island; 
                # path: to record the tranverse-path of the island
                path = []
                
                # classic BFS :
                q.append((i,j))
                while q:
                    a,b = q.popleft()
                    for x,y in [(a+1,b),(a-1,b),(a,b+1),(a,b-1)]:
                        if 0<=x<M and 0<=y<N:
                            if grid[x][y] == 1:
                                path.append((x-i,y-j))
                                grid[x][y] = 2
                                q.append((x,y))
                                
                s.add(tuple(path)) 
    return len(s)
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


"""
210. Course Schedule II

There are a total of n courses you have to take, labeled from 0 to n-1.

Some courses may have prerequisites, for example to take course 0 you have to first take course 1, which is expressed as a pair: [0,1]

Given the total number of courses and a list of prerequisite pairs, return the ordering of courses you should take to finish all courses.

There may be multiple correct orders, you just need to return one of them. If it is impossible to finish all courses, return an empty array.

Example 1:

Input: 2, [[1,0]] 
Output: [0,1]
Explanation: There are a total of 2 courses to take. To take course 1 you should have finished   
             course 0. So the correct course order is [0,1] 
"""
def courseFinish(numCourse, prerequisite):
    courseDict = defaultdict(list)
    for rel in prerequisite:
        nextCourse, prevCourse = rel[0], rel[1]
        courseDict[prevCourse].append(nextCourse)
    print(courseDict)
    checked = [False] * numCourse
    path = [False]* numCourse
    courseList = []
    for currCourse in range(numCourse):
        if iscyclic(currCourse, courseDict,checked,path):
            return []
        else:
            courseList.append(currCourse)
    #print(courseList)
    return courseList

def iscyclic(currC, courseD, checked, path):
    if checked[currC]:
        return False
    if path[currC]:
        return True
    path[currC] = True
    ret = False
    for child in courseD[currC]:
        ret = iscyclic(child, courseD, checked, path)
        if ret: break
    path[currC] = False
    checked[currC] = True
    print(checked)
    return ret

def courseWork(courses):
    courseDict = defaultdict(list)
    for rel in courses:
        preq,course = rel[0],rel[1]
        courseDict[preq].append(course)
    print(courseDict)
    seen = set()
    return DFSCourse(courses[0][0], seen, courseDict,[])

def DFSCourse(course, seen, courseDict,resultPath):
    seen.add(course)
    resultPath.append(course)
    for c in courseDict[course]:
        if c not in seen:
            DFSCourse(c,seen,courseDict,resultPath)
    return resultPath



def itemAssociation(items):
    itemDict = defaultdict(list)
    for item in items:
        currItem, assItem = item[0], item[1]
        itemDict[currItem].append(assItem)
    itemList = []
    for curItem in items:
        print(curItem)
        #itemList.append(curItem[0])
        result = iscyclicAmazon(curItem[0],itemDict,[])
        itemList.append(result)
    print(itemList)
    return max(itemList, key = lambda i: len(i))

def iscyclicAmazon(currC, courseD,resultPath):

    print("CurrC",currC)
    resultPath.append(currC)
    for child in courseD[currC]:
        iscyclicAmazon(child, courseD,resultPath)
    print(resultPath)
    return resultPath


"""
Product of the Array:
    
    Given an array nums of n integers where n > 1,  return an array output such that output[i] is equal to the product of all the elements of nums except nums[i].

Example:

Input:  [1,2,3,4]
Output: [24,12,8,6]
Constraint: It's guaranteed that the product of the elements of any prefix or suffix of the array (including the whole array) fits in a 32 bit integer.

Note: Please solve it without division and in O(n).
"""
def productExceptSelf(nums):
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
240. Search a 2D Matrix II

Write an efficient algorithm that searches for a value in an m x n matrix. This matrix has the following properties:

Integers in each row are sorted in ascending from left to right.
Integers in each column are sorted in ascending from top to bottom.
Example:

Consider the following matrix:

[
  [1,   4,  7, 11, 15],
  [2,   5,  8, 12, 19],
  [3,   6,  9, 16, 22],
  [10, 13, 14, 17, 24],
  [18, 21, 23, 26, 30]
]
Given target = 5, return true.

Given target = 20, return false.
"""
class Solution:
    def binary_search(self, matrix, target, start, vertical):
        lo = start
        hi = len(matrix[0])-1 if vertical else len(matrix)-1

        while hi >= lo:
            mid = (lo + hi)//2
            if vertical: # searching a column
                if matrix[start][mid] < target:
                    lo = mid + 1
                elif matrix[start][mid] > target:
                    hi = mid - 1
                else:
                    return True
            else: # searching a row
                if matrix[mid][start] < target:
                    lo = mid + 1
                elif matrix[mid][start] > target:
                    hi = mid - 1
                else:
                    return True
        
        return False

    def searchMatrix(self, matrix, target):
        # an empty matrix obviously does not contain `target`
        if not matrix:
            return False

        # iterate over matrix diagonals starting in bottom left.
        for i in range(min(len(matrix), len(matrix[0]))):
            vertical_found = self.binary_search(matrix, target, i, True)
            horizontal_found = self.binary_search(matrix, target, i, False)
            if vertical_found or horizontal_found:
                return True
        
        return False


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
Tic Tac Toe Game
"""

class TicTacToe:
    def __init__(self, n: int):
        """
        Initialize your data structure here.
        """
        self.mat = [[0]*n for i in range(n)]
        self.size = n

    def move(self, row: int, col: int, player: int) -> int:
        """
        Player {player} makes a move at ({row}, {col}).
        @param row The row of the board.
        @param col The column of the board.
        @param player The player, can be either 1 or 2.
        @return The current winning condition, can be either:
                0: No one wins.
                1: Player 1 wins.
                2: Player 2 wins.
        """
        if self.mat[row][col] == 0: self.mat[row][col] = player
        for i in range(self.size):
            if self.mat[row][i] != player: break
            if i == self.size - 1:return player
        for i in range(self.size):
            if self.mat[i][col] != player: break
            if i == self.size - 1:return player
        for i in range(self.size):
            if self.mat[i][i] != player: break
            if i == self.size-1:return player
        for i in range(self.size):
            if self.mat[i][self.size-i-1] != player: break
            if i == self.size-1:return player 
        return 0

"""
Time complexity. GetRandom is always \mathcal{O}(1)O(1). 
Insert and Delete both have \mathcal{O}(1)O(1) average time complexity, 
and \mathcal{O}(N)O(N) in the worst-case scenario when the operation exceeds the capacity of currently allocated array/hashmap and invokes space reallocation.

Space complexity: \mathcal{O}(N)O(N), to store N elements.
380. Insert Delete GetRandom O(1)

Implement the RandomizedSet class:

bool insert(int val) Inserts an item val into the set if not present. Returns true if the item was not present, false otherwise.
bool remove(int val) Removes an item val from the set if present. Returns true if the item was present, false otherwise.
int getRandom() Returns a random element from the current set of elements (it's guaranteed that at least one element exists when this method is called). Each element must have the same probability of being returned.
Follow up: Could you implement the functions of the class with each function works in average O(1) time?

 

Example 1:

Input
["RandomizedSet", "insert", "remove", "insert", "getRandom", "remove", "insert", "getRandom"]
[[], [1], [2], [2], [], [1], [2], []]
Output
[null, true, false, true, 2, true, false, 2]

Explanation
RandomizedSet randomizedSet = new RandomizedSet();
randomizedSet.insert(1); // Inserts 1 to the set. Returns true as 1 was inserted successfully.
randomizedSet.remove(2); // Returns false as 2 does not exist in the set.
randomizedSet.insert(2); // Inserts 2 to the set, returns true. Set now contains [1,2].
randomizedSet.getRandom(); // getRandom() should return either 1 or 2 randomly.
randomizedSet.remove(1); // Removes 1 from the set, returns true. Set now contains [2].
randomizedSet.insert(2); // 2 was already in the set, so return false.
randomizedSet.getRandom(); // Since 2 is the only number in the set, getRandom() will always return 2.
"""
from random import choice
class RandomizedSet():
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.dict = {}
        self.list = []

        
    def insert(self, val: int) -> bool:
        """
        Inserts a value to the set. Returns true if the set did not already contain the specified element.
        """
        if val in self.dict:
            return False
        self.dict[val] = len(self.list)
        self.list.append(val)
        return True
        

    def remove(self, val: int) -> bool:
        """
        Removes a value from the set. Returns true if the set contained the specified element.
        """
        if val in self.dict:
            # move the last element to the place idx of the element to delete
            last_element, idx = self.list[-1], self.dict[val]
            self.list[idx], self.dict[last_element] = last_element, idx
            # delete the last element
            self.list.pop()
            del self.dict[val]
            return True
        return False

    def getRandom(self) -> int:
        """
        Get a random element from the set.
        """
        return choice(self.list)
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
from collections import defaultdict
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
            if s not in g or e not in g: res.append(-1.0); continue
            elif s == e                : res.append( 1.0); continue
            elif e in g[s]             : res.append(g[s][e]); continue
            self.res = -1.0            
            dfs(s, e, 1.0, set())
            res.append(self.res)
        return res    
s = Solution()
print(s.calcEquation([["a", "b"],["b", "c"]],[2.0,3.0],[["a", "c"],["b", "a"],["a", "e"],["a", "a"],["x", "x"]]))
"""
437. Path Sum III
Medium

4403

309

Add to List

Share
You are given a binary tree in which each node contains an integer value.

Find the number of paths that sum to a given value.

The path does not need to start or end at the root or a leaf, but it must go downwards (traveling only from parent nodes to child nodes).

The tree has no more than 1,000 nodes and the values are in the range -1,000,000 to 1,000,000.

Example:

root = [10,5,-3,3,2,null,11,3,-2,null,1], sum = 8

      10
     /  \
    5   -3
   / \    \
  3   2   11
 / \   \
3  -2   1

Return 3. The paths that sum to 8 are:

1.  5 -> 3
2.  5 -> 2 -> 1
3. -3 -> 11
"""
def pathSum(root, sum):
    count = 0
    def preOrder(node,currSum):
        nonlocal count
        if not node:
            return
        currSum += node.val
        if currSum == sum:
            count += 1
        
        count += h[currSum-sum]
        h[currSum] += 1
        
        preOrder(node.left,currSum)
        preOrder(node.right,currSum)
        
        h[currSum] -= 1
        
    h = defaultdict(int)
    preOrder(root,0)
    return count

"""
692. Top K Frequent Words

Given a non-empty list of words, return the k most frequent elements.

Your answer should be sorted by frequency from highest to lowest. If two words have the same frequency, then the word with the lower alphabetical order comes first.

Example 1:
Input: ["i", "love", "leetcode", "i", "love", "coding"], k = 2
Output: ["i", "love"]
Explanation: "i" and "love" are the two most frequent words.
    Note that "i" comes before "love" due to a lower alphabetical order.
"""

def topKFrequent(words, k):
    count = collections.Counter(words)
    candidates = list(count.keys())
    print(candidates,count)
    candidates.sort(key = lambda w: (-count[w], w))
    print(candidates)
    return candidates[:k]



"""
763. Partition Labels

A string S of lowercase English letters is given. 
We want to partition this string into as many parts as possible so that each letter appears in at most one part, 
and return a list of integers representing the size of these parts.

 

Example 1:

Input: S = "ababcbacadefegdehijhklij"
Output: [9,7,8]
Explanation:
The partition is "ababcbaca", "defegde", "hijhklij".
This is a partition so that each letter appears in at most one part.
A partition like "ababcbacadefegde", "hijhklij" is incorrect, because it splits S into less parts.
"""
def partitionLabels(S):
    d = {}
    for i in range(len(S)):
        d[S[i]] = i
    left = 0
    right = 0
    last = -1
    ans = []
    while right < len(S):
        letter = S[right]
        last = max(last, d[letter])
        if right == last:
            ans.append(right-left + 1)
            left = right + 1
        right += 1
    return ans

def mostCommonWord(paragraph,banned):
    
    clean_paragraph = ""
    for c in paragraph:
        if c in {',', '.', '!', ' '}:
            clean_paragraph += " "
        elif c.isalpha():
            clean_paragraph += c.lower()
            
    banned = set([x.lower().strip() for x in banned])
            
    words = [x for x in clean_paragraph.split() if x not in banned]
    
    counts = collections.Counter(words)
    print(counts)
    return counts.most_common(1)[0][0]

"""
863. All Nodes Distance K in Binary Tree

We are given a binary tree (with root node root), a target node, and an integer value K.

Return a list of the values of all nodes that have a distance K from the target node.  The answer can be returned in any order.

Example 1:

Input: root = [3,5,1,6,2,0,8,null,null,7,4], target = 5, K = 2

Output: [7,4,1]

Explanation: 
The nodes that are a distance 2 from the target node (with value 5)
have values 7, 4, and 1.

Note that the inputs "root" and "target" are actually TreeNodes.
The descriptions of the inputs above are just serializations of these objects.
"""
class Solution():
    def distanceK(self, root, target, K):
        def dfs(node, par = None):
            if node:
                node.par = par
                dfs(node.left, node)
                dfs(node.right, node)

        dfs(root)

        queue = collections.deque([(target, 0)])
        seen = {target}
        while queue:
            if queue[0][1] == K:
                return [node.val for node, d in queue]
            node, d = queue.popleft()
            for nei in (node.left, node.right, node.par):
                if nei and nei not in seen:
                    seen.add(nei)
                    queue.append((nei, d+1))
        return []

"""
901. Online Stock Span

Write a class StockSpanner which collects daily price quotes for some stock, and returns the span of that stock's price for the current day.

The span of the stock's price today is defined as the maximum number of consecutive days (starting from today and going backwards) for which the price of the stock was less than or equal to today's price.

For example, if the price of a stock over the next 7 days were [100, 80, 60, 70, 60, 75, 85], then the stock spans would be [1, 1, 1, 2, 1, 4, 6].

Example 1:

Input: ["StockSpanner","next","next","next","next","next","next","next"], [[],[100],[80],[60],[70],[60],[75],[85]]
Output: [null,1,1,1,2,1,4,6]
Explanation: 
First, S = StockSpanner() is initialized.  Then:
S.next(100) is called and returns 1,
S.next(80) is called and returns 1,
S.next(60) is called and returns 1,
S.next(70) is called and returns 2,
S.next(60) is called and returns 1,
S.next(75) is called and returns 4,
S.next(85) is called and returns 6.
Note that (for example) S.next(75) returned 4, because the last 4 prices
(including today's price of 75) were less than or equal to today's price.
"""
class StockSpanner():
    def __init__(self):
        self.stack = []

    def next(self, price):
        weight = 1
        while self.stack and self.stack[-1][0] <= price:
            weight += self.stack.pop()[1]
        self.stack.append((price, weight))
        return weight


"""
957. Prison Cells After N Days

There are 8 prison cells in a row, and each cell is either occupied or vacant.

Each day, whether the cell is occupied or vacant changes according to the following rules:

If a cell has two adjacent neighbors that are both occupied or both vacant, then the cell becomes occupied.
Otherwise, it becomes vacant.
(Note that because the prison is a row, the first and the last cells in the row can't have two adjacent neighbors.)

We describe the current state of the prison in the following way: cells[i] == 1 if the i-th cell is occupied, else cells[i] == 0.

Given the initial state of the prison, return the state of the prison after N days (and N such changes described above.)

Example 1:

Input: cells = [0,1,0,1,1,0,0,1], N = 7
Output: [0,0,1,1,0,0,0,0]
Explanation: 
The following table summarizes the state of the prison on each day:
Day 0: [0, 1, 0, 1, 1, 0, 0, 1]
Day 1: [0, 1, 1, 0, 0, 0, 0, 0]
Day 2: [0, 0, 0, 0, 1, 1, 1, 0]
Day 3: [0, 1, 1, 0, 0, 1, 0, 0]
Day 4: [0, 0, 0, 0, 0, 1, 0, 0]
Day 5: [0, 1, 1, 1, 0, 1, 0, 0]
Day 6: [0, 0, 1, 0, 1, 1, 0, 0]
Day 7: [0, 0, 1, 1, 0, 0, 0, 0]
"""
def prisonAfterNDays(cells,N):
    #if N>10:
    pre=cells[:]
    print(pre)
    N=(N-1)%14 + 1
    for i in range(N):
        post=pre[:]
        #print(post)
        for j in range(1,7):
            pre[j]= int(post[j-1]==post[j+1])
        pre[0],pre[-1]=0,0
    return pre

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

def kClosest(points,K):
    points.sort(key=lambda P:P[0]**2+P[1]**2)
    return points[:K]
"""
212. Word Search II

Given an m x n board of characters and a list of strings words, return all words on the board.

Each word must be constructed from letters of sequentially adjacent cells, where adjacent cells are horizontally or vertically neighboring. The same letter cell may not be used more than once in a word.

Example 1:

Input: board = [["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]], words = ["oath","pea","eat","rain"]
Output: ["eat","oath"]
"""

def findWords(board,words):
    WORD_KEY = '$'
    
    trie = {}
    for word in words:
        node = trie
        for letter in word:
            # retrieve the next node; If not found, create a empty node.
            node = node.setdefault(letter, {})
        # mark the existence of a word in trie node
        node[WORD_KEY] = word
    print("Trie",trie)
    print("Node",node)
    rowNum = len(board)
    colNum = len(board[0])
    
    matchedWords = []
    
    def backtracking(row, col, parent):    
        
        letter = board[row][col]
        currNode = parent[letter]
        # check if we find a match of word
        word_match = currNode.pop(WORD_KEY, False)
        print("Word",word_match)
        if word_match:
            # also we removed the matched word to avoid duplicates,
            #   as well as avoiding using set() for results.
            matchedWords.append(word_match)
        
        # Before the EXPLORATION, mark the cell as visited 
        board[row][col] = '#'
        
        # Explore the neighbors in 4 directions, i.e. up, right, down, left
        for (rowOffset, colOffset) in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
            newRow, newCol = row + rowOffset, col + colOffset     
            if newRow < 0 or newRow >= rowNum or newCol < 0 or newCol >= colNum:
                continue
            if not board[newRow][newCol] in currNode:
                continue
            backtracking(newRow, newCol, currNode)
    
        # End of EXPLORATION, we restore the cell
        board[row][col] = letter
    
        # Optimization: incrementally remove the matched leaf node in Trie.
        if not currNode:
            parent.pop(letter)

    for row in range(rowNum):
        for col in range(colNum):
            # starting from each of the cells
            if board[row][col] in trie:
                print("In Trie",board[row][col],trie)
                backtracking(row, col, trie)
    
    return matchedWords       

"""
239. Sliding Window Maximum

You are given an array of integers nums, there is a sliding window of size k which is moving from the very left of the array to the very right. You can only see the k numbers in the window. Each time the sliding window moves right by one position.

Return the max sliding window.

 

Example 1:

Input: nums = [1,3,-1,-3,5,3,6,7], k = 3
Output: [3,3,5,5,6,7]
Explanation: 
Window position                Max
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7
"""

def maxSlidingWindow(nums,k):
    n = len(nums)
    if n * k == 0:
        return []
    
    return [max(nums[i:i + k]) for i in range(n - k + 1)]

"""
273. Integer to English Words

Convert a non-negative integer num to its English words representation.

 

Example 1:

Input: num = 123
Output: "One Hundred Twenty Three"
"""

def numberToWords(num):
    if num == 0:
        return 'Zero'
    Ths = ['','Thousand ','Million ','Billion ']
    numdict = {1:'One ',2:'Two ',3:'Three ',4:'Four ',5:'Five ', 6:'Six ',7:'Seven ',8:'Eight ',9:'Nine ',10:'Ten ',11:'Eleven ', 12:'Twelve ',13:'Thirteen ',14:'Fourteen ',15:'Fifteen ',16:'Sixteen ', 17:'Seventeen ',18:'Eighteen ',19:'Nineteen '}
    num10dict = {2:'Twenty ',3:'Thirty ', 4:'Forty ',5:'Fifty ',6:'Sixty ',7:'Seventy ',8:'Eighty ',9:'Ninety '}
    sections = []
    while num:
        sections.append(num%1000)
        num //= 1000
    print(sections)
    sections = sections[::-1]
    print(sections)
    out = []
    for i,s in enumerate(sections):
        temp = s//100
        print(temp)
        if temp:
            out.append(numdict[temp]+'Hundred ')
        temp = s%100
        if temp and temp<20:
            out.append(numdict[temp])
        elif temp and temp>=20:
            temp1 = temp//10
            out.append(num10dict[temp1])
            temp2 = temp%10
            if temp2:
                out.append(numdict[temp2])
        if s:
            out.append(Ths[len(sections)-i-1])
    print(out)
    return ''.join(out)[:-1] 

"""
295. Find Median from Data Stream

Median is the middle value in an ordered integer list. If the size of the list is even, there is no middle value. So the median is the mean of the two middle value.

For example,
[2,3,4], the median is 3

[2,3], the median is (2 + 3) / 2 = 2.5

Design a data structure that supports the following two operations:

void addNum(int num) - Add a integer number from the data stream to the data structure.
double findMedian() - Return the median of all elements so far.
 

Example:

addNum(1)
addNum(2)
findMedian() -> 1.5
addNum(3) 
findMedian() -> 2
"""
class MedianFinder(object):

    def __init__(self):
        self.maxHeap = []
        self.minHeap = []
        
    def addNum(self, num):
        # Push the number onto one of the heaps
        if len(self.maxHeap) == 0 or -self.maxHeap[0] >= num:
            heappush(self.maxHeap, -num)
        else:
            heappush(self.minHeap, num)
        
        print("Before Max and Min Heap",self.maxHeap,self.minHeap)
        # Balance out the heaps
        if len(self.maxHeap) > len(self.minHeap) + 1:
            heappush(self.minHeap, -heappop(self.maxHeap))
        elif len(self.minHeap) > len(self.maxHeap):
            heappush(self.maxHeap, -heappop(self.minHeap))
        
        print("Max and Min Heap",self.maxHeap,self.minHeap)

    def findMedian(self):
        # If heap sizes are equal
        if len(self.maxHeap) == len(self.minHeap):
            return -self.maxHeap[0] / 2.0 + self.minHeap[0] / 2.0
        
        # If maxHeap > minHeap
        return -self.maxHeap[0]
    

if __name__ == '__main__':
    VT = MedianFinder()
    VT.addNum(9)
    VT.addNum(7)
    VT.addNum(18)
    VT.addNum(11)
    print("Median",VT.findMedian())
    
"""
1335. Minimum Difficulty of a Job Schedule

You want to schedule a list of jobs in d days. Jobs are dependent (i.e To work on the i-th job, you have to finish all the jobs j where 0 <= j < i).

You have to finish at least one task every day. The difficulty of a job schedule is the sum of difficulties of each day of the d days. The difficulty of a day is the maximum difficulty of a job done in that day.

Given an array of integers jobDifficulty and an integer d. The difficulty of the i-th job is jobDifficulty[i].

Return the minimum difficulty of a job schedule. If you cannot find a schedule for the jobs return -1.

 

Example 1:


Input: jobDifficulty = [6,5,4,3,2,1], d = 2
Output: 7
Explanation: First day you can finish the first 5 jobs, total difficulty = 6.
Second day you can finish the last job, total difficulty = 1.
The difficulty of the schedule = 6 + 1 = 7 
""" 
def minDifficulty(jobDifficulty, d): #O(d*n^2) and Space O(n*d)Bottom UP
    """
    :type jobDifficulty: List[int]
    :type d: int
    :rtype: int
    """
    if len(jobDifficulty) < d:
        return -1
    
    n = len(jobDifficulty)
    
    maxval = collections.defaultdict(int)
    for i in range(1,n+1):
        for k in range(i):
            maxval[(k,i)] = max(jobDifficulty[k:i])
    
    dp = [[float('inf')]*(d+1) for _ in range(n+1) ]
    dp[0][0] = 0
    for i in range(1,n+1):
        for j in range(1,min(i+1,d+1)):
            for k in range(i):
                dp[i][j] = min( dp[i][j], dp[k][j-1]+  maxval[(k,i)])
    
    return dp[n][d]

def minDifficulty(jobDifficulty,d):
    def arraymax(start,end):
        return max(jobDifficulty[start:end])

    @lru_cache(None)
    def recursion(prev, day):
        
        if day == 1:
            return arraymax(prev, len(jobDifficulty))
        difficulty = float('inf')
        for i in range(prev + 1, len(jobDifficulty) - day + 2):
            cur = arraymax(prev, i) + recursion(i, day - 1)
            difficulty = min(cur, difficulty)
        return difficulty

    res = recursion(0, d)
    if res == float('inf'):
        return -1
    else:
        return res

"""
472. Concatenated Words

Given a list of words (without duplicates), please write a program that returns all concatenated words in the given list of words.
A concatenated word is defined as a string that is comprised entirely of at least two shorter words in the given array.

Example:
Input: ["cat","cats","catsdogcats","dog","dogcatsdog","hippopotamuses","rat","ratcatdogcat"]

Output: ["catsdogcats","dogcatsdog","ratcatdogcat"]

Explanation: "catsdogcats" can be concatenated by "cats", "dog" and "cats"; 
 "dogcatsdog" can be concatenated by "dog", "cats" and "dog"; 
"ratcatdogcat" can be concatenated by "rat", "cat", "dog" and "cat".
"""

def findAllConcatenatedWordsInADict(words):
    seen={}
    s=set(words)
    words=list(s)
		
    def dfs(word):
        if not word or word in s:
            return True
        if word in seen:
            return seen[word]
        seen[word]=False
        for i in range(1,len(word)):
            if word[:i] in s:
                if dfs(word[i:]):
                    seen[word]=True
                    return True
        return seen[word]
    result=[]
    for i in words:
        s.remove(i)
        if len(i)>=2 and dfs(i) :
            result.append(i)
        s.add(i)                
    return result

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
130. Surrounded Regions

Given a 2D board containing 'X' and 'O' (the letter O), capture all regions surrounded by 'X'.

A region is captured by flipping all 'O's into 'X's in that surrounded region.

Example:

X X X X
X O O X
X X O X
X O X X
After running your function, the board should be:

X X X X
X X X X
X X X X
X O X X
Explanation:

Surrounded regions shouldn’t be on the border, which means that any 'O' on the border of the board are not flipped to 'X'. Any 'O' that is not on the border and it is not connected to an 'O' on the border will be flipped to 'X'. Two cells are connected if they are adjacent cells connected horizontally or vertically.
"""
class Board():
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
        borders = list(product(range(self.ROWS), [0, self.COLS-1])) + list(product([0, self.ROWS-1], range(self.COLS)))
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
    
    
    def DFS(self, board, row, col):
        if board[row][col] != 'O':
            return
        board[row][col] = 'E'
        if col < self.COLS-1: self.DFS(board, row, col+1)
        if row < self.ROWS-1: self.DFS(board, row+1, col)
        if col > 0: self.DFS(board, row, col-1)
        if row > 0: self.DFS(board, row-1, col)
    

b =Board()
print("Solve", b.solve([['X','X','X','X'],['X','O','O','X'],['X','X','O','X'],['X','O','X','X']]))

"""
588. Design In-Memory File System

Design an in-memory file system to simulate the following functions:

ls: Given a path in string format. If it is a file path, return a list that only contains this file's name. If it is a directory path, return the list of file and directory names in this directory. Your output (file and directory names together) should in lexicographic order.

mkdir: Given a directory path that does not exist, you should make a new directory according to the path. If the middle directories in the path don't exist either, you should create them as well. This function has void return type.

addContentToFile: Given a file path and file content in string format. If the file doesn't exist, you need to create that file containing given content. If the file already exists, you need to append given content to original content. This function has void return type.

readContentFromFile: Given a file path, return its content in string format.

Example:

Input: 
["FileSystem","ls","mkdir","addContentToFile","ls","readContentFromFile"]
[[],["/"],["/a/b/c"],["/a/b/c/d","hello"],["/"],["/a/b/c/d"]]

Output:
[null,[],null,null,["a"],"hello"]
"""
class FileSystem(object):

    def __init__(self):
        self.trie = {}

    def ls(self, path):
        """
        :type path: str
        :rtype: List[str]
        """
        if len(path) == 1: 
            return sorted(self.trie.keys())
        path = path.split('/')
        node = self.trie
        for p in path[1:]:
            node = node.setdefault(p, {})
        if type(node) == str:
            return [path[-1]]
        return sorted(node.keys())
        

    def mkdir(self, path):
        """
        :type path: str
        :rtype: None
        """
        path = path.split('/')
        node = self.trie
        for p in path[1:]:
            node = node.setdefault(p, {})
        

    def addContentToFile(self, filePath, content):
        """
        :type filePath: str
        :type content: str
        :rtype: None
        """
        path = filePath.split('/')
        f = path[-1]
        node = self.trie
        for p in path[1:-1]:
            node = node.setdefault(p, {})
        if f not in node:
            node[f] = content
        else:
            node[f] += content
        

    def readContentFromFile(self, filePath):
        """
        :type filePath: str
        :rtype: str
        """
        path = filePath.split('/')
        f = path[-1]
        node = self.trie
        for p in path[1:-1]:
            node = node.setdefault(p, {})
        
        return node[f]
    

"""
There are n servers numbered from 0 to n-1 connected by undirected server-to-server connections forming a network where connections[i] = [a, b] represents a connection between servers a and b. Any server can reach any other server directly or indirectly through the network.

A critical connection is a connection that, if removed, will make some server unable to reach some other server.

Return all critical connections in the network in any order.

Example 1:

Input: n = 4, connections = [[0,1],[1,2],[2,0],[1,3]]
Output: [[1,3]]
Explanation: [[3,1]] is also accepted.
"""
def criticalNetwork(n,connection):
    g = defaultdict(list)
    for u, v in connection:
        g[u].append(v)
        g[v].append(u)
    
    print(g)
    dfn = [None]*n
    low = [None]*n
    depth = 0
    ans = []
    def dfs(u,parent):
        nonlocal depth
        dfn[u]=low[u]= depth
        depth += 1
        print(dfn,low,depth)
        for v in g[u]:
            print("v", v)
            if dfn[v]== None:
                dfs(v,u)
                if dfn[u] < low[v]:
                    '''
                        low[x] = essentially a strongly connected network defined by the earliest node...
                        
                        
                        dfn[u] < low[v]
                        if depth of recursion of u is earlier than the "network of v defined by the earliest node,"
                        then its guaranteed that v is not reachable without the existing connection.
                        
                        dfn[u] >= low[v]
                        if depth of recursion of u is later than or equal to the "network of v defined by the earliest node,"
                        we know that u comes later than the network, so it is reachable
                    '''

                    print("Inside ans",dfn,low,depth)
                    ans.append([u,v])                
            if v!=parent:
                low[u] = min(low[u],low[v])
    dfs(0,None)
    return ans

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
103. Binary Tree Zigzag Level Order Traversal

Given a binary tree, return the zigzag level order traversal of its nodes' values. (ie, from left to right, then right to left for the next level and alternate between).

For example:
Given binary tree [3,9,20,null,null,15,7],
    3
   / \
  9  20
    /  \
   15   7
return its zigzag level order traversal as:
[
  [3],
  [20,9],
  [15,7]
]

BFS Python Solution O(N)
"""
def zigzagLevelOrder(root):
        if not root:
            return []
        res = []
        q = []
        q.append(root)
        #if odd get left->right level order
        #if even get right->left level order
        levelNum = 1
        while q:
            level = []
            for i in range(len(q)):
                cur = q.pop()
                if cur.left:
                    q.insert(0, cur.left)
                if cur.right:
                    q.insert(0, cur.right)
                level.append(cur.val)
            if levelNum%2==1:
                res.append(level)
                levelNum += 1
            else:
                res.append(reversed(level))
                levelNum += 1
        return res

"""
547. Friend Circles
 
There are N students in a class. Some of them are friends, while some are not. Their friendship is transitive in nature. For example, if A is a direct friend of B, and B is a direct friend of C, then A is an indirect friend of C. And we defined a friend circle is a group of students who are direct or indirect friends.

Given a N*N matrix M representing the friend relationship between students in the class. If M[i][j] = 1, then the ith and jth students are direct friends with each other, otherwise not. And you have to output the total number of friend circles among all the students.

Example 1:

Input: 
[[1,1,0],
 [1,1,0],
 [0,0,1]]
Output: 2
Explanation:The 0th and 1st students are direct friends, so they are in a friend circle. 
The 2nd student himself is in a friend circle. So return 2.
"""   
def friendCircle(matrix):
    n = len(matrix)
    seen = set()
    count = 0
    
    def DFS(node):
        for neighbor in range(n):
            if matrix[node][neighbor] and neighbor not in seen:
                seen.add(neighbor)
                DFS(neighbor)
    for i in range(n):
        if i not in seen:
            count += 1
            seen.add(i)
            DFS(i)
    
    return count

"""
572. Subtree of Another Tree

Given two non-empty binary trees s and t, check whether tree t has exactly the same structure and node values with a subtree of s. A subtree of s is a tree consists of a node in s and all of this node's descendants. The tree s could also be considered as a subtree of itself.

Example 1:
Given tree s:

     3
    / \
   4   5
  / \
 1   2
Given tree t:
   4 
  / \
 1   2
Return true, because t has the same structure and node values with a subtree of s.
"""

def isSubtree(s, t):
    """
    :type s: TreeNode
    :type t: TreeNode
    :rtype: bool
    """
    def isSame(s, t):
        if not s and not t:
            return True
        if not s or not t:
            return False
        return s.val == t.val and isSame(s.left, t.left) and isSame(s.right, t.right)
    
    queue = collections.deque([s])
    while queue:
        tree = queue.popleft()
        if isSame(tree, t):
            return True
        if tree.left:
            queue.append(tree.left)
        if tree.right:
            queue.append(tree.right)
            
    return False

"""
682. Baseball Game

You are keeping score for a baseball game with strange rules. The game consists of several rounds, where the scores of past rounds may affect future rounds' scores.

At the beginning of the game, you start with an empty record. You are given a list of strings ops, where ops[i] is the ith operation you must apply to the record and is one of the following:

An integer x - Record a new score of x.
"+" - Record a new score that is the sum of the previous two scores. It is guaranteed there will always be two previous scores.
"D" - Record a new score that is double the previous score. It is guaranteed there will always be a previous score.
"C" - Invalidate the previous score, removing it from the record. It is guaranteed there will always be a previous score.
Return the sum of all the scores on the record.

Example 1:

Input: ops = ["5","2","C","D","+"]
Output: 30
Explanation:
"5" - Add 5 to the record, record is now [5].
"2" - Add 2 to the record, record is now [5, 2].
"C" - Invalidate and remove the previous score, record is now [5].
"D" - Add 2 * 5 = 10 to the record, record is now [5, 10].
"+" - Add 5 + 10 = 15 to the record, record is now [5, 10, 15].
The total sum is 5 + 10 + 15 = 30.
"""

def calPoints(ops):
    stack = []
    for op in ops:
        if op == '+':
            stack.append(stack[-1] + stack[-2])
        elif op == 'C':
            stack.pop()
        elif op == 'D':
            stack.append(2 * stack[-1])
        else:
            stack.append(int(op))

    return sum(stack)

"""
726. Number of Atoms

Given a chemical formula (given as a string), return the count of each atom.

The atomic element always starts with an uppercase character, then zero or more lowercase letters, representing the name.

One or more digits representing that element's count may follow if the count is greater than 1. If the count is 1, no digits will follow. For example, H2O and H2O2 are possible, but H1O2 is impossible.

Two formulas concatenated together to produce another formula. For example, H2O2He3Mg4 is also a formula.

A formula placed in parentheses, and a count (optionally added) is also a formula. For example, (H2O2) and (H2O2)3 are formulas.

Given a formula, return the count of all elements as a string in the following form: the first name (in sorted order), followed by its count (if that count is more than 1), followed by the second name (in sorted order), followed by its count (if that count is more than 1), and so on.

Example 1:

Input: formula = "H2O"
Output: "H2O"
Explanation: The count of elements are {'H': 2, 'O': 1}.
"""

def countOfAtoms(formula):
    N = len(formula)
    stack = [collections.Counter()]
    i = 0
    while i < N:
        if formula[i] == '(':
            stack.append(collections.Counter())
            i += 1
        elif formula[i] == ')':
            top = stack.pop()
            i += 1
            i_start = i
            while i < N and formula[i].isdigit(): i += 1
            multiplicity = int(formula[i_start: i] or 1)
            for name, v in top.items():
                stack[-1][name] += v * multiplicity
        else:
            i_start = i
            i += 1
            while i < N and formula[i].islower(): i += 1
            name = formula[i_start: i]
            i_start = i
            while i < N and formula[i].isdigit(): i += 1
            multiplicity = int(formula[i_start: i] or 1)
            stack[-1][name] += multiplicity
    print(stack)

    return "".join(name + (str(stack[-1][name]) if stack[-1][name] > 1 else '')
                   for name in sorted(stack[-1]))

"""
909. Snakes and Ladders

On an N x N board, the numbers from 1 to N*N are written boustrophedonically starting from the bottom left of the board, and alternating direction each row.  For example, for a 6 x 6 board, the numbers are written as follows:

You start on square 1 of the board (which is always in the last row and first column).  Each move, starting from 
 x, consists of the following:

You choose a destination square S with number x+1, x+2, x+3, x+4, x+5, or x+6, provided this number is <= N*N.
(This choice simulates the result of a standard 6-sided die roll: ie., there are always at most 6 destinations, regardless of the size of the board.)
If S has a snake or ladder, you move to the destination of that snake or ladder.  Otherwise, you move to S.
A board square on row r and column c has a "snake or ladder" if board[r][c] != -1.  The destination of that snake or ladder is board[r][c].

Note that you only take a snake or ladder at most once per move: if the destination to a snake or ladder is the start of another snake or ladder, you do not continue moving.  (For example, if the board is `[[4,-1],[-1,3]]`, and on the first move your destination square is `2`, then you finish your first move at `3`, because you do not continue moving to `4`.)

Return the least number of moves required to reach square N*N.  If it is not possible, return -1.

Example 1:

Input: [
[-1,-1,-1,-1,-1,-1],
[-1,-1,-1,-1,-1,-1],
[-1,-1,-1,-1,-1,-1],
[-1,35,-1,-1,13,-1],
[-1,-1,-1,-1,-1,-1],
[-1,15,-1,-1,-1,-1]]
Output: 4
Explanation: 
At the beginning, you start at square 1 [at row 5, column 0].
You decide to move to square 2, and must take the ladder to square 15.
You then decide to move to square 17 (row 3, column 5), and must take the snake to square 13.
You then decide to move to square 14, and must take the ladder to square 35.
You then decide to move to square 36, ending the game.
It can be shown that you need at least 4 moves to reach the N*N-th square, so the answer is 4.
"""

def snakesAndLadders(board):
    r, c = len(board), len(board[0])
    q = deque([1])
    steps = 0
    visited = set([1])
    while q:
        for _ in range(len(q)):
            cur = q.popleft()
            if cur == r * c:
                return steps
            for offset in range(1, 7):
                neighbor = min(cur + offset, r * c)
                y = r - (neighbor - 1) // c - 1
                x = (neighbor - 1) % c if (neighbor - 1) // c % 2 == 0 else c - (neighbor - 1) % c - 1
                if board[y][x] != -1:
                    neighbor = board[y][x]
                if neighbor not in visited:
                    visited.add(neighbor)
                    q.append(neighbor)
        steps += 1
    return -1

"""
1155. Number of Dice Rolls With Target Sum

You have d dice, and each die has f faces numbered 1, 2, ..., f.

Return the number of possible ways (out of fd total ways) modulo 10^9 + 7 to roll the dice so the sum of the face up numbers equals target.


Example 1:

Input: d = 1, f = 6, target = 3
Output: 1
Explanation: 
You throw one die with 6 faces.  There is only one way to get a sum of 3.
Example 2:

Input: d = 2, f = 6, target = 7
Output: 6
Explanation: 
You throw two dice, each with 6 faces.  There are 6 ways to get a sum of 7:
1+6, 2+5, 3+4, 4+3, 5+2, 6+1.
"""
def numRollsToTarget(d,f,target):
    def helper(h,d,target):
        if target <= 0 or target > d*f:
            return 0
        if d == 1:
            return 1
        if (d,target) in h:
            return h[(d,target)]
        
        res = 0
        for i in range(1,f+1):
            res += helper(h,d-1,target-i)
        h[(d,target)] = res
        return h[(d,target)]
    
    h = {}
    return helper(h,d,target) % (10**9 + 7)

"""
17. Letter Combinations of a Phone Number

Given a string containing digits from 2-9 inclusive, return all possible letter combinations that the number could represent. Return the answer in any order.

A mapping of digit to letters (just like on the telephone buttons) is given below. Note that 1 does not map to any letters.

Example 1:

Input: digits = "23"
Output: ["ad","ae","af","bd","be","bf","cd","ce","cf"]
"""
def letterCombinations(digits):
    res = []
    if not digits:
        return res
    
    
    N = len(digits)
    mapping = {2:'abc', 3:'def', 4:'ghi', 5:'jkl', 6:'mno', 7:'pqrs', 8:'tuv', 9: 'wxyz'}
    
    def dfs(idx, word):
        if idx == N:
            res.append(word)
            return
        
        for ch in mapping[int(digits[idx])]:
            dfs(idx+1, word + ch)
            
    dfs(0, "")
    return res

"""
937. Reorder Data in Log Files

You have an array of logs.  Each log is a space delimited string of words.

For each log, the first word in each log is an alphanumeric identifier.  Then, either:

Each word after the identifier will consist only of lowercase letters, or;
Each word after the identifier will consist only of digits.
We will call these two varieties of logs letter-logs and digit-logs.  It is guaranteed that each log has at least one word after its identifier.

Reorder the logs so that all of the letter-logs come before any digit-log.  The letter-logs are ordered lexicographically ignoring identifier, with the identifier used in case of ties.  The digit-logs should be put in their original order.

Return the final order of the logs.

Example 1:

Input: logs = ["dig1 8 1 5 1","let1 art can","dig2 3 6","let2 own kit dig","let3 art zero"]
Output: ["let1 art can","let3 art zero","let2 own kit dig","dig1 8 1 5 1","dig2 3 6"]
"""
def reorderLogFiles(logs):
    res = []
    temp = []
    for log in logs:
        if log.split()[1].isdigit():
            temp.append(log)
        else:
            res.append(log)
        
    res.sort(key=lambda x: [' '.join(x.split()[1:]),x.split()[0]])
    return res+temp

"""
316. Remove Duplicate Letters
Given a string s, remove duplicate letters so that every letter appears once and only once. You must make sure your result is the smallest in lexicographical order among all possible results.


1081. Smallest Subsequence of Distinct Characters
Return the lexicographically smallest subsequence of s that contains all the distinct characters of s exactly once.

Note: This question is the same as 316: https://leetcode.com/problems/remove-duplicate-letters/

Example 1:

Input: s = "bcabc"
Output: "abc"

Example 2:

Input: s = "cbacdcbc"
Output: "acdb"
"""

def removeDuplicateLetters(s):
    stack = []
    seen = set()
    occdict = {c:i for i,c in enumerate(s)}
    print(occdict)
    for i,c in enumerate(s):
        if c not in seen:
            while stack and c < stack[-1] and i < occdict[stack[-1]]:
                seen.discard(stack.pop())
            seen.add(c)
            stack.append(c)
    
    return "".join(stack)

"""
1258. Synonymous Sentences

Given a list of pairs of equivalent words synonyms and a sentence text, 
Return all possible synonymous sentences sorted lexicographically.
 

Example 1:

Input:
synonyms = [["happy","joy"],["sad","sorrow"],["joy","cheerful"]],
text = "I am happy today but was sad yesterday"
Output:
["I am cheerful today but was sad yesterday",
"I am cheerful today but was sorrow yesterday",
"I am happy today but was sad yesterday",
"I am happy today but was sorrow yesterday",
"I am joy today but was sad yesterday",
"I am joy today but was sorrow yesterday"]
"""

def generateSentences(self, synonyms: List[List[str]], text: str) -> List[str]:
    adjList = defaultdict(set)
    for syn in synonyms:
        adjList[syn[0]].add(syn[1])
        adjList[syn[1]].add(syn[0])
        
    
    res = set()
    q = deque()
    q.append(text)
    
    while q:
        curr_text = q.popleft()
        res.add(curr_text)
        words = curr_text.split()
        for idx,word in enumerate(words):
            for k,v in adjList.items():
                if word == k:
                    for newword in v:
                        new_text = words[:idx] + [newword] + words[idx+1:]
                        new_text = " ".join(new_text)
                        if new_text not in res:
                            q.append(new_text)
    
    return sorted(res)


"""
Sum of Root To Leaf Binary Numbers
You are given the root of a binary tree where each node has a value 0 or 1. 
Each root-to-leaf path represents a binary number starting with the most significant bit.  
For example, if the path is 0 -> 1 -> 1 -> 0 -> 1, then this could represent 01101 in binary, which is 13.

For all leaves in the tree, consider the numbers represented by the path from the root to that leaf.

Return the sum of these numbers. The answer is guaranteed to fit in a 32-bits integer.

Example 1:

Input: root = [1,0,1,0,1,0,1]
Output: 22
Explanation: (100) + (101) + (110) + (111) = 4 + 5 + 6 + 7 = 22
"""

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def sumRootToLeaf(self, root: TreeNode) -> int:
        root_to_leaf = 0
        stack = [(root,0)]        
        while stack:
            root,curr_num = stack.pop()
            if root is not None:
                curr_num = (curr_num << 1) | root.val
                if root.left is None and root.right is None:
                    root_to_leaf += curr_num
                else:
                    stack.append((root.left,curr_num))
                    stack.append((root.right,curr_num))
        
        return root_to_leaf
    
"""
59. Spiral Matrix II

Given a positive integer n, generate an n x n matrix filled with elements from 1 to n2 in spiral order.

Example 1:

Input: n = 3
Output: [[1,2,3],[8,9,4],[7,6,5]]
"""

def generateMatrix(self, n: int) -> List[List[int]]:
    matrix = [[0 for j in range(n)] for i in range(n)]
    count = 1
    top = 0
    bottom = len(matrix)-1
    left = 0
    right = len(matrix[0])-1
    direc = 0
    while (top<=bottom and left<=right):
        if direc == 0:
            for i in range(left,right+1):
                matrix[top][i] = count
                count += 1
            top += 1
        elif direc == 1:
            for i in range(top,bottom+1):
                matrix[i][right] = count
                count += 1
            right -= 1
        elif direc == 2:
            for i in range(right,left-1,-1):
                matrix[bottom][i] = count
                count += 1
            
            bottom -= 1
        
        elif direc == 3:
            for i in range(bottom,top-1,-1):
                matrix[i][left] = count 
                count += 1
            left += 1
        
        direc = (direc+1) % 4
    
    return matrix
