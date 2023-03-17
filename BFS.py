# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 19:50:24 2020

@author: prchandr
"""

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

# return the path from the Origin
def bfs(origin):
    outp_str = ""
    queue = []
    queue.append(origin)
    visited = set()
    visited.append(origin)
    
    while(queue):
      curr_vertex = queue.pop(0)
      outp_str += curr_vertex.id
      for neighbor in curr_vertex.edges:
        if neighbor not in visited:
          queue.append(neighbor)
          visited.append(neighbor)
    return outp_str
"""
675. Cut Off Trees for Golf Event

You are asked to cut off trees in a forest for a golf event. The forest is represented as a non-negative 2D map, in this map:

0 represents the obstacle can't be reached.
1 represents the ground can be walked through.
The place with number bigger than 1 represents a tree can be walked through, and this positive number represents the tree's height.
In one step you can walk in any of the four directions top, bottom, left and right also when standing in a point which is a tree you can decide whether or not to cut off the tree.

You are asked to cut off all the trees in this forest in the order of tree's height - always cut off the tree with lowest height first. And after cutting, the original place has the tree will become a grass (value 1).

You will start from the point (0, 0) and you should output the minimum steps you need to walk to cut off all the trees. If you can't cut off all the trees, output -1 in that situation.

You are guaranteed that no two trees have the same height and there is at least one tree needs to be cut off.

Example 1:

Input: 
[
 [1,2,3],
 [0,0,4],
 [7,6,5]
]
Output: 6
 
"""

def cutOffTree(forest):
    treeLocation = []
    rows, cols = len(forest) , len(forest[0])
    
    for row in range(rows):
        for col in range(cols):
            if forest[row][col] > 1:
                treeLocation.append((forest[row][col], (row,col)))
    def bfs(root, target):
        if forest[root[0][root[1]]] == target:
            return 0
        q = deque()
        q.append(root,0)
        visited =set()
        visited.add(root)
        
        while q:
            currL, level = q.popleft()
            r,c = currL
            for row, col in [(r+1,c),(r-1,c), (r,c+1),(r,c-1)]:
                if 0<=row<=rows and 0<=col<cols and (row,col) not in visited:
                    visited.add((row,col))
                    if forest[row][col] == target:
                        return level+1
                    
                    if forest[row][col] > 0:
                        q.append((row,col),level+1)
        return -1
    
    
    startingPos = (0,0)
    ans = 0
    for loc in sorted(treeLocation):
        temp = bfs(startingPos, loc[0])
        if temp!=-1:
            ans += temp
            startingPos = loc[1]
    
    return ans
    
            
"""
126. Word Ladder II

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
  ["hit","hot","lot","log","cog"]
]

"""          

from collections import defaultdict, Counter,deque
def findLadder(begin,end,words):
    dist = defaultdict(list)
    adjMatrix = defaultdict(set)
    wordLen = len(words)
    
    for word in words:
        for i in range(wordLen):
            key = word[:i]+'$'+word[i+1:]
            dist[key].append(word)
    print(dist)
    queue = deque()
    queue.append([begin,1])
    seen = Counter()
    seen[begin] = 0
    
    while queue:
        curr, level = queue.popleft()
        if curr == end:
            print("Length", level+1)
            break
        for i in range(wordLen):
            key = curr[:i]+'$'+curr[i+1:]
            for neig in dist[key]:
                if neig not in seen:
                    seen[neig] = seen[curr]+1
                    queue.append([neig, level+1])
                
                if neig in seen and seen[neig] == seen[curr]+1:
                    adjMatrix[curr].add(neig)
    print(adjMatrix)
    
    queue = deque()
    queue.append([begin,[]])
    result = []
    while queue:
        
        curr, path = queue.popleft()
        if curr == end:
            result.append(path+[end])
            
        for neig in adjMatrix[curr]:
            queue.append([neig, path+[curr]])
    
    return result

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
854. K-Similar Strings

Strings A and B are K-similar (for some non-negative integer K) if we can swap the positions of two letters in A exactly K times so that the resulting string equals B.

Given two anagrams A and B, return the smallest K for which A and B are K-similar.

Example 1:

Input: A = "ab", B = "ba"
Output: 1
Example 2:

Input: A = "abc", B = "bca"
Output: 2
Example 3:

Input: A = "abac", B = "baca"
Output: 2
Example 4:

Input: A = "aabc", B = "abca"
Output: 2
"""
def kSimilarity(A,B):
    
    
    if A==B:
        return 0
    
    def swap(s,a,b):
        ca, cb = s[a], s[b]
        s = f'{s[:a]}{cb}{s[a+1:]}'
        s = f'{s[:b]}{ca}{s[b+1:]}'
        return s
    
    q = deque()
    q.append((A,0,0))
    done = set()
    done.add(A)
    
    while q:
        s,i,swaps = q.popleft()
        
        assert i < len(s)-1
        
        while s[i] == B[i]:
            q.append((s,i+1,swaps))
            i += 1
        
        for j in range(i+1, len(s)):
            if s[j]==B[i]:
                new_word = swap(s,i,j)
            
            if new_word == B:
                return swaps+1
            if new_word not in done:
                q.append((new_word,i+1,swaps+1))
                done.add(new_word)
    
    print('not expected')