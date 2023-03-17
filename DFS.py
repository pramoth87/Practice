# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 19:51:04 2020

@author: prchandr
"""

"""
Remember the story of Little Match Girl? By now, you know exactly what matchsticks the little match girl has, please find out a way you can make one square by using up all those matchsticks. You should not break any stick, but you can link them up, and each matchstick must be used exactly one time.

Your input will be several matchsticks the girl has, represented with their stick length. Your output will either be true or false, to represent whether you could make one square using all the matchsticks the little match girl has.

Example 1:
Input: [1,1,2,2,2]
Output: true

Explanation: You can form a square with length 2, one side of the square came two sticks with length 1.
"""
from collections import defaultdict, Counter


def makeSquare(nums):
    if not nums:
        return False
    L = len(nums)
    perimeter = sum(nums)
    possible_Sides = perimeter//4
    print(possible_Sides)
    if possible_Sides*4 != perimeter:
        return False
    nums.sort(reverse=True)
    print(nums)
    sums = [0 for _ in range(4)]
    def DFS(index):
        if index==L:
            return sums[0]==sums[1]==sums[2]==possible_Sides # If 3 equal sides were formed, 4th will be the same as these three and answer should be True in that case.
        
        for i in range(4):
            if sums[i]+nums[index] <= possible_Sides:
                sums[i] += nums[index]
                if DFS(index +1):
                    return True
                sums[i] -= nums[index]
        return False
    return DFS(0)


"""
Given an m x n matrix of non-negative integers representing the height of each unit cell in a continent, the "Pacific ocean" touches the left and top edges of the matrix and the "Atlantic ocean" touches the right and bottom edges.

Water can only flow in four directions (up, down, left, or right) from a cell to another one with height equal or lower.

Find the list of grid coordinates where water can flow to both the Pacific and Atlantic ocean.
"""
class Solution:
    
    def pacificAtlantic(self,matrix):
        if matrix ==[]:
            return []
        r = len(matrix)
        c = len(matrix[0])
        result = []
        self.direction = [(1,0),(-1,0),(0,1),(0,-1)]
        pacific = [[0 for i in range(c)]for j in range(r)]
        atlantic = [[0 for i in range(c)]for j in range(r)]
        
        for i in range(r):
            self.dfsOcean(matrix,i,0,pacific,r,c)
            self.dfsOcean(matrix,i,c-1,atlantic,r,c)
        for j in range(c):
            self.dfsOcean(matrix,0,j,pacific,r,c)
            self.dfsOcean(matrix,r-1,j,atlantic,r,c)
        for i in range(r):
            for j in range(c):
                if pacific[i][j] and atlantic[i][j]:
                    result.append([i,j])
        return result
    
    def dfsOcean(self,matrix,i,j,visited,r,c):
        visited[i][j] = 1
        for direct in self.direction:
            x,y = i + direct[0], j + direct[1]
            if x<0 or x>r-1 or y <0 or y>c-1 or visited[x][y] or matrix[x][y]<matrix[i][j]:
                continue
            self.dfsOcean(matrix,x,y,visited,r,c)

pa = Solution()
print(pa.pacificAtlantic([[1,2,2,3,5],[3,2,3,4,4],[2,4,5,3,1],[6,7,1,4,5],[5,1,1,2,4]]))



"""
The thief has found himself a new place for his thievery again. There is only one entrance to this area, called the "root." Besides the root, each house has one and only one parent house. After a tour, the smart thief realized that "all houses in this place forms a binary tree". It will automatically contact the police if two directly-linked houses were broken into on the same night.

Determine the maximum amount of money the thief can rob tonight without alerting the police.

Example 1:

Input: [3,2,3,null,3,null,1]

     3
    / \
   2   3
    \   \ 
     3   1

Output: 7 
Explanation: Maximum amount of money the thief can rob = 3 + 3 + 1 = 7.
"""

def rob(root):
    def roberMind(root):
        if not root:
            return (0, 0)
        robbedLeft, skippedLeft = roberMind(root.left)
        robbedRight, skippedRight = roberMind(root.right)
        return (root.key+skippedLeft+skippedRight, max(robbedLeft+robbedRight, skippedLeft+skippedRight, skippedLeft+robbedRight, robbedLeft+skippedRight))
    return max(roberMind(root))


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

def depthSum(nestedList):
    if not nestedList:
        return 0
    stack = [(l,1) for l in nestedList]
    print(stack)
    total = 0
    while stack:
        l, index = stack.pop()
        print("Index",l,index)
        if isinstance(l,int):
            print("Inside",l,index)
            total += l * index
        else:
            for item in l:
                print("Item:",item,index)
                stack.append((item, index+1))
    return total

def nestedListSum(nestedList):
    myDict = {}
    total = 0
    for n in nestedList:
        stack = [(n,1)]
        while stack:
            curr, level = stack.pop()    
            if isinstance(curr,int):
                if level in myDict:
                    myDict[level] += curr
                else:
                    myDict[level] = curr
            else:
                for l in curr:
                    stack.append((l,level+1))
    print(myDict.keys(),myDict)
    max_level = max(myDict.keys()) if myDict else 0
    for k,v in myDict.items():
        print("insite For",k,v)
        mul = max_level - k +1
        total += mul*v
    return total
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

Surrounded regions shouldn’t be on the border, 
which means that any 'O' on the border of the board are not flipped to 'X'.
 Any 'O' that is not on the border and it is not connected to an 'O' on the border will be flipped to 'X'. 
 Two cells are connected if they are adjacent cells connected horizontally or vertically.
"""
class Solution(object):
    def solve(self, board):
        """
        :type board: List[List[str]]
        :rtype: None Do not return anything, modify board in-place instead.
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
        print("borders:" ,borders)

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

s = Solution()
board = [['X','X','X','X'],['X','O','O','X'],['X','X','O','X'],['X','O','X','X']]
print(s.solve(board))

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



class Node:
    def __init__(self, data):
        self.key = data
        self.left = None
        self.right = None
        self.count = 1

    
    def insert(node,key):
        if node is None:
            k = Node(key)
            return k       
        if key == node.key:
            node.count +=1
            return node
        if key <= node.key:
            node.left = Node.insert(node.left,key)
        else:
            node.right = Node.insert(node.right,key)
        return node
    
    def inorder(root):
        if root != None:
            Node.inorder(root.left)
            print(root.key, "(", root.count, ")", end = "")
            Node.inorder(root.right)
    
#Finiding the Min Diff in the BST. Alternative Solution is posted on the BST file
    def minDiffInBST(root):    
        val = []
        def dfs(node):
            if node:
                val.append(node.key)
                dfs(node.left)
                dfs(node.right)
        dfs(root)
        val.sort()
        return min(val[i+1] - val[i] for i in range(len(val)-1))


# =============================================================================
# #"""
# #Given a binary tree, determine if it is a valid binary search tree (BST).
# #
# #Assume a BST is defined as follows:
# #
# #The left subtree of a node contains only nodes with keys less than the node's key.
# #The right subtree of a node contains only nodes with keys greater than the node's key.
# #Both the left and right subtrees must also be binary search trees.
# # 
# #
# #Example 1:
# #
# #    2
# #   / \
# #  1   3
# #
# #Input: [2,1,3]
# #Output: true
# =============================================================================
    def isValidBST(root):
        if not root:
            return True
        stack = [(root, float('-inf'),float('inf'))]
        while stack:
            root, lower, upper = stack.pop()
            if not root:
                continue
            print(root.key, lower, upper)
            val = root.key
            if val <= lower or val >= upper:
                print(val,lower,upper)
                return False
            stack.append((root.right, val, upper))
            stack.append((root.left, lower, val))
        return True
# Finding if both the tree is Identical, wont work now as there needs to be another tree
    def isSameTree(p,q):
        if not p and not q:
            return True
        if not p or not q:
            return False
        if p.key != q.key:
            return False
        return Node.isSameTree(p.left,q.left) and Node.isSameTree(p.right,q.right)
    
    def isSymmetry(root):
        return Node.helper(root,root)
    
    def helper(l1,l2):
        if not l1 and not l2:
            return True
        if not l1 or not l2:
            return False
        if l1.key != l2.key:
            return False
        return helper(l1.left,l2.right) and helper(l1.right,l2.left)
    
    def maxDepth(root):
        stack = []
        if root is not None:
            stack.append((1,root))
        depth = 0
        while stack != []:
            current_depth,root = stack.pop()
            if root is not None:
                depth = max(depth, current_depth)
                stack.append((current_depth+1 , root.left))
                stack.append((current_depth+1, root.right))
        return depth
    
    def sumPath(root, addsum):
        pathList = []
        Node.recurssiveTreeSum(root,addsum,[],pathList)
        return pathList
    
    def recurssiveTreeSum(root,remainingSum,pathNode,pathList):
        if not root:
            return
        pathNode.append(root.key)
        if root.key ==  remainingSum and not root.left and not root.right:
            pathList.append(list(pathNode))
        else:
            Node.recurssiveTreeSum(root.left, remainingSum-root.key, pathNode, pathList)
            Node.recurssiveTreeSum(root.right, remainingSum-root.key, pathNode, pathList)
        pathNode.pop()

    def maxPathSum(root):
        def solution(root):
            nonlocal maxSum
            right, left = 0,0
            if root.left is not None: 
                left = solution(root.left)
            if root.right is not None:
                right = solution(root.right)
            localMax = max(max(left, right) + root.key, root.key)
            maxSum = max(maxSum, left + right + root.key, localMax, root.key)
            print(maxSum,localMax)
            return localMax
        
        maxSum = float("-inf")
        solution(root)
        return maxSum
    
    def sumNumbers(root):
        
        def solution(root,conNodes):
            nonlocal roottoleaf
            if root:
                conNodes = int(f"{conNodes}{root.key}")
                print("Values", roottoleaf,conNodes)
                if not root.left or not root.right:
                    roottoleaf += conNodes
                solution(root.left,conNodes)
                solution(root.right,conNodes)
        roottoleaf = 0
        solution(root, 0)
        return roottoleaf
    
    def stackSumNumbers(root):
        roottoleaf = 0
        stack = [(root,0)]
        while stack:
            root, conNodes = stack.pop()
            if root is not None:
                conNodes = int(f"{conNodes}{root.key}") #concatenate two numbers
                print("Values", roottoleaf,conNodes)
                if root.left is None and root.right is None:
                    roottoleaf += conNodes
                stack.append((root.left,conNodes))
                stack.append((root.right,conNodes))
        return roottoleaf
    
    def rob(root):
        def roberMind(root):
            if not root:
                return (0, 0)
            robbedLeft, skippedLeft = roberMind(root.left)
            robbedRight, skippedRight = roberMind(root.right)
            print("Values",robbedLeft,skippedLeft,robbedRight,skippedRight )
            return (root.key+skippedLeft+skippedRight, max(robbedLeft+robbedRight, skippedLeft+skippedRight, skippedLeft+robbedRight, robbedLeft+skippedRight))
        return max(roberMind(root))
    
    def FindLeaves(root):
        res = []
        d = defaultdict(list)
        def dfs(node):
            if node == None:
                return 0
            leftHeight = dfs(node.left)
            rightHeight = dfs(node.right)
            currHeight = max(leftHeight, rightHeight)+1
            print("Heigh",leftHeight,rightHeight,currHeight)
            d[currHeight].append(node.key)
            print(d)
            return currHeight
        dfs(root)
        for v in d.values():
            res.append(v)
        return res
    
    def bsttogst(root):
        sum = 0
        def dfs(node):
            nonlocal sum
            if not node:
                return
            dfs(node.right)
            node.key += sum
            sum = node.key
            dfs(node.left)
            return
        dfs(root)
        return root
    
    def findDuplicates(root):
        count = Counter()
        ans = []
        def collectDFS(root):
            if not root:
                return "#"
            serial = "{},{},{}".format(root.key, collectDFS(root.left),collectDFS(root.right))
            count[serial] += 1
            print(count)
            if count[serial] == 2:
                ans.sppend(root)
            return serial
        
        collectDFS(root)
        return ans
    
if __name__ == '__main__': 
      
    # Let us create following BST  
    # 12(3)  
    # / \  
    # 10(2) 20(1)  
    # / \  
    # 9(1) 11(1)
    root = None
    root = Node.insert(root, 12)  
    root = Node.insert(root, 10)  
    root = Node.insert(root, 20)  
    root = Node.insert(root, 9)  
    root = Node.insert(root, 11)  
    root = Node.insert(root, 10)  
    root = Node.insert(root, 12)  
    root = Node.insert(root, 12) 
    print("Inorder traversal of the given tree")  
    Node.inorder(root)  
    print()
    print("MinDiff in BST",Node.minDiffInBST(root))
    print()
    print("Is the Tree is a valid BST: ", Node.isValidBST(root))
    print()
    print("Whether two tree are identical: ", Node.isSameTree(root,root))
    print("Max Depth: ",Node.maxDepth(root))
    print()
    print("Path Sum: ", Node.sumPath(root,31))
    print()
    print("Max Sum Path: ", Node.maxPathSum(root))
    print()
    print("The Path Sum:",Node.stackSumNumbers(root))
    print("Robbed Maximum:", Node.rob(root))
    print("Find the leaves:", Node.FindLeaves(root))
    print("BST to greater sum tree:", Node.bsttogst(root))
    print("FindDuplicate: ", Node.findDuplicates(root))
    Node.inorder(root)  

"""
638. Shopping Offers
In LeetCode Store, there are some kinds of items to sell. Each item has a price.

However, there are some special offers, and a special offer consists of one or more different kinds of items with a sale price.

You are given the each item's price, a set of special offers, and the number we need to buy for each item. The job is to output the lowest price you have to pay for exactly certain items as given, where you could make optimal use of the special offers.

Each special offer is represented in the form of an array, the last number represents the price you need to pay for this special offer, other numbers represents how many specific items you could get if you buy this offer.

You could use any of special offers as many times as you want.

Example 1:
Input: [2,5], [[3,0,5],[1,2,10]], [3,2]
Output: 14
Explanation: 
There are two kinds of items, A and B. Their prices are $2 and $5 respectively. 
In special offer 1, you can pay $5 for 3A and 0B
In special offer 2, you can pay $10 for 1A and 2B. 
You need to buy 3A and 2B, so you may pay $10 for 1A and 2B (special offer #2), and $4 for 2A.
"""   

def shoppingOffers(price,special,needs):
    def dfs(needs, memo, price, special):
        key = tuple(needs)
        print(key)
        if not key in memo:
            memo[key] = sum(needs[i]*price[i] for i in range(len(needs)))
            for offer in special:
                new_needs = [needs[i] - offer[i] for i in range(len(needs)) if needs[i] >= offer[i]]
                if len(new_needs) == len(needs):
                    memo[key] = min(memo[key], dfs(new_needs, memo, price, special) + offer[-1])
                    print(memo)
        return memo[key]
    
    memo = {}
    return dfs(needs, memo, price, special)


# FInd all paths using DFS
def find_all_paths(origin,destination):
  visited = set()
  path = []
  finalPath = []
  return DFS(origin,destination,visited,path,finalPath)

def DFS(u,v,visited,path,finalPath):
  visited.add(u.id)
  path.append(u.id)
  if u.id == v:
    finalPath.append(path)
  else:
    for neighbor in u.edges:
      if neighbor.id not in visited:
        DFS(neighbor,v,visited,path, )
  return finalPath


def containVirus(grid):
        m=len(grid)
        n=len(grid[0])
        dirs=[(0,1),(1,0),(0,-1),(-1,0)]        
        def dfs(x,y,sn):
            if grid[x][y]==1:
                v[sn].add((x,y))
                seen.add((x,y))
                for dx,dy in dirs:
                    nx=x+dx
                    ny=y+dy
                    if 0<=nx<m and 0<=ny<n:                             
                        if grid[nx][ny]==0:
                            f[sn].add((nx,ny))
                            w[sn]+=1
                        if grid[nx][ny]==1 and (nx,ny) not in v[sn]:
                            v[sn].add((nx,ny))
                            dfs(nx,ny,sn)                                  
        res=0
        while True:
            v=defaultdict(set)  # infected area
            f=defaultdict(set)  # potential risk area
            w=defaultdict(int)  # walls in need
            seen=set()
                        
            for i in range(m):
                for j in range(n):
                    if grid[i][j]==1 and (i,j) not in seen:
                        dfs(i,j,(i,j))
             
            if len(f)==0:   
                break

            nnfrontiers = sorted(f,key=lambda x:len(f[x]),reverse=True)
            print(w,f,nnfrontiers)
            if len(nnfrontiers)>0:
                nnfrontier = nnfrontiers[0]                
                res+=w[nnfrontier]

                for x,y in v[nnfrontier]:   # get isolated
                    grid[x][y]=2
                
                if len(nnfrontiers)>1:      # spread the frontiers
                    for i in range(1,len(nnfrontiers)):
                        nnfrontier2 = nnfrontiers[i]
                        for x,y in f[nnfrontier2]:
                            grid[x][y]=1
        return res


"""
685. Redundant Connection II


In this problem, a rooted tree is a directed graph such that, 
there is exactly one node (the root) for which all other nodes are descendants of this node, 
plus every node has exactly one parent, except for the root node which has no parents.

The given input is a directed graph that started as a rooted tree with N nodes (with distinct values 1, 2, ..., N), with one additional directed edge added. The added edge has two different vertices chosen from 1 to N, and was not an edge that already existed.

The resulting graph is given as a 2D-array of edges. Each element of edges is a pair [u, v] that represents a directed edge connecting nodes u and v, where u is a parent of child v.

Return an edge that can be removed so that the resulting graph is a rooted tree of N nodes. If there are multiple answers, return the answer that occurs last in the given 2D-array.
"""
def reduntantConnection(edges):
    N = len(edges)
    parent = {}
    candidates = []
    for u, v in edges:
        if v in parent:
            candidates.append((parent[v], v))
            candidates.append((u, v))
        else:
            parent[v] = u
    print("List and Dict",candidates,parent)
    def orbit(node):  #DFS
        seen = set()
        while node in parent and node not in seen:
            seen.add(node)
            node = parent[node]
            print("While",node)
        print(node,seen)
        return node, seen
    
    root = orbit(1)[0]
    print("root",root)
    
    if not candidates:
        print(candidates)
        cycle = orbit(root)[1]
        for u, v in edges:
            if u in cycle and v in cycle:
                ans = u, v
        return ans
    
    children = collections.defaultdict(list)
    for v in parent:
        print(v,parent[v])
        children[parent[v]].append(v)
    print(children)
    seen = [True] + [False] * N
    print(seen)
    stack = [root]
    while stack:
        print(stack)
        node = stack.pop()
        if not seen[node]:
            seen[node] = True
            stack.extend(children[node])
    print(seen,all(seen),candidates[True])
    return candidates[all(seen)]

def longestPath(edges):
    paths = defaultdict(list)
    finalPath = []
    visited = set()
    for u,v in edges:
        paths[u].append(v)
    print(paths)
    return dfs(edges[0][0],edges[0][1],visited, paths, finalPath)

def dfs(u,v, visited, paths,finalPath):
    visited.add(u)
    finalPath.append(u)
    for neighbours in paths[u]:
        if neighbours not in visited:
            dfs(neighbours,v,visited,paths,finalPath)
    return finalPath
        

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
Given the root of a binary tree, return the sum of values of its deepest leaves.
 

Example 1:


Input: root = [1,2,3,4,5,null,6,7,null,null,null,null,8]
Output: 15
"""

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def deepestLeavesSum(self, root: TreeNode) -> int:
        depth = finalSum = 0
        stack = [(root,0)]
        while stack:
            node, curr_depth = stack.pop()
            if node.left is None and node.right is None:
                if depth < curr_depth:
                    finalSum = node.val
                    depth = curr_depth 
                elif depth == curr_depth:
                    finalSum += node.val
            else:
                if node.right:
                    stack.append((node.right,curr_depth+1))
                if node.left:
                    stack.append((node.left,curr_depth+1))
        
        return finalSum
    
"""

You have n gardens, labeled from 1 to n, and an array paths where paths[i] = [xi, yi] describes a bidirectional path between garden xi to garden yi. In each garden, you want to plant one of 4 types of flowers.

All gardens have at most 3 paths coming into or leaving it.

Your task is to choose a flower type for each garden such that, for any two gardens connected by a path, they have different types of flowers.

Return any such a choice as an array answer, where answer[i] is the type of flower planted in the (i+1)th garden. The flower types are denoted 1, 2, 3, or 4. It is guaranteed an answer exists.

 

Example 1:

Input: n = 3, paths = [[1,2],[2,3],[3,1]]
Output: [1,2,3]
Explanation:
Gardens 1 and 2 have different types.
Gardens 2 and 3 have different types.
Gardens 3 and 1 have different types.
Hence, [1,2,3] is a valid answer. Other valid answers include [1,2,4], [1,4,2], and [3,2,1].

"""
def gardenNoAdj(n, paths):
    color = [0] * (n+1)
    graph = defaultdict(list)
    for u,v in paths:
        graph[u].append(v)
        graph[v].append(u)
    
    
    def dfs(node):
        if color[node] > 0:
            return True
        useless = set(color[v] for v in graph[node] if color[v]>0)
        for i in range(1,5):
            if i not in useless:
                color[node] = i
                isok = True
                for v in graph[node]:
                    if not dfs(v):
                        isok = False
                        break
                if isok:
                    return True
                color[node] = 0
    for u in range(1, n+1):
        dfs(u)
    
    return color[1:]