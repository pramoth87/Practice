# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 17:07:02 2019

@author: prchandr
"""
import collections
import sys
from itertools import combinations

#Find the largest Number with Minimum Frequency
def largeButMinFrequency(arr):
    n = len(arr)
    myDict = {}
    for i in arr:
        if i not in myDict:
            myDict[i] = 1
        else:
            myDict[i] += 1
    print(myDict)
    sortDict = dict(sorted(myDict.items()))
    print(sortDict)
    keyList = list(sortDict.keys())
    print(keyList)
    if sortDict[keyList[-1]]<sortDict[keyList[-2]]:
        print(keyList[-1])
    minV =  min(myDict.keys(), key=(lambda k: myDict[k]))
    maxV =  myDict[max(myDict.keys(), key=(lambda k: myDict[k]))]
    res = 0
    print(minV,maxV)
    for i in range(n):
        if arr[i] > minV:
            minV = arr[i]
            if(myDict[arr[i]] <= maxV):
                maxV = myDict[arr[i]]
                res = arr[i]
    return res

# Find the A exp B and Mod C (apowb)%c
def modExponential(A):
    a = A.split()
    power = pow(int(a[0]),int(a[1]))
    print(power)
    mod = power % int(a[2])
    return mod
# finding power without using Math Libraries
#    powerdummy = 1
#    i = 1
#    while(i <= int(a[1])):
#        powerdummy = powerdummy * int(a[0])
#        i = i + 1
#    print(powerdummy)
# Same as above problem
def power(x, y, p) : 
    res = 1     # Initialize result 
  
    # Update x if it is more 
    # than or equal to p 
    x = x % p  
  
    while (y > 0) : 
          
        # If y is odd, multiply 
        # x with result 
        if ((y & 1) == 1) : 
            res = (res * x) % p 
  
        # y must be even now 
        y = y >> 1      # y = y/2 
        x = (x * x) % p 
          
    return res

# Count of strings that can be formed using a, b and c under given constraints. with b one time and C is allowed two times minimum
    
def countStr(n): 
    return (1 + (n * 2) + (n * ((n * n) - 1) // 2)) 

##########################################
# Sum of root to leaf refer in BST section
##########################################


# Word Boggle
    
#Overlapping Intervals
    
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

##Alien Dictionary
def main():
    words = ("baa", "abcd", "abca", "cab", "cad")
    chars = collections.OrderedDict()
    nodes_map = collections.defaultdict(list)
    for i, word in enumerate(words[:-1]):
        nxt = words[i+1]
        for a, b in zip(word, nxt):
            if a != b:
                nodes_map[a] += [b]
                break
#    for i in topological_sort(nodes_map, chars):
#         print (i)

if __name__ == '__main__':
    main()
    
##Form a Palindrome
    
def minPalindromeFormation(string,l,h):
    print(l,h)
    if (l>h):
        return sys.maxsize
    if (l == h):
        return 0
    if (l == h-1):
        if (string[l]==string[h]):
            return 0 
        else: 
            return 1
    if string[l] == string[h]:
        return minPalindromeFormation(string, l+1, h-1)
    else:
        return (min(minPalindromeFormation(string,l,h-1), minPalindromeFormation(string,l+1,h)) +1)
    
def largestWord(dic,strg):
    count = tempCount = 0
    str1 = None
    for word in dic:
        count = 0
        for char in word:
            if char not in strg:
                break
            else:
                count +=1 
                continue
        if count == len(word) and tempCount < len(word):
            tempCount = len(word)
            str1 = word
            count = 0
    return str1

def isSubSequence(str1, str2): 
  
    m = len(str1); 
    n = len(str2); 
  
    j = 0; # For index of str1 (or subsequence 
  
    # Traverse str2 and str1, and compare current 
    # character of str2 with first unmatched char 
    # of str1, if matched then move ahead in str1 
    i = 0; 
    while (i < n and j < m):
        #print(str1[j], str2[i])
        if (str1[j] == str2[i]): 
            j += 1
        i += 1
    #print(j==m)
    # If all characters of str1 were found in str2 
    return (j == m)
  
# Returns the longest string in dictionary which is a 
# subsequence of str. 
def findLongestString(dict1, str1): 
    result = ""; 
    length = 0; 
  
    # Traverse through all words of dictionary 
    for word in dict1:           
        # If current word is subsequence of str and is largest 
        # such word so far. 
        if (length < len(word) and isSubSequence(word, str1)): 
            result = word; 
            length = len(word); 
  
    # Return longest string 
    return result; 
  
# Driver program to test above function 
  
dict1 = ["ale", "apple", "monkey", "plea"]; 
str1 = "abpcplea" ; 
#print(findLongestString(dict1, str1)); 

class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

def kth_smallest(root, k):
    stack = []
    while root or stack:
        while root:
            stack.append(root)
            root = root.left
        root = stack.pop()
        k -= 1
        if k == 0:
            break
        root = root.right
    return root.val

#root = TreeNode(8)  
#root.left = TreeNode(5)  
#root.right = TreeNode(14) 
#root.left.left = TreeNode(4)  
#root.left.right = TreeNode(6) 
#root.left.right.left = TreeNode(8)  
#root.left.right.right = TreeNode(7)  
#root.right.right = TreeNode(24) 
#root.right.right.left = TreeNode(22)  

#print(kth_smallest(root, 2))
#print(kth_smallest(root, 3))

def triangeForm(arr):
    myList = []
    n = len(arr)
    arr.sort()
    if n < 3:
        print("Triangle cannot be formed")
    for i in range(0,n-2):
        if arr[i]+arr[i+1] > arr[i+2]:
            myList.append((arr[i],arr[i+1],arr[i+2]))
    return myList

def kClosest(points, K):
    points.sort()
    print(points)
    points.sort(key = lambda P: P[0]**2 + P[1]**2)
    print(points)
    return points[:K]

def logestParenthesis(myStr):
    openList = ["{","[","("]
    closeList =["}","]",")"]
    stack = []
    for i in myStr:
        if i in openList:
            stack.append(i)
        if i in closeList:
            pos = closeList.index(i)
            if len(stack) > 0 and (openList[pos] == stack[len(stack)-1]):
                stack.pop()
    if len(stack) == 0:
        return "Balanced"
    else:
        return "Unbalanced"

        
# Max Points on the same Line
        #:type points: List[Point]
        #:rtype: int
class Point:
    def __init__(self, a=0, b=0):
        self.x = a
        self.y = b
class Solution(object):
    def maxPoints(self, points):
        """
        :type points: List[Point]
        :rtype: int
        """
        max_points = 0
        for i, start in enumerate(points):
            print("The Points",i,start.x)
            slope_count, same = collections.defaultdict(int), 1
            for j in range(i + 1, len(points)):
                end = points[j]
                print(end.x,end.y)
                if start.x == end.x and start.y == end.y:
                    same += 1
                else:
                    slope = float("inf")
                    if start.x - end.x != 0:
                        slope = (start.y - end.y) * 1.0 / (start.x - end.x)
                        print("slope", slope)
                    slope_count[slope] += 1
            current_max = same
            print("current_max",current_max)
            for slope in slope_count:
                current_max = max(current_max, slope_count[slope] + same)

            max_points = max(max_points, current_max)

        return max_points

if __name__ == "__main__":
    print(Solution().maxPoints([Point(4,3), Point(3,5), Point()]))

def fourSum(num,target):
    res  = set()
    for i in combinations(num,4):
        if sum(i) == target:
            tup = sorted(i)
            res.add(" ".join([str(x) for x in tup]))
    out = "$ ".join(res)
            #res.add(" ".join(sorted([str(x) for x in i])))
    #myList = list([[int(z) for z in x.split()] for x in res])
    #myList =  list(([sorted(i) for i in myList]))
    return out

def maxSubArray(nums):
    n = len(nums)
    curr_sum = max_sum = nums[0]
    for i in range(1, n):
        curr_sum = max(nums[i], curr_sum + nums[i])
        max_sum = max(max_sum, curr_sum)
            
    return max_sum

def paintHouse(costs):
    res = 0
    for i in costs:
        res = res + min(i)
    return res

def editDistance(str1,str2,m,n):
    print(m,n)
    if m==0:
        return n
    if n==0:
        return m
    if str1[m-1] == str2[n-1]:
        return editDistance(str1,str2,m-1,n-1)
        print(str1[m-1],str2[n-1])
    else:
        return 1 + min(editDistance(str1,str2,m,n-1),
                       editDistance(str1,str2,m-1,n),
                       editDistance(str1,str2,m-1,n-1))
        
class MaxStack(list):
    def push(self, x):
        m = max(x, self[-1][1] if self else 0)
        self.append((x, m))
    def pop(self):
        return list.pop(self)[0]
        
    def top(self):
        return self[-1][0]

    def peekMax(self):
        return self[-1][1]

    def popMax(self):
        print(self)
        m = self[-1][1]
        b = []
        while self[-1][0] != m:
            b.append(self.pop())
        
        print(self)
        self.pop()
        map(self.push, reversed(b))
        print(m)
        return m
    
#obj = MaxStack()
#obj.push(5)
#obj.push(1)
#obj.push(4)
#obj.top()
#obj.popMax()
#obj.top()
#obj.pop()

class Solution(object):
    def canPartitionKSubsets(self, nums, k):
        target, rem = divmod(sum(nums), k)
        print(target,rem)
        if rem: return False

        def search(groups):
            if not nums: return True
            v = nums.pop()
            for i, group in enumerate(groups):
                if group + v <= target:
                    groups[i] += v
                    print(group)
                    if search(groups): return True
                    groups[i] -= v
                if not group: break
            nums.append(v)
            return False
        nums.sort()
        if nums[-1] > target: return False
        while nums and nums[-1] == target:
            nums.pop()
            k -= 1
        return search([0] * k)

if __name__ == "__main__":
    print(Solution().canPartitionKSubsets([4, 3, 2, 3, 5, 2, 1], 4))

#Given a flowerbed (represented as an array containing 0 and 1, where 0 means empty and 1 means not empty), and a number n, return if n new flowers can be planted in it without violating the no-adjacent-flowers rule.
def canPlaceFlower(bed, n):
    for i in range(len(bed)):
        if (bed[i]==0 and (i==0 or bed[i-1]==0) and (i==len(bed)-1 or bed[i+1]==0)):
            bed[i] == 1
            n -= 1
    return n<=0

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
                                    dp[i-1][j],
                                    dp[i-1][j-1])
    return dp[m][n]


# Program to count islands in boolean 2D matrix 
class Graph: 

	def __init__(self, row, col, g): 
		self.ROW = row 
		self.COL = col 
		self.graph = g 

	# A function to check if a given cell 
	# (row, col) can be included in DFS 
	def isSafe(self, i, j, visited): 
		# row number is in range, column number 
		# is in range and value is 1 
		# and not yet visited 
		return (i >= 0 and i < self.ROW and
				j >= 0 and j < self.COL and
				not visited[i][j] and self.graph[i][j]) 
			

	# A utility function to do DFS for a 2D 
	# boolean matrix. It only considers 
	# the 8 neighbours as adjacent vertices 
	def DFS(self, i, j, visited): 

		# These arrays are used to get row and 
		# column numbers of 8 neighbours 
		# of a given cell 
		rowNbr = [-1, -1, -1, 0, 0, 1, 1, 1]; 
		colNbr = [-1, 0, 1, -1, 1, -1, 0, 1]; 
		
		# Mark this cell as visited 
		visited[i][j] = True

		# Recur for all connected neighbours 
		for k in range(8): 
			if self.isSafe(i + rowNbr[k], j + colNbr[k], visited): 
				self.DFS(i + rowNbr[k], j + colNbr[k], visited) 


	# The main function that returns 
	# count of islands in a given boolean 
	# 2D matrix 
	def countIslands(self): 
		# Make a bool array to mark visited cells. 
		# Initially all cells are unvisited 
		visited = [[False for j in range(self.COL)]for i in range(self.ROW)] 

		# Initialize count as 0 and travese 
		# through the all cells of 
		# given matrix 
		count = 0
		for i in range(self.ROW): 
			for j in range(self.COL): 
				# If a cell with value 1 is not visited yet, 
				# then new island found 
				if visited[i][j] == False and self.graph[i][j] == 1: 
					# Visit all cells in this island 
					# and increment island count 
					self.DFS(i, j, visited) 
					count += 1

		return count 


graph = [[1, 1, 0, 0, 0], 
		[0, 1, 0, 0, 1], 
		[1, 0, 0, 1, 1], 
		[0, 0, 0, 0, 0], 
		[1, 0, 1, 0, 1]] 


row = len(graph) 
col = len(graph[0]) 

g = Graph(row, col, graph) 

print ("Number of islands is:")
print (g.countIslands() )

def trap(height):
    areas = 0
    max_l = max_r = 0
    l = 0
    r = len(height)-1
    while l < r:
        if height[l] < height[r]:
            print("l",height[l])
            if height[l] > max_l:
                max_l = height[l]
                print("max_l",max_l)
            else:
                areas += max_l - height[l]
            l +=1
        else:
            if height[r] > max_r:
                print(height[r])
                max_r = height[r]
            else:
                areas += max_r - height[r]
            r -=1
    return areas

import os
def processLogs():
    if os.path.exists("req_host_Logs.txt"):
        os.remove("req_host_Logs.txt")
    val = []
    myDict = {}
    with open("host_Logs.txt","r") as logs:
        for line in logs:
            val.append(line.split(" "))
    for i in range(len(val)):
        if val[i][3].strip("[") in myDict:
            myDict[val[i][3].strip("[")] += 1
        else:
            myDict[val[i][3].strip("[")] = 1
    for items in myDict.keys():
        if myDict[items] > 1:
            f = open("req_host_Logs.txt","a")
            f.write(items + '\n')
            f.close()

import tableauserverclient as TSC

def tableauWB():
    server = TSC.Server("http://hawk-ui")
    #server.user_server_version()
    tableau_auth = TSC.TableauAuth("ebeamsys","lotus")
    with server.auth.sign_in(tableau_auth):
        new_workbook = TSC.WorknookItem(name="TC18", project_id="")
        publish_mode = TSC.Server.PublishMode.CreateNew
        new_workbook = server.workbooks.publish(new_workbook,"http://hawk-ui/t/hawk-ui/views/BeamCurrentDiag_MultipleTools/FactoryBeamsDashboard?iframeSizedToWindow=true&:embed=y&:showAppBanner=false&:display_count=no&:showVizHome=no",publish_mode)
        
        print("Workbook Published".format(new_workbook.id))

