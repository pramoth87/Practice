# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 13:59:21 2019

@author: prchandr
"""

class Node:
    def __init__(self,data):
        self.left = None
        self.right = None
        self.count = 1
        self.key = data
    def search(root,key):
        if root is None or root.key == key:
            return root
        if root.key<key:
            return Node.search(root.right,key)
        else:
            return Node.search(root.left,key)
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
    def minValueNode(node):
        current = node
        while current.left != None:
            current = current.left
        return current
    def deleteNode(root,key):
        try:
            if root == None:
                return root
            if key < root.key:
                root.left = Node.deleteNode(root.left, key)
            elif key > root.key:
                root.right = Node.deleteNode(root.right,key)
            else:
                if root.count > 1:
                    root.count -= 1
                    return root
                if root.left == None:
                    temp = root.right
                    return temp
                elif root.right == None:
                    temp = root.left
                    return temp
                temp = Node.minValueNode(root.right)
                root.key = temp.key
                root.right = Node.deleteNode(root.right, temp.key)
            return root
        except ValueError:
            
            print("Invalid Entry - try again")
    def treePathsSumUtil(root,val):
        if root is None:
            return 0
        val = (val*10 + root.key)
        if root.left is None and root.right is None:
            return val
        return (Node.treePathsSumUtil(root.left, val)+ Node.treePathsSumUtil(root.right, val))
    def treePathsSum(root):
        return Node.treePathsSumUtil(root,0)
    
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
    def minDiffBSTUpdate(self,root):
        def dfs(node):
            if node:
                dfs(node.left)
                self.ans = min(self.ans, node.key - self.prev)
                self.prev = node.key
                dfs(node.right)
            return ans
        self.prev = float('-inf')
        self.ans = float('inf')
        dfs(root)       
                
            
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
    
    print("Delete 20")  
    root = Node.deleteNode(root, 20)  
    print("Inorder traversal of the modified tree")  
    Node.inorder(root)  
    print() 
  
    print("Delete 12") 
    root = Node.deleteNode(root, 12)  
    print("Inorder traversal of the modified tree")  
    Node.inorder(root)  
    print() 
  
    print("Delete 9") 
    root = Node.deleteNode(root, 9)  
    print("Inorder traversal of the modified tree")  
    Node.inorder(root)
    print()
    
    print("Sum of all Paths", Node.treePathsSum(root))
    
    
    print(Node.kth_smallest(root, 2))
    print(Node.kth_smallest(root, 3))
    print("MinDiff in BST",Node.minDiffInBST(root))
    

    
# Python Program for Lowest Common Ancestor in a Binary Tree 
# O(n) solution to find LCS of two given values n1 and n2 
# A binary tree node 
class Node: 
	# Constructor to create a new binary node 
	def __init__(self, key): 
		self.key = key 
		self.left = None
		self.right = None
# Finds the path from root node to given root of the tree. 
# Stores the path in a list path[], returns true if path 
# exists otherwise false 
def findPath( root, path, k): 
	# Baes Case 
	if root is None: 
		return False
	# Store this node is path vector. The node will be 
	# removed if not in path from root to k 
	path.append(root.key) 
	# See if the k is same as root's key 
	if root.key == k : 
		return True
	# Check if k is found in left or right sub-tree 
	if ((root.left != None and findPath(root.left, path, k)) or
			(root.right!= None and findPath(root.right, path, k))): 
		return True
	# If not present in subtree rooted with root, remove 
	# root from path and return False 
	path.pop() 
	return False
# Returns LCA if node n1 , n2 are present in the given 
# binary tre otherwise return -1 
def findLCA(root, n1, n2): 

	# To store paths to n1 and n2 fromthe root 
	path1 = [] 
	path2 = [] 

	# Find paths from root to n1 and root to n2. 
	# If either n1 or n2 is not present , return -1 
	if (not findPath(root, path1, n1) or not findPath(root, path2, n2)): 
		return -1
	# Compare the paths to get the first different value 
	i = 0
	while(i < len(path1) and i < len(path2)): 
		if path1[i] != path2[i]: 
			break
		i += 1
	return path1[i-1] 

# Driver program to test above function 
# Let's create the Binary Tree shown in above diagram 
root = Node(1) 
root.left = Node(2) 
root.right = Node(3) 
root.left.left = Node(4) 
root.left.right = Node(5) 
root.right.left = Node(6) 
root.right.right = Node(7) 

print ("LCA(4, 5) = %d" %(findLCA(root, 4, 5,))) 
print ("LCA(4, 6) = %d" %(findLCA(root, 4, 6))) 
print ("LCA(3, 4) = %d" %(findLCA(root,3,4)) )
print ("LCA(2, 4) = %d" %(findLCA(root,2, 4))) 

import collections 

class TreeNode:
    def __init__(self,data):
        self.left = None
        self.right = None
        self.key = data
    
    def search(root,key):
        if root is None and root.key == key:
            return root
        elif root.key > key:
            return TreeNode.search(root.left, key)
        else:
            return TreeNode.search(root.right,key)
    
    def insert(root,key):
        if root is None:
            return TreeNode(key)
        elif root.key >= key:
            root.left = TreeNode.insert(root.left, key)
        else:
            root.right = TreeNode.insert(root.right, key)
        return root
    
    def inorder(root):
        if root != None:
            TreeNode.inorder(root.left)
            print(root.key, end = " ")
            TreeNode.inorder(root.right)
    
    def deleteNode(root,key):
        if root == None:
            return root
        elif root.key > key:
            root.left = TreeNode.deleteNode(root.left,key)
        elif root.key < key:
            root.right = TreeNode.deleteNode(root.right, key)
        else:
            if root.left == None:
                temp = root.right
                return temp
            elif root.right == None:
                temp = root.left
                return temp
            temp = TreeNode.getMinValue(root.right)
            root.key = temp.key
            root.right = TreeNode.deleteNode(root.right, temp.key)
        return root
    
    def maxDepth(node):
        if node == None:  
            return 0
        return 1 + max(TreeNode.maxDepth(node.left),  
                    TreeNode.maxDepth(node.right))  
    
    def getMinValue(root):
        current = root
        while current.left != None:
            current = root.left
        return current
    
    def verticalTraversal(root):
        seen = collections.defaultdict(lambda: collections.defaultdict(list))
        print("Seen",seen)
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
                report.extend(sorted(node.key for node in seen[x][y]))
                print(report)
            ans.append(report)
    
        return ans
    
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
    

if __name__ == '__main__': 
      
    # Let us create following BST  
    # 12(3)  
    # / \  
    # 10(2) 20(1)  
    # / \  
    # 9(1) 11(1)
    root = None
    root = TreeNode.insert(root, 12)  
    root = TreeNode.insert(root, 10)  
    root = TreeNode.insert(root, 20)  
    root = TreeNode.insert(root, 9)  
    root = TreeNode.insert(root, 11)  
    #root = TreeNode.insert(root, 13)  
    #root = TreeNode.insert(root, 14)  
    #root = TreeNode.insert(root, 15) 
    print("Inorder traversal of the given tree")  
    TreeNode.inorder(root)  
    print() 
    
    #print("Delete 20")  
    #root = TreeNode.deleteNode(root, 20)  
    print("Inorder traversal of the modified tree")  
    TreeNode.inorder(root)  
    print() 
  
    #print("Delete 14") 
    #root = TreeNode.deleteNode(root, 14)  
    print("Inorder traversal of the modified tree")  
    TreeNode.inorder(root)  
    print() 
  
    #print("Delete 12") 
    #root = TreeNode.deleteNode(root, 12)  
    print("Inorder traversal of the modified tree")  
    TreeNode.inorder(root)
    print()
    
    print("Max Depth", TreeNode.maxDepth(root))
    print("Vertical Traversal", TreeNode.verticalTranversalBFS(root))



# Data structure to store a Binary Tree node
class Node:
	def __init__(self, data, left=None, right=None):
		self.data = data
		self.left = left
		self.right = right


# Recursive function to insert a key into BST
def insert(root, key):

	# if the root is None, create a new node and return it
	if root is None:
		return Node(key)

	# if given key is less than the root node, recur for left subtree
	if key < root.data:
		root.left = insert(root.left, key)

	# if given key is more than the root node, recur for right subtree
	else:
		root.right = insert(root.right, key)

	return root


# Recursive function to build a BST from given sequence
def buildTree(seq):

	# construct a BST by inserting keys from the given sequence
	root = None
	for key in seq:
		root = insert(root, key)

	# return root node
	return root


# Function to compare the preorder traversal of a BST with given sequence
def comparePreOrder(root, seq, index):

	# base case
	if root is None:
		return True, index

	# return false if next element in the given sequence doesn't match
	# with the next element in preorder traversal of BST
	if seq[index] is not root.data:
		return False, index

	# increment index
	index = index + 1

	# compare the left and right subtrees
	left, index = comparePreOrder(root.left, seq, index)
	right, index = comparePreOrder(root.right, seq, index)

	return (left and right), index


# Function to check if a given sequence represents preorder traversal of a BST
def isBST(seq):

	""" 1. Construct the BST from given sequence """

	root = buildTree(seq)

	""" 2. Compare the preorder traversal of BST with given sequence """

	# index stores index of next unprocessed node in preorder sequence
	success, index = comparePreOrder(root, seq, 0)
	return success and (index == len(seq))


if __name__ == '__main__':

	seq = [15, 10, 8, 12, 20, 16, 25]

	if isBST(seq):
		print("YES")
	else:
		print("Given sequence doesn't represent preorder traversal of a BST")
        
"""
Trim a Binary Search Tree
Given the root of a binary search tree and the lowest and highest boundaries as low and high, trim the tree so that all its elements lies in [low, high]. Trimming the tree should not change the relative structure of the elements that will remain in the tree (i.e., any node's descendant should remain a descendant). It can be proven that there is a unique answer.

Return the root of the trimmed binary search tree. Note that the root may change depending on the given bounds.

Example 1:


Input: root = [1,0,2], low = 1, high = 2
Output: [1,null,2]
"""    
class Solution:
    def trimBST(self, root: TreeNode, low: int, high: int) -> TreeNode:
        def trim(node):
            if not node:
                return None
            elif node.val>high:
                return trim(node.left)
            elif node.val<low:
                return trim(node.right)
            else:
                node.left = trim(node.left)
                node.right = trim(node.right)
                return node
        return trim(root)

        
        

    

