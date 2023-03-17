# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 11:36:43 2019

@author: prchandr
"""

class Node: 
  
    # Function to initialise the node object 
    def __init__(self, data): 
        self.data = data  # Assign data 
        self.next = None  # Initialize next as null 
  
  
# Linked List class contains a Node object 
class LinkedList: 
  
    # Function to initialize head 
    def __init__(self): 
        self.head = None
  
  
    # Functio to insert a new node at the beginning 
    def push(self, new_data): 
  
        # 1 & 2: Allocate the Node & 
        #        Put in the data 
        new_node = Node(new_data) 
  
        # 3. Make next of new Node as head 
        new_node.next = self.head 
  
        # 4. Move the head to point to new Node 
        self.head = new_node 
  
  
    # This function is in LinkedList class. Inserts a 
    # new node after the given prev_node. This method is 
    # defined inside LinkedList class shown above */ 
    def insertAfter(self, prev_node, new_data): 
  
        # 1. check if the given prev_node exists 
        if prev_node is None: 
            print ("The given previous node must inLinkedList.")
            return
  
        #  2. create new node & 
        #      Put in the data 
        new_node = Node(new_data) 
  
        # 4. Make next of new Node as next of prev_node 
        new_node.next = prev_node.next
  
        # 5. make next of prev_node as new_node 
        prev_node.next = new_node 
  
  
    # This function is defined in Linked List class 
    # Appends a new node at the end.  This method is 
    # defined inside LinkedList class shown above */ 
    def append(self, new_data): 
  
        # 1. Create a new node 
        # 2. Put in the data 
        # 3. Set next as None 
        new_node = Node(new_data) 
        # 4. If the Linked List is empty, then make the 
        #    new node as head 
        if self.head is None: 
            self.head = new_node 
            return
        # 5. Else traverse till the last node 
        last = self.head 
        while (last.next): 
            last = last.next
        # 6. Change the next of last node 
        last.next =  new_node 

    #Utility function to print the linked list 
    def printList(self): 
        temp = self.head 
        while (temp):
            print(temp.data)
            temp = temp.next
    
    def condense(self,head):
        if head is None:
            return head
        tempList = []
        temp = head
        print(temp.data)
        while temp!= None:
            if temp.next.data in tempList:
                if temp.next.next == None:
                    temp.next = None
                else:
                    temp.next = temp.next.next
            else:
                tempList.append(temp.data)
            temp = temp.next
        print(tempList)
        return temp
    
    def condense(head):
    tempList = []
    cur1 = head
    prev = None
    while cur1 != None:
        if cur1.data in tempList:
            prev.next = cur1.next
        else:
            tempList.append(cur1.data)
            prev = cur1
        cur1 =  prev.next
    return head
    
# =============================================================================
#     # swap Nodes, given values of two nodes that we want to switch
#     def swap(head, a, b):
#       node_a = None
#       p_a  = None
#       node_b = None
#       p_b = None
#       
#       dummy = ListNode(None)
#       dummy.next = head
#       
#       # hunting
#       cur = head
#       parent = dummy
#       
#       while cur != None:
#         if cur.value == a and node_a is None:
#           node_a = cur
#           p_a = parent
#         
#         if cur.value == b and node_b is None:
#           node_b = cur
#           p_b = parent
#         
#         parent = cur
#         cur = cur.next
#       
#           
#       # if not found
#       if node_a == None or node_b == None:
#         return head
#         
#       # swap
#       p_a.next = node_b
#       p_b.next = node_a
#       ref = node_b.next
#       node_b.next = node_a.next
#       node_a.next = ref
#       
#       return dummy.next
#         
#     print_forward(swap(ll,1, 15))        
# =============================================================================
            
# Code execution starts here 
if __name__=='__main__': 
  
    # Start with the empty list 
    llist = LinkedList() 
  
    # Insert 6.  So linked list becomes 6->None 
    llist.append(6) 
  
    # Insert 7 at the beginning. So linked list becomes 7->6->None 
    llist.push(7); 
  
    # Insert 1 at the beginning. So linked list becomes 1->7->6->None 
    llist.push(1); 
  
    # Insert 4 at the end. So linked list becomes 1->7->6->4->None 
    llist.append(4) 
    llist.append(6)
  
    # Insert 8, after 7. So linked list becomes 1 -> 7-> 8-> 6-> 4-> None 
    llist.insertAfter(llist.head.next, 8) 
    
    llist.condense(llist.head)
  
    print ('Created linked list is:')
    
    llist.printList()
   
"""
2. Add Two Numbers
Medium

12734

2910

Add to List

Share
You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.

Example 1:


Input: l1 = [2,4,3], l2 = [5,6,4]
Output: [7,0,8]
Explanation: 342 + 465 = 807.

"""
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
Add Two Numbers II
You are given two non-empty linked lists representing two non-negative integers. The most significant digit comes first and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.

Example 1:


Input: l1 = [7,2,4,3], l2 = [5,6,4]
Output: [7,8,0,7]
"""
class Solution:
    def reverseList(self,head):
        last = None
        while head:
            tmp = head.next
            head.next = last
            last = head
            head = tmp
        
        return last
    
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        l1= self.reverseList(l1)
        l2 = self.reverseList(l2)
        
        head = None
        carry = 0
        
        while l1 or l2:
            val1 = (l1.val if l1 else 0)
            val2 = (l2.val if l2 else 0)
            TSum = (val1 + val2+carry)%10
            carry = (val1 + val2+carry)//10
            curr = ListNode(TSum)
            curr.next = head
            head = curr
            
            l1 = (l1.next if l1 else None)
            l2 = (l2.next if l2 else None)
        if carry:
            curr = ListNode(carry)
            curr.next = head
            head = curr
        return head
    


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

    
    