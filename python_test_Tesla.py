#  Question 1
#  Write a function to implement the state machine shown in the diagram below.
#  * The initial state of the state machine should be A
#  * State machine inputs are provided as an argument to the function
#  * The function should output the current state of the state machine
#
#
#     Input = 1   +---------+   Input = 4
#   +------------>|         |<------------+
#   |             | State A |             |
#   |  Input = 2  |         |  Input = 3  |
#   |   +-------- +---------+ --------+   |
#   |   |                             |   |
#   |   V                             V   |
# +---------+      Input = 6      +---------+
# |         |<--------------------|         |
# | State C |                     | State B |
# |         |-------------------->|         |
# +---------+      Input = 5      +---------+
#
#                                 +---------+
#              Input =            |         |
#              Unexpected Value   |  FAULT  |
#             +------------------>|         |
#                                 +---------+

# Example:
#
# result = state_machine([2, 1, 3, 4, 2, 5])
# assert(result == 'B')
# result = state_machine([4])
# assert(result == 'FAULT')

        
class State:
    def __init__(self):
        self.current = 'A'
        self.dict = {('A',2):'C',('A',3):'B',('B',4):'A',('B',6):'C',('C',1):'A',('C',5):'B'}
    
    def state_machine(self,inputArr):
        for inp in inputArr:
            if (self.current,inp) in self.dict:
                self.current = self.dict[(self.current,inp)]
            else:
                return 'FAULT'
        
        return self.current

cs = State()
print("Current State:", cs.state_machine([2, 1, 3, 4, 2, 5]))

# Question 2:
# Given an array of monotonically increasing integers 
# that's been rotated by an arbitrary amount X, write a 
# function to return the arbitrary rotation amount X. 
#  
# Examples: 
#     6 = find_rotation([5,6,7,8,9,10,2,3,4]) 
#     3 = find_rotation([9,12,17,2,3,5,6,8])


# Brute Force method to find the rotation

def find_rotation(arr):
    minVal = arr[0]
    for i in range(len(arr)):
        if minVal > arr[i]:
            minVal = arr[i]
            minIndex = i
    return minIndex

# Using Binary Search to find the Rotation for better Time Complexity

def findRotationBinarySearch(arr):
    start, end = 0, len(arr)-1
    while start < end:
        mid = start+(end-start) // 2
        if arr[mid] < arr[mid-1]:
            return mid
        elif arr[mid] > arr[start]:
            start = mid+1
        else:
            end = mid
    return 0


#Question 3:
# Given an input integer n, return a list with numbers from 1 to n with the following constraint: 
# Use 2 threads, one that returns the even numbers and another thread that returns the odd numbers.  
# Example: 
# [1,2,3,4,5] = threaded_numbers(5)



from threading import Thread,Event

class Threading:
    def __init__(self,n):
        self.limit = n
        self.number = 0
        self.even_avai = Event()
        self.odd_avai = Event()
        self.finalList = []
    
    def even_thread(self):
        while self.number < self.limit:
            self.even_avai.wait()
            self.finalList.append(self.number)
            self.number += 1
            self.even_avai.clear()
            self.odd_avai.set()
    
    def odd_thread(self):
        while self.number < self.limit:
            self.odd_avai.wait()
            self.finalList.append(self.number)
            self.number += 1
            self.odd_avai.clear()
            self.even_avai.set()

    def naturalNumbers(self):
        even = Thread(target=self.even_thread)
        odd = Thread(target=self.odd_thread)
        self.even_avai.set()
        even.start()
        odd.start()
        even.join()
        odd.join()
        
        return self.finalList
    
e = Threading(5)
print("List of Number", e.naturalNumbers())



#Question 4:
# Given a node that has children and a name, determine the ancestry path from the given node to a specified node.
# Use the following function to create a test case.
#def createTree(): return Node(1,[Node(2,[Node(5),Node(6,[Node(10)])]),Node(3,[Node(7,ode(11),Node(12)])]),Node(4,[Node(8,[Node(13)]),Node(9,[Node(14)])])])
#               1
#             / | \
#            /  |  \         
#           2   3   4
#         / |   |   | \
#        /  |   |   |  \        
#       5   6   7   8   9
#         /   / |   |     \
#        /   /  |   |      \
#       10  11  12  13     14
#
# Return the acestry path from the top to an arbitrary node.
#
# Example: Given the node structure above and if asked to find the   
# ancestry path to node 11, the result would be [1, 3, 7, 11].


class Node:
    def __init__(self,x):
        self.data = x
        self.child = []
    
def printRootToLeafPaths(root,target,result=[],ans=[]):
    result.append(root.data)    
    if len(root.child) == 0 and result[-1]==target:
        print("The ancestry Path is",result)
        return result
    else:
        for subtree in root.child:
            printRootToLeafPaths(subtree,target,result,ans)
        result.pop() 
        
if __name__ == '__main__':
    root = Node(1)
    root.child.append(Node(2))
    root.child.append(Node(3))
    root.child.append(Node(4))
    root.child[0].child.append(Node(5))
    root.child[0].child.append(Node(6))
    root.child[1].child.append(Node(7))
    root.child[2].child.append(Node(8))
    root.child[2].child.append(Node(9))
    root.child[1].child[0].child.append(Node(10))
    root.child[1].child[0].child.append(Node(11))
    root.child[2].child[0].child.append(Node(12))
    root.child[2].child[0].child.append(Node(13))
    
    printRootToLeafPaths(root,11)
            
