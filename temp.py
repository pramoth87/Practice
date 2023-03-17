# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import re
from itertools import groupby
from collections import defaultdict


def twoSum(num, target):
    tempDict = {num[0]:0}
    for i in range(1,len(num)):
        checkNum = target-num[i]
        if(checkNum in tempDict.keys()):
            return[tempDict[checkNum],i]
            print("Here is the output", num[tempDict[checkNum]], num[i])
        else:
            tempDict[num[i]]=i
            
def addTwoNumbers(l1, l2):
    solution = []
    carry = 0
    for i,j in zip(l1,l2):
        print(i,j)
        if carry == 1:
            s1 = carry+i+j
            carry = 0
            print(s1)
        else:
            s1 = i+j
            print(s1)
        if s1>=10:
            print(s1)
            solution.append(s1%10)
            print(solution)
            carry = 1
        else:
            print(s1)
            solution.append(s1)
            print(solution[::-1])
    #print(solution.reverse())

def lengthOfLongestSubstring(word):
    n = len(word)
    longest = 0
    for i in range(n):
        seen = set()
        print("Before J",seen)
        for j in range(i, n):
            if word[j] in seen: break
            seen.add(word[j])
            print("Inside J",seen)
        longest = max(len(seen), longest)
        print(seen)
    return longest

def medianSortedArrays(num1, num2):
    num1.extend(num2)
    mergeList = num1
    print(mergeList)
    if len(mergeList)%2 != 0:
        pointer = int(len(mergeList)/2)
        print("Median is: ", mergeList[pointer])
    if len(mergeList)%2 == 0:
        pointer = int(len(mergeList)/2)
        median = (mergeList[pointer-1] + mergeList[pointer])/2
        print("Median is: ", median)

def longestPalindrome(s):
    m = ''  # Memory to remember a palindrome
    for i in range(len(s)):  # i = start, O = n
        print(s[i])
        for j in range(len(s), i, -1):  # j = end, O = n^2
            print(s[j-1])
            if len(m) >= j-i:  # To reduce time
                break
            elif s[i:j] == s[i:j][::-1]:
                print(s[i:j])
                m = s[i:j]
                break
    return len(m)

def reverse(num): 
  rev = 0
  n = num
  if (num<0):
      n *= -1
  while (n>0):     
      rev = (rev *10) + n %10    
      n = n //10
  if (num<0):
      rev *= -1
  return rev
      
def palindrome(num):
    if str(num) == str(num)[::-1]:
        print("Number is Palidrome")
    else:
        print("Not Palindrome")

def intPalindrome(num):
    print(num)
    rev = 0
    n = num
    while(n>0):
        rev = (rev*10) + n%10
        n = n // 10
    if str(num) == str(n):
        print ("Number is Palindrome")
    else:
        print("Nope")

def isMatch(s, p):
    grps = groupby(p)
    print("Expected One", grps)
    l = [(x, len(list(y))) for x, y in grps]
    new_p = ''
    for c, cnt in l:
        if c == '*':
            new_p = ''.join([new_p, '.*'])
        elif c == '?':
            new_p = ''.join([new_p, '.{', str(cnt), '}'])
        else:
            new_p = ''.join([new_p, c * cnt])
    p = '^' + new_p + '$'
    MY_RE = re.compile(p)
    return True if MY_RE.match(s) else False

def simpleArraySum(ar):
    c = 0
    for i in range(len(ar)):
        c += ar[i]
    print ("The Sum of the Array", c)

def compareTriplets(a, b):
    Alice = 0
    Bob = 0
    for i,j in zip(a,b):
        print(i,j)
        if i>j:
            Alice += 1
        if i<j:
            Bob += 1
    return Alice,Bob

def diagonalDifference(a,n):
    # Complete this function  
    sum1  = 0
    sum2  = 0
    for i in range(n):
        print(i)
        sum1 += int(a[i][i])
        sum2 += int(a[i][n-i-1])
    return abs(sum1 - sum2)

def plusMinus(arr):
    n = len(arr)
    neg = 0
    pos = 0
    zer = 0
    for i in range(n):
        if arr[i]<1:
            neg +=1
        if arr[i]>1:
            pos += 1
        if arr[i] == 0:
            zer += 1
    print(pos/n)
    print(neg/n)
    print(zer/n)

def staircase(n):
    for i in range(1, n+1):
         print(("#" * i).rjust(n))
     
def maxArea(A): 
    l = 0
    r = len(A) -1
    print(r)
    area = 0
    while l < r: 
        print(area, A[l], A[r], (r-1))
        # Calculating the max area 
        area = max(area, min(A[l],  
                        A[r]) * (r - l)) 
        print(area, A[l], A[r])
      
        if A[l] < A[r]: 
            l += 1
        else: 
            r -= 1
    return area        

def miniMaxSum(arr):
    arr.sort()
    low =0
    high=0
    for i in range(len(arr)-1):
        low += arr[i]
    for i in range(1,len(arr)):
        high += arr[i]
    print(low, high)

def birthdayCakeCandles(ar):
    candles = max(ar)
    count = ar.count(candles)
    if count > 1:
        candles = count
        return candles
    else:
        return 1

def num2roman(num):
    num_map = [(1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'), (100, 'C'), (90, 'XC'),
           (50, 'L'), (40, 'XL'), (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')]
    roman = ''
    while num > 0:
        for i, r in num_map:
            while num >= i:
                roman += r
                num -= i
    return roman

def timeConversion(s):
    if s[-2:]=="AM" and s[:2]=="12":
        return "00" + s[2:-2]
    elif s[-2:]=="AM":
        return s[:-2]
    elif s[-2:]=="PM" and s[:2]=="12":
        return s[:-2]
    else:
        return str(int(s[:2])+12) + s[2:-2]

def hackerrankInString(s):
    for a0 in range(len(s)):
        print(s[a0])
        a="hackerrank"
        c=0
        for i in s:
            if i==a[c]:
                c+=1
                if c>=10:
                    break
    if c==10:
        print("YES")
    else:
        print ("NO")
        
def is_pangram(phrase):
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    for char in alphabet.lower():
        if char not in phrase.lower():
            return "Not Pangram"
    return "Pangram"        
    
def weightedUniformStrings(s, queries):
    weights = set()
    prev = -1
    length = 0
    for c in s:
        weight = ord(c) - ord('a') + 1
        weights.add(weight)
        print(c)
        print(prev,c)
        if prev == c:
            length += 1
            weights.add(length*weight)
        else:
            prev = c
            length = 1
    rval = []
    for q in queries:
        if q in weights:
            rval.append("Yes")
        else:
            rval.append("No")
    return rval

def separateNumbers(s):
    # Complete this function
    if s[0] == s:
        print('NO')
        return
    for i in range(1, len(s)):
        mystack = []
        mystack.append(s[:i])
        while len(''.join(mystack)) < len(s):
            mystack.append(str(int(mystack[-1]) + 1))
        if ''.join(mystack) == s:
            print('YES', mystack[0])
            break
        if i == len(s) - 1:
          print('NO')

import string
from heapq import heappush, heappop
def word_count_engine(document):
  res = []
  hash_table = {} # Dict[str, int]
  heap = []
  words = document.strip(string.punctuation) 
  words = words.split(" ")
  for w in words:
    word = w.strip(string.punctuation).lower()
    if word not in hash_table:
      hash_table[word] = 1
    else:
      hash_table[word] += 1
  
  for word, count in hash_table.items():
    heappush(heap, (-count, word))

  while heap:
    count, word = heappop(heap)
    res.append(
      [word, str(-count)]
    )   
  print(res)
  return res
  
def funnyString(s):
    sList = []
    rList = []
    res1 =[]
    res2 = []
    revStr = s[::-1]
    for code in map(ord, s):
        sList.append(code)
    for code in map(ord,revStr):
        rList.append(code)
    for i in range(0,len(sList)-1):
        if i == len(sList)-1:
            break
        a = abs(sList[i] - sList[i+1])
        res1.append(a)
    for i in range(0,len(rList)-1):
        if i == len(rList)-1:
            break
        a = abs(rList[i] - rList[i+1])
        res2.append(a)
    if res1 == res2:
        print("Funny")
    else:
        print("Not Funny")
    
def gradingStudents(grades):
    result =[]
    for i in grades:
        if len(grades)>60 or len(grades) <= 0:
            print("Thank You! Please enter Grade")
            break
        elif i >=100:
            print("One of the grade is more than 100, PLEASE CHECK!!")
            break
        elif i<38:
            result.append(i)
        elif i%5==3:
            i = i+2
            result.append(i)
        elif i%5 == 4:
            i = i+1
            result.append(i)
        else:
            result.append(i)
    print(result)
    return result

def countApplesAndOranges(s, t, a, b, apples, oranges):
    finalApple = []
    finalOrange = []
    appleLoc = [x+a for x in apples]
    orangeLoc = [x+b for x in oranges]
    print(appleLoc, orangeLoc)
    for i in appleLoc:
        if i>=s and i<=t:
            finalApple.append(i)
    for i in orangeLoc:
        if i>=s and i<=t:
            finalOrange.append(i)
    print(len(finalApple))
    print(len(finalOrange))
    return len(finalApple), len(finalOrange)

def kangaroo(x1, v1, x2, v2):
    for n in range(10000):
        if((x1+v1)==(x2+v2)):
            return "YES"
        x1+=v1
        x2+=v2
    return "NO"

def getTotalX(a, b):
    nmax,nmin,count = max(a),min(b),0
    for i in range(1,int(nmin/nmax)+1):
        if(sum((i*nmax)%n for n in a)+sum(n%(i*nmax) for n in b))==0:
            count+=1
    return count

def breakingRecords(scores):
    minCnt, maxCnt = 0,0
    a = scores[0]
    b = scores[0]
    for i in range(len(scores)):
        if scores[i] > a:
            maxCnt += 1
            a = scores[i]
        if scores[i] < b:
            minCnt += 1
            b = scores[i]
    return maxCnt, minCnt
        
def birthday(s, d, m):
    count = 0
    for i in range(len(s)):
        if len(s[slice(i,m+i)])==m and sum(s[slice(i,m+i)])==d:
            count += 1
    return count

def divisibleSumPairs(n, k, ar):
    count = 0
    for i in range(len(ar)):
        for j in range(len(ar)):
            if i < j and ((ar[i]+ar[j])%k)==0:
                count += 1
    return count

from collections import Counter
def migratoryBirds(arr):
    counter = Counter(arr)
    maximum = max(counter, key=counter.get)
    print(maximum)

def fib(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fib(n-1) + fib(n-2)

def dayOfProgrammer(year):
    if (year == 1918):
        return '26.09.1918'
    elif ((year <= 1917) & (year%4 == 0)) or ((year>1918) and (year%400 == 0 or ((year%4==0) and (year%100 !=0)))):
        return '12.09.%s' %year
    else: 
        return '13.09.%s' %year

def bonAppetit(bill, k, b):
    totalBill = sum(bill)
    chargedBill = (totalBill - bill[k])/2
    if b == chargedBill:
        print("Bon Appetit")
    else:
        return abs(int(chargedBill - b))

def bubbleSort(arr): 
    n = len(arr) 
    # Traverse through all array elements 
    for i in range(n): 
        swapped = False
        # Last i elements are already 
        #  in place 
        for j in range(0, n-i-1): 
            # traverse the array from 0 to 
            # n-i-1. Swap if the element  
            # found is greater than the 
            # next element 
            if arr[j] > arr[j+1] : 
                arr[j], arr[j+1] = arr[j+1], arr[j] 
                swapped = True
        # IF no two elements were swapped 
        # by inner loop, then break 
        if swapped == False: 
            break
    print(arr)

def sockMerchant(ar):
    mylist = []
    count = 0
    for i in range(len(ar)):
        if ar[i] in mylist:
            continue
        a = ar.count(ar[i])
        mylist.append(ar[i])
        if a > 1:
            if a%2==0:
                count += (a//2)
            if a%2 != 0:
                count += ((a-1)//2)
    print(count)

from collections import Mapping
from itertools import chain
def flatten_dict(init_dict):
    res_dict = {}
    if type(init_dict) is not dict:
        return res_dict

    for k, v in init_dict.items():
        if type(v) == dict:
            res_dict.update(flatten_dict(v))
        else:
            res_dict[k] = v

    return res_dict
                
def flatten_dic(d):
    items = []
    for k, v in d.items():
        try:
            if (type(v)==type([])): 
                for l in v: items.extend(flatten_dic(l).items())
            else: 
                items.extend(flatten_dic(v).items())
        except AttributeError:
            items.append((k, v))
    return dict(items)

def flatten_json(nested_json):
    out = {}
    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '.')
                print(x,a,x[a],name,out)
        #elif type(x) is list:
         #   i = 0
          #  for a in x:
           #     flatten(a, name + str(i) + '.')
            #    i += 1
        else:
            out[name[:-1]] = x

    flatten(nested_json)
    return out

def pageCount(n, p):
    print(min(int(p/2), round(n//2-p//2)))

def countingValleys(s):
    curlevel = 0
    valleys = 0
    for c in s:
        if c == 'U':
            curlevel += 1
        else:
            curlevel -= 1 
        if curlevel == 0 and c == 'U':
            valleys = valleys + 1
    print (str(valleys))
    
def getMoneySpent(keyboards, drives, b):
    ans=-1
    for i in keyboards:
        for j in drives:
            if i+j > b:
                break
            ans = max(ans, i+j)
    print(ans)

def catAndMouse(x, y, z):
    if abs(x-z)==abs(y-z):
        print("Mouse C")
    elif abs(x-z)<abs(y-z):
        print("Cat A")
    elif abs(x-z)>abs(y-z):
        print("Cat B")

def formingMagicSquare(s):
    magicsquares=[]
    costs=[]
    for a in [2,4,6,8]:
        print(a)
        for b in [2,4,6,8]: 
            if a+b!=10 and a!=b:
                m=[a,15-a-b, b, 15-a-(10-b), 5, 15-b-(10-a), 10-b, 10-(15-a-b), 10-a]
                magicsquares.append(m)
                costs.append(sum([abs(x-y) for x,y in zip(s,m)]))
                print(a,b,costs)
    return(min(costs))
    
def pickingNumbers(a):
    counts = {}
    for i in a:
        if i not in counts:
            counts[i] = 1
        else:
            counts[i] += 1
    curmax = None    
    for k in counts:
        curcount = counts[k]
        print(k-1,k+1,counts)
        leftcount = 0
        rightcount = 0
        if k-1 in counts:
            leftcount = counts[k-1]
        if k+1 in counts:
            rightcount = counts[k+1]
        thismax = curcount + max(leftcount, rightcount)
        if curmax == None or thismax > curmax:
            curmax = thismax
    print(curmax)

def climbingLeaderboard(scores, alice):
    newList = scores + alice
    newList.sort(reverse=True)
    count = 0
    rank = []
    for i in alice:
        scores.append(i)
        scores.sort(reverse=True)
        print(scores)
        for j in range(len(scores)):
            print(scores[j])
            if scores[j]==scores[j-1] and scores[j-1] != scores[-1]:
                count+=0
            else:
                count+=1
            if scores[j] == i:
                rank.append(count)
                scores.remove(i)
                count = 0
                break
    print(rank)

def climbingLeaderboard1(scores, alice):
    unique_scores = list(reversed(sorted(set(scores))))
    print(unique_scores)

    i = len(alice)-1
    j = 0
    ans = []
    while i >= 0:
        if j >= len(unique_scores) or unique_scores[j] <= alice[i]:
            ans.append(j+1)
            i -= 1
            print(j)
        else:
            j += 1

    ans.sort(reverse=True)
    return(ans)

def hurdleRace(k, height):
    maxi = max(height)
    if maxi < k:
        return 0
    else:
        return (maxi-k)

def designerPdfViewer(h, word):
    s = string.ascii_lowercase
    w = word.lower()
    mydict = {}
    i=0
    myList=[]
    for ch in s:
        mydict[ch]=h[i]
        i +=1
    for st in w:
        myList.append(mydict[st])
    return max(myList)*len(myList)

def designerPdfViewer1(h, word):
    print(max([h[ord(c) - 97] for c in word])*len(word))

def solution(A):
    # write your code in Python 3.6
    if max(A) < 1:
        return 1
    if len(A) == 1:
        if A[0]==1:
            return 2
        else:
            return 1
    l = [0] * max(A)
    for i in range(len(A)):
        print(i)
        if A[i] >0:
            if l[A[i]-1]!=1:
                l[A[i]-1] = 1
                print(A[i]-1)
    print(l)
    for i in range(len(l)):
        if l[i] == 0:
            return i+1
    return i+2

def sliceArray(A):
    myList = []
    for i in range(len(A)):
        if A[i]==A[-1]:
            break
        if A[i]==A[i+1]:
            myList.append(A[i])
            myList.append(A[i+1])
    return(myList)

def maxLen(arr): 
    # initialize result 
    max_len = [arr[0]]
    # pick a starting point 
    for i in range(len(arr)): 
        # initialize sum for every starting point 
        #curr_sum = 0
        # try all subarrays starting with 'i' 
        for j in range(i,len(arr)): 
            # if curr_sum becomes 0, then update max_len 
            if arr[j] in max_len:
                max_len.append(arr[j])
  
    return max_len 
def maxSlice(A):
    max_sum = sub_sum = A[0]
    for i in range(1, len(A)): 
        sub_sum = max(sub_sum + A[i], A[i])
        max_sum = max(max_sum, sub_sum)
    
    return max_sum

def utopianTree(n):
    h = 0
    for i in range(n+1):
        if i%2 == 0:
            h = h+1
        else:
            h += h
    return h

def angryProfessor(k, a):
    newList = [i for i in a if i<=0]
    if len(newList)>=k:
        print("NO")
    else:
        print("YES")

def beautifulDays(i, j, k):
    count = 0
    for day in range(i,j+1):
        #print(abs(day-reverse(day))%k)
        if abs(day-reverse(day))%k==0:
            count += 1
    return count
            
def viralAdvertising(n):
    myList =[2]
    for i in range(n-1):
        myList.append(int(3*myList[i]/2))
    return sum(myList)

def saveThePrisoner(n, m, s):
    a=0
    a=(m+s-1)%n
    if a==0:
        return n
    else:
        return a

def circularArrayRotation(a, k, queries):
    a = (a[-k:] + a[:-k])
    for i in range(len(queries)):
        return(a[queries[i]])

def dictionaryValue(content):
    sk = sorted(content)
    for i in range(len(sk)-1):
        print(sk)
        if (content[sk[i]] > content[sk[i+1]]):
            print(sk[i])

def permutationEquation(p):
    max_a = max(p)
    res =[]
    for i in range(1, max_a+1):
        a = p.index(i)+1
        res.append(p.index(a)+1)
    return res

def jumpingOnClouds(c, k):
    e=100
    energy=0
    i=0
    while(i<=len(c) and i!=len(c)):
        if(c[i]==1):
            energy=e-3
            e=energy
        else:
            energy=e-1
            e=energy
        i+=k
    return energy
            
def findDigits(n):
    n = int(n)
    print(sum([1 for i in str(n).replace('0','')if n%int(i)==0]))

def extraLongFactorials(n):
    cache = {}
    def rec(n):
        if n in cache:
            print(cache[n])
            return cache[n]
        if n == 1:
            return 1
        result = n * rec(n-1)
        cache[n] = result
        return result
    return rec(n)

def appendAndDelete(s, t, k):
    lead = 0
    for i in range(min(len(s),len(t))):
        if s[i] != t[i]:
            lead = i
            break
        else:
            lead = i + 1
    d = len(s) - lead + len(t) - lead
    if k >= len(s) + len(t):
        print("Yes")
    elif d <= k and (d % 2) == (k % 2):
        print("Yes")
    else:
        print("No")

import math
def squares(a, b):
    count = 0
    for i in range(a,b+1):
        if (math.sqrt(i)==math.floor(math.sqrt(i))):
            count += 1
    return count

def squares_btw(a,b):
    count = math.floor(math.sqrt(b)) - math.floor(math.sqrt(a - 1))
    return count

def remString(s):
    l = []
    if s==None:
        return l
    for i in s:
        l.append(i)
    return l

def libraryFine(d1, m1, y1, d2, m2, y2):
    if y1-y2 > 0:
        return 10000
    if y1==y2 and m1-m2>0:
        return (m1-m2)*500
    if y1==y2 and m1==m2 and d1-d2>0:
        return (d1-d2)*15
    else: 
        return 0

def cutTheSticks(arr):
    count = 0
    myList = []
    while len(arr)>0:
        minval = min(arr)
        for i in range(0, len(arr)):
            arr[i] = arr[i]-minval
            count += 1
        while 0 in arr:
            arr.remove(0)
        myList.append(count)
        count = 0
    return myList

def primeValidate(arr):
    myList = []
    for i in arr:
        for j in range(2,i):
            if i == 1:
                myList.append("T")
                break
            elif (i%j) == 0:
                myList.append("F")
                break
        else:
            myList.append("T")
    return myList
            
def chocolateFeast(n, c, m):
    num = n//c
    b = True
    total = num
    temCount = 0
    while b:
        if num >= m:
            num = num-m
            total += 1
            temCount +=1
        else:
            num += temCount
            temCount = 0
            if num < m:
                b = False
    return total

def serviceLane(n, cases):
    for x,y in cases:
        print(min(n[x:y+1]))
    return [min(n[x:y+1]) for x,y in cases]

def repeatedString(s, n):
    return (s.count("a") * (n// len(s)) + s[:n % len(s)].count("a"))

def jumpingAvoidCloud(c):
    n = len(c)
    current = 0
    end = n - 1
    jumps = 0
    while current < end:
        if((current + 2) <= end) and (c[current + 2] == 0):
            current += 2
            jumps += 1
        elif c[current + 1] == 0:
            current += 1
            jumps += 1
    print(jumps)
    
      
def equalizeArray(arr):
    count=0
    G={}
    for i in arr:
        if i not in G:
            for j in arr:
                if i==j:
                    count=count+1
            G[i]=count
            count=0
    print(G)        
    max_val=max(G, key=G.get)
    print(max_val)
    return (len(arr)-G[max_val])

def validation():
    d = {}
    with open("file.txt") as f:
        for line in f:
            print("line:",line)
            (key,val) = line.split()
            d[key] = val
    print(d)

def taumBday(b, w, bc, wc, z):
    c = 0
    if bc > wc+z:
        c = (b+w)*wc + b*z
    elif wc > bc+z:
        c = (b+w)*bc + w*z
    else:
        c = b * bc + w * wc
    return c

    