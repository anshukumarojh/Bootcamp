#!/usr/bin/env python
# coding: utf-8

# # 1. Convert a string to a zigzag pattern on a given number of rows and then read it row by row.

# In[3]:


def convert_to_zigzag(s: str, numRows: int) -> str:
    if numRows <= 1 or numRows >= len(s):
        return s

    rows = [''] * numRows
    current_row = 0
    going_down = False

    for char in s:
        rows[current_row] += char
        
        if current_row == 0:
            going_down = True
        elif current_row == numRows - 1:
            going_down = False
        current_row += 1 if going_down else -1

    for i in range(numRows):
        print(f"Row {i}: {rows[i]}")

    return ''.join(rows)

s = "PAYPALISHIRING"
numRows = 3
output = convert_to_zigzag(s, numRows)
print("Final output:", output)   


# # 2. Implement a method to perform basic string compression using the counts of repeated characters.

# In[4]:


def string_compression(s):
    result = ""
    if not s:
        return result
    char_count = 1   
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            char_count += 1
        else:
            result += s[i - 1] + str(char_count)
            char_count = 1
    result += s[-1] + str(char_count)
    return result

string = "aabcccccaaa"
print(string_compression(string))


# # 3. Write a function that generates all possible permutations of a given string.
#    - Input: "ABC"
# 

# In[5]:


import itertools

def permute(strs: str) -> list:
   res = [''.join(p) for p in itertools.permutations(strs)]
   return res


strs = "ABC"


result = permute(strs)
print(result)


# # 4. Given an array of strings, group anagrams together.

# In[6]:


from typing import List
from collections import defaultdict

l = ["eat", "tea", "tan", "ate", "nat", "bat"]

def groupAnagrams(l: List[str]) -> List[List[str]]:
    d = defaultdict(list)
    for s in l:
        k = ''.join(sorted(s))
        d[k].append(s)
    return list(d.values())

result = groupAnagrams(l)
print(result)


# 
# # 5. Write a function to multiply two large numbers represented as strings.

# In[7]:


def multiply(num1: str, num2: str) -> str:
   if num1 == "0" or num2 == "0":
       return "0"
   m, n = len(num1), len(num2)
   arr = [0] * (m + n)
   for i in range(m - 1, -1, -1):
       a = int(num1[i])
       for j in range(n - 1, -1, -1):
           b = int(num2[j])
           arr[i + j + 1] += a * b
   for i in range(m + n - 1, 0, -1):
       arr[i - 1] += arr[i] // 10
       arr[i] %= 10
   i = 0 if arr[0] else 1
   return "".join(str(x) for x in arr[i:])

num1 = "123"
num2 = "456"

result = multiply(num1, num2)
print(result)


# # 6. Given two strings, check if one is a rotation of the other using only one call to a string method.

# In[8]:


def is_rotation(str1: str, str2: str) -> bool:
   return len(str1) == len(str2) and str2 in (str1 + str1)

str1 = "waterbottle"
str2 = "erbottlewat"

result = is_rotation(str1, str2)
print(result)


# # 7. Given a string containing just the characters (, ), {, }, [, and ], determine if the input string is valid.

# In[9]:


def isValid(parentheses: str) -> bool:
   stack = []  
   brackets = {')': '(', '}': '{', ']': '['}   
   for char in parentheses:
       if char in brackets.values():  
           stack.append(char)
       elif char in brackets: 
           if stack and stack[-1] == brackets[char]:
               stack.pop()
           else:  
               return False
       else:  
           return False

   return not stack   

parentheses = "()[]{}"

result = isValid(parentheses)
print(result)


# # 8. Implement the function atoi which converts a string to an integer

# In[10]:


def atoi(s: str) -> int:
     
    s = s.lstrip()
    
    if not s:
        return 0

    
    sign = 1   
    result = 0
    index = 0
    n = len(s)

     
    if s[index] == '-':
        sign = -1
        index += 1
    elif s[index] == '+':
        index += 1

     
    while index < n and s[index].isdigit():
        digit = int(s[index])
        
         
        if result > (2**31 - 1) // 10 or (result == (2**31 - 1) // 10 and digit > 7):
            return 2**31 - 1 if sign == 1 else -2**31

        result = result * 10 + digit
        index += 1

    return sign * result

 
input_str = "4193 with words"
output = atoi(input_str)
print(output)   


# # 9. Write a function that generates the nth term of the "count and say" sequence.

# In[11]:


def countAndSay(n: int) -> str:
   s = '1'
   for _ in range(n - 1):
       i = 0
       t = []
       while i < len(s):
           j = i
           while j < len(s) and s[j] == s[i]:
               j += 1
           t.append(str(j - i))
           t.append(str(s[i]))
           i = j
       s = ''.join(t)
   return s

n = 5

result = countAndSay(n)
print(result)


# # 10. Given two strings s and t, return the minimum window in s which will contain all the characters in t.

# In[12]:


s = "ADOBECODEBANC"
t = "ABC"

from collections import Counter
from math import inf

def minWindow(s: str, t: str) -> str:
    need = Counter(t)
    window = Counter()
    cnt, j, k, mi = 0, 0, -1, inf
    for i, c in enumerate(s):
        window[c] += 1
        if need[c] >= window[c]:
            cnt += 1
        while cnt == len(t):
            if i - j + 1 < mi:
                mi = i - j + 1
                k = j
            if need[s[j]] >= window[s[j]]:
                cnt -= 1
            window[s[j]] -= 1
            j += 1
    return '' if k < 0 else s[k : k + mi]

result = minWindow(s,t)
print(result) 


# # 11. Given a string, find the length of the longest substring without repeating characters

# In[13]:


s = "abcabcbb"
def lengthOfLongestSubstring(s: str) -> int:
    ss = set()
    ans = i = 0
    for j, c in enumerate(s):
        while c in ss:
            ss.remove(s[i])
            i += 1
        ss.add(c)
        ans = max(ans, j - i + 1)
    return ans
result = lengthOfLongestSubstring(s)
print(result) 


# # 12. Given three strings s1, s2, and s3, determine if s3 is formed by the interleaving of s1 and s2.

# In[14]:


s1 = "aabcc"
s2 = "dbbca"
s3 = "aadbbcbcac"

 
def isInterleave(s1: str, s2: str, s3: str) -> bool:
 
    def dfs(i: int, j: int) -> bool:
        if i >= m and j >= n:
            return True
        k = i + j
        if i < m and s1[i] == s3[k] and dfs(i + 1, j):
            return True
        if j < n and s2[j] == s3[k] and dfs(i, j + 1):
            return True
        return False

    m, n = len(s1), len(s2)
    if m + n != len(s3):
        return False
    return dfs(0, 0)
result = isInterleave(s1,s2,s3)
print(result) 


# # 13. Write a function to convert a Roman numeral to an integer.

# In[15]:


S="MCMXCIV"

roman = {'I':1,'V':5,'X':10,'L':50,'C':100,'D':500,'M':1000}

def romanToInt(S: str) -> int:
    ans = 0
    for i in range(len(S)-1,-1,-1):
        num = roman[S[i]]
        if 4 * num < ans: ans -= num
        else: ans += num
    return ans

result = romanToInt(S)
print(result) 


# # 14. Find the longest common substring between two strings.

# In[16]:


str1 = "ABABC"
str2 = "BABCA"

def longest_common_subsequence(str1, str2):
    n, m = len(str1), len(str2)
     
    dp = [[0 for j in range(m + 1)] for i in range(n + 1)]

    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
             
            if str1[i - 1] == str2[j - 1]:
                 
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                 
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

     
    letters = dp[n][m]
    
    result = ['' for i in range(letters)]
    i = n
    j = m

    while (i != 0) and (j != 0):
         
        if str1[i - 1] == str2[j - 1]:
            letters -= 1
            result[letters] = str1[i - 1]
            j -= 1
            i -= 1
        elif dp[i - 1][j] < dp[i][j - 1]:
            j -= 1
        else:
            i -= 1

    return ''.join(result)

result = longest_common_subsequence(str1, str2)
print(result)  


# # 15. Given a string s and a dictionary of words, check if s can be segmented into a space-separated sequence of one or more dictionary words

# In[17]:


s = "leetcode"
wordDict = ["leet", "code"]

def wordBreak(s: str, wordDict: List[str]) -> bool:
    dp ={"":True}
    def valid(curr):
        if curr in dp:
            return dp[curr]

        for i in range(len(curr)-1,-1,-1):
            if curr[i:] in wordDict:
                dp[curr[i:]] = valid(curr[:i])
                if dp[curr[i:]]:
                    return True
        dp[curr]=False
        return False

    return valid(s)

result = wordBreak(s, wordDict)
print(result)


# # 16. Remove the minimum number of invalid parentheses to make the input string valid. Return all possible results.

# In[19]:


s="()())()"

def removeInvalidParentheses(s: str) -> List[str]:
    def dfs(i, l, r, lcnt, rcnt, t):
        if i == n:
            if l == 0 and r == 0:
                ans.add(t)
            return
        if n - i < l + r or lcnt < rcnt:
            return
        if s[i] == '(' and l:
            dfs(i + 1, l - 1, r, lcnt, rcnt, t)
        elif s[i] == ')' and r:
            dfs(i + 1, l, r - 1, lcnt, rcnt, t)
        dfs(i + 1, l, r, lcnt + (s[i] == '('), rcnt + (s[i] == ')'), t + s[i])

    l = r = 0
    for c in s:
        if c == '(':
            l += 1
        elif c == ')':
            if l:
                l -= 1
            else:
                r += 1
    ans = set()
    n = len(s)
    dfs(0, l, r, 0, 0, '')
    return list(ans)

result = removeInvalidParentheses(s)
print(result)


# In[ ]:




