#!/usr/bin/env python
# coding: utf-8

# # 1. Remove all occurrences of a specific value from a list

# In[1]:


def remove_occurrences(lst, value):
    lst.remove(value)
    return lst

my_list = [1, 2, 3, 4, 3, 5]
value_to_remove = 3
my_list = [x for x in my_list if x != value_to_remove]
print(my_list)


# # 2. Concatenate a list of strings into a single string separated by spaces

# In[2]:


def concatenate_strings(string_list):
    return ' '.join(string_list)

my_strings = ['Hello', 'world', 'this', 'is', 'Python']
result = concatenate_strings(my_strings)
print(result)


# # 3 :- Reverse a list of integers

# In[3]:


def reverse_list(lst):
    lst.reverse()
    return lst

my_list = [1, 2, 3, 4, 5]
reversed_list = reverse_list(my_list)
print(reversed_list)


# # 4 Sort a list of numbers in descending order

# In[7]:


l4 = [4,1,6,7,1,7,0,4]
l4.sort(reverse=True)
l4


# # 5 Combine two lists and remove duplicates
# 

# In[8]:


list1 = [1, 2, 3, 4, 5]
list2 = [3, 4, 5, 6, 7]
combined_list = list(set(list1 + list2))
combined_list


# # 6 Convert a tuple into a list and remove the first and last elements

# In[9]:


t = (3,8,5,1,9,4)
t1=list(t)[1:-1]
# tup = t1[1:-1]
t1


# # 7 .Given a list of tuples, extract all the first elements of each tuple into a separate list using tuple unpacking.

# In[10]:


ls = [(1,2),(3,4),(5,6)]
sep = [x for x, _ in ls]
sep


# # 8. Combine two tuples into a single tuple.

# In[11]:


tuple1 = (1, 2, 3)
tuple2 = (4, 5, 6)
comb_tuple = tuple1 + tuple2
comb_tuple


# # 9. Find the maximum and minimum values in a tuple of numbers using tuple methods.
# 

# In[12]:


num = (10, 20, 5, 40, 30)
maxi = max(num)
mini = min(num)
print(maxi)  
print(mini)


# # 10. Convert a tuple of strings into a single string, with each element separated by a comma, using tuple methods.

# In[13]:


s_tup = ("How", "are", "you")
comb_s = ', '.join(s_tup)
comb_s


# # 11. Given two sets, return the union of the two sets using a set operation.
# 

# In[15]:


set1 = {1, 2, 3}
set2 = {3, 4, 1, 5}
opr = set1.union(set2)
opr


# # 12. Find the intersection of two sets using a set method.

# In[17]:


inter_opr = set1.intersection(set2)
inter_opr


# # 13. Remove all elements from one set that are also present in another set using a set method.

# In[18]:


set1.intersection_update(set2)
set1


# # 14. Find the difference between two sets using a set method.

# In[19]:


diff_opr = set1.difference(set2)
diff_opr


# # 15. Convert a list of integers into a set and then back to a list to remove duplicates, using set operations.

# In[20]:


num_list = [1, 2, 3, 4, 5, 1, 2, 3]
num_set = set(num_list)
s = list(num_set)
s


# # 16. Merge two dictionaries into a single dictionary using a dictionary method.
# 

# In[21]:


dict1 = {'a': 1, 'b': 2}
dict2 = {'b': 3, 'c': 4}
merged_dict = {**dict1, **dict2}
merged_dict


# # 17. Extract all values from a dictionary into a list using a dictionary method.
# 

# In[22]:


my_dict = {'a': 1, 'b': 2, 'c': 3}
values_list = list(my_dict.values())
values_list


# In[ ]:




