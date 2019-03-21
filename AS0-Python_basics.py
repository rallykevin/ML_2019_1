#!/usr/bin/env python
# coding: utf-8

# # M2608.001300 기계학습 기초 및 전기정보 응용<br> Assignment 0: Python Basics

# ## Problem 1: Bubblesort
# 
# 아래 bubblesort 함수를 구현해보세요. 
# YOUR CODE COMES HERE 라는 주석이 있는 곳을 채우면 됩니다.

# In[1]:


def bubblesort(arr):
    # YOUR CODE COMES HERE
    for i in range(0, len(arr)):
        for j in range(0, len(arr)-i-1):
            if arr[j]>arr[j+1]:
                arr[j], arr[j+1]=arr[j+1],arr[j]
    return arr


# In[2]:


import random
array = [random.randint(0, 20) for _ in range(20)]
print(array)

array_sorted = bubblesort(array)
print(array_sorted)

print()
print('Q: Is the array sorted?')
print('A:', sorted(array) == array_sorted)


# ## Problem 2: Classes
# 
# Quicksort, bubblesort, insertionsort 를 아래 class의 instance method로 구현해 보세요. YOUR CODE COMES HERE 라는 주석이 있는 곳을 채우면 됩니다.

# In[5]:


class Sorter:
    def __init__(self, method):
        self.method = method
        
    @staticmethod
    def of(method):
        return Sorter(method)
        
    def sort(self, arr):
        if self.method == 'quicksort':
            return self.quicksort(arr)
        
        elif self.method == 'bubblesort':
            return self.bubblesort(arr)
        
        elif self.method == 'insertionsort':
            return self.insertionsort(arr)
        
        else:
            raise ValueError('Unknown method: %s' % method)

    def quicksort(self, arr):
        # YOUR CODE COMES HERE
        self.arr=arr
        if len(self.arr)>1: #if it is 1, no need to sort
            pivot=self.arr[len(arr)-1]
            left, middle, right = [],[],[]
            for i in range(len(self.arr)-1):
                if self.arr[i]<pivot:
                    left.append(self.arr[i])
                elif self.arr[i]>pivot:
                    right.append(self.arr[i])
                else:
                    middle.append(self.arr[i])
            middle.append(pivot)
            return self.quicksort(left)+middle+self.quicksort(right)
        else:
            return self.arr
    
    def bubblesort(self, arr):
        # YOUR CODE COMES HERE
        self.arr=arr
        copy=self.arr[:] #copy=arr would change arr afterwards, and cause problems
        for i in range(len(copy)):
            for j in range(0, len(copy)-i-1):
                if copy[j]>copy[j+1]:
                    copy[j], copy[j+1]=copy[j+1],copy[j]
        return copy
    
    def insertionsort(self, arr):
        # YOUR CODE COMES HERE
        self.arr=arr
        copy=self.arr[:] #copy=arr would change arr afterwards, and cause problems
        for i in range(1, len(copy)):
            val = copy[i]
            j = i
            while j > 0 and copy[j-1] > val:
                copy[j] = copy[j-1]
                j -= 1
            copy[j] = val
        return copy


# In[6]:


array = [random.randint(0, 20) for _ in range(20)]

algorithms = ['quicksort', 'bubblesort', 'insertionsort']
for algorithm in algorithms:
    sorter = Sorter.of(algorithm)
    array_sorted = sorter.sort(array)
    print('%s sorted? %s' % (algorithm, sorted(array) == array_sorted))


# In[ ]:




