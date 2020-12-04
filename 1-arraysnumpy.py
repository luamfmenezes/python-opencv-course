import numpy as np

# comments on python

mylist = np.array([1, 2, 3])

arange = np.arange(0,10,2);

zeros = np.zeros((5,5))

ones = np.ones((5,5))

np.random.seed(101);

arr1 = np.random.randint(0,100,10)

# max, max_index and average
print(arr1.max());
print(arr1.argmax());

# min, min_index, average
print(arr1.min());
print(arr1.argmin());
print(arr1.min());

# return the shape of the array
print(arr1.shape)
reshapedArray = arr1.reshape(2,5)
print(reshapedArray)

mat = np.arange(0,100).reshape(10,10)

print(mat)

print(mat[0,2])

print(mat[2,:])
print(mat[2])

print(mat[:,2])

print(mat[2:4,2:5])

mynewmat = mat.copy()

mynewmat[0:3,0:3] = 0


print(mynewmat)