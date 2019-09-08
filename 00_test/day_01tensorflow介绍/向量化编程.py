# 由于在进行计算的时候，最好不要使用for循环去进行计算，因为有Numpy可以进行更加快速的向量化计算。


import numpy as np
import time
a = np.random.rand(100000)
b = np.random.rand(100000)

# 第一种for 循环
c = 0
start = time.time()
for i in range(100000):
    c += a[i]*b[i]
print(c)
end = time.time()

print("计算所用时间%s " % str(1000*(end-start)) + "ms")

# 向量化运算
start = time.time()
c = np.dot(a, b)
print(c)
end = time.time()
print("计算所用时间%s " % str(1000*(end-start)) + "ms")








