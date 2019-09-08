'''
写一个小程序，将形状位MxN的2维矩阵顺时针旋转K位
例如：将下面左边的 M=3 N=4 的矩阵，旋转 K=1位，得到右边3x4的矩阵
[[1, 2, 3, 4],                 [[10, 1, 2, 3],
[10, 11, 12, 5],     >>>>       [9, 12, 11, 4],
[9, 8, 7, 6]]                   [8, 7, 6, 5]]

'''


import numpy as np
# MxN 3x4
matrix = [[1, 2, 3, 4],
          [10, 11, 12, 5],
          [9, 8, 7, 6]]
matrix_arr = np.array(matrix)
M = matrix_arr.shape[0]
N = matrix_arr.shape[1]

# 第一行元素添加到列表
# list_1 = []
a = lambda M, N: M if M < N else N
b = a(M, N)

for k in range(b//2 + 1):
    # 第k行元素添加
    list_1 = matrix[k][:-1]
    print(list_1)
    list_1 = matrix[k][k:-1-k]
    print('行数', list_1)
    # 最右侧元素添加到列表（不含首尾两行）
    for num_last in range(1+k, M-1-k):
        list_1.append(matrix[num_last][N-1-k])

    # 最后一行元素添加到列表
    for i in matrix[M-1-k]:
        list_1.append(i)

    # 最左侧元素添加（不含首尾元素）
    for num_first in range(1+k, M-1-k):
        list_1.append(matrix[num_first][0])
        print(list_1)
print(list_1)






