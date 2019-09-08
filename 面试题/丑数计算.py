
# def isUgly(self, num):
#         """
#         :type num: int
#         :rtype: bool
#         """
#         if num < 1:
#             return False
#         while num % 2 == 0 or num % 3 == 0 or num % 5 == 0:
#             if num % 2 == 0:
#                 num //= 2
#             elif num % 3 == 0:
#                 num //= 3
#             elif num % 5 == 0:
#                 num //= 5
#         if num != 1 and num != 2 and num != 3 and num != 5:
#             return False
#         else:
#             return True

'''
题目：我们把只含有因子2、3、5的数称为丑数。
例如6、8都是丑数，而14不是丑数，因为它含有因子7.
通常也把1当做丑数。编程找出1500以内的全部丑数。
注意：使用的算法效率应尽量高

后面的丑数肯定是已存在的丑数乘以2、3或者5，
找到比现有丑数大的且是最小的丑数作为下一个丑数（如何找是关键）。
用2分别从现有丑数中从前往后乘以丑数，找到第一个大于当前所有丑数的值以及位置，3、5同样如此，
再把他们相乘之后的结果做对比，取最小的。下次将从上一次的位置开始往下找，这样将不会出现冗余。
'''

# 前几位丑数 1，2，3，4，5，6，8，9，10
# import math
# a = math.inf
# # print(a)
# # print(type(a))

import time

def get_ugly_number():
    list_null = []
    start = time.time()
    for i in range(10):
        for j in range(10):
            for k in range(10):
                x = pow(2, i)*pow(3, j)*pow(5, k)  # 0.0012402534484863281
                # x = (2**i)*(3**j)*(5**k)  #0.0024423599243164062
                list_null.append(x)
    # print(list_null)
    a = sorted(list_null)
    print(a)
    stop = time.time()
    interval = stop - start
    print(interval)
if __name__ == '__main__':
    get_ugly_number()









