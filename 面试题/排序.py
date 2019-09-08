# # 递归调用的思想
# def quickSort(num, l, r):
#     if l >= r:  # 如果只有一个数字时，结束递归
#         return
#     flag = l  # 是L 不是1
#     for i in range(l + 1, r + 1):  # 以第一个数字作为基准，从第二个数开始比较
#         if num[flag] > num[i]:
#             tmp = num[i]
#             del num[i]
#             num.insert(flag, tmp)
#             flag += 1
#     quickSort(num, l, flag - 1)  # 将基准的 前后部分分别递归排序
#     quickSort(num, flag + 1, r)
#
#
# num = [1, -9, 5, 7, 9, 3, 2, 8]
#
# quickSort(num, 0, 7)


def quicksort(l,start,end):

    left = start
    right = end
    # 结束递归的条件，最终相比较需要列表里len的长度为1才能结束
    if left < right:
        key = l[left]
        # 左右指针不重合的情况
        while left < right:
            # 如果列表右边的数,比基准数大或相等,则前移一位直到有比基准数小的数出现
            while left < right and l[right] >= key:
                right -= 1
            l[left],l[right] = l[right],l[left]
            # 如果列表右边的数，比基准数小或者相等，则交换位置
            while left < right and l[left] <= key:
                left += 1
            l[left],l[right] = l[right],l[left]
        quicksort(l,start,left - 1)
        quicksort(l,right + 1,end)
    return l


l = [3,9,2,2,1,6,5,5,4,9,8,3]
print(quicksort(l,start=0,end=len(l)-1))


