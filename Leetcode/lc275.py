
import os

class Solution(object):
    def hIndex(self, citations):
        """
        :type citations: List[int]
        :rtype: int
        """
        count = 0
        i = len(citations) - 1
        while i >= 0:
            if citations[i] > count:
                count += 1
            elif citations[i] < count:
                break
            i -= 1
        return count


    def hIndex_BS(self, citations):
        """
        :type citations: List[int]
        :rtype: int
        """
        l, r = 0, len(citations) - 1
        while l <= r:
            mid = l + (r - l)//2
            num = len(citations) - mid
            if citations[mid] == num:
                return num
            elif citations[mid] > num:
                r = mid - 1
            else:
                l = mid + 1
        return len(citations) - l

s = Solution()
b = [0]
print b
print s.hIndex_BS(b)
c = [0, 1, 3, 5, 6]
print c
print s.hIndex_BS(c)
d = [0, 1, 2, 3, 4]
print d
print s.hIndex_BS(d)
e = [0, 1, 2, 3, 4, 5]
print e
print s.hIndex_BS(e)

f = [ 0, 1, 2, 3, 4, 5, 6 ]
print f
print s.hIndex_BS(f)