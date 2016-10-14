
import os

class Solution(object):
    def hIndex(self, citations):
        """
        :type citations: List[int]
        :rtype: int
        """
        citations.sort()
        print citations
        count = 0
        i = len(citations) - 1
        while i >= 0:
            if citations[i] > count:
                count += 1
            elif citations[i] < count:
                break
            i -= 1
        return count


s = Solution()
b = [0]
print b
print s.hIndex(b)
c = [3, 0, 6, 1, 5]
print c
print s.hIndex(c)
d = [ 3, 3, 3, 6, 4]
print d
print s.hIndex(d)
