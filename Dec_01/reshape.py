import numpy as np
from sklearn.utils.extmath import weighted_mode
import collections


a = np.array([2,1,3,5,6])
b = np.array([4,5,1,5,7])

a = a.reshape(-1,1)
b = b.reshape(-1,1)

c = np.c_[a,b]
#print(c)



result=np.array([[0,3,0,3],[3,0,3,0]])

'''for row in result:
    row = np.delete(row,0)
    a = weighted_mode(row, np.full(len(row),1))
    print(int(a[0]))'''

a = [2,4,5,6,7,6,7]
c = collections.Counter(a)
print(c)