import numpy

a = numpy.empty((2,5), dtype=float)

#print(a)

a[0][1] = 222222

#print(a)



kernels = numpy.empty((7,7), dtype=float)
#print(kernels)


train_data = [1,4,6,1,4,7,3]


for x1, x1_ in enumerate(train_data):
    for x2, x2_ in enumerate(train_data):
         print(x1,x2)
         kernels[x1][x2] = x1_

print(kernels)
