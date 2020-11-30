import numpy as np
generator = np.random.RandomState(42)

betas = np.zeros(10)
permutation = generator.permutation(10)




'''K = np.empty((train_data.shape[0], train_data.shape[0]), dtype=np.float)
for i in range(0, train_data.shape[0]):
    for j in range(0, train_data.shape[0]):
        xi = train_data[i]
        xj = train_data[j]
        K[i][j] = kernel(xi, xj)[0]'''


'''my_test = []
for i in range(len(test_data)):
    sums = 0
    for j in range(len(betas)):
        sums += betas[j] * kernel(test_data[i], test_data[j])[0]
    my_test.append(sums)'''


gradient_components = 0
for i in range(10):
    print(i)
    gradient_components += 1
    if gradient_components == 5:
        print(100+i)