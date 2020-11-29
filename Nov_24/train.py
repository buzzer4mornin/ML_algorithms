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

'''if args.plot:
    # If you want the plotting to work (not required for ReCodEx), compute the `test_predictions`.
    test_predictions = my_test

    plt.plot(train_data, train_targets, "bo", label="Train targets")
    plt.plot(test_data, test_targets, "ro", label="Test targets")
    plt.plot(test_data, test_predictions, "g-", label="Predictions")
    plt.legend()
    if args.plot is True: plt.show()
    else: plt.savefig(args.plot, transparent=True, bbox_inches="tight")'''