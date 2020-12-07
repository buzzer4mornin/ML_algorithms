import numpy as np


STAR = [[1, 1, 1, 1, 1],\
        [1, 1, 2, 1, 1],\
        [1, 2, 3, 2, 1],\
        [1, 1, 2, 1, 1],\
        [1, 1, 1, 1, 1,]]
STAR = np.asarray(STAR)


# Initialize STACK to be an empty array the same size as STAR
STACK = np.zeros(STAR.shape, np.float_)
print(STACK)

# A function that appends STAR to STACK
def add_Star(star):
    out = np.dstack((STACK,star))

    return out

# Call the function
STACK = add_Star(STAR)

#print(STACK)
