import numpy as np
import random
Q_init = 0
dist_comb = [(1, 1)]
action_space = np.array([[0, 1, 2, 3, 4]])

print([[[Q_init]] * action_space.size for i in range(action_space.size)] )

#calculate the power of n of a number
def square(x, n):
    x = x**n
    return x
print(square(2, 3))