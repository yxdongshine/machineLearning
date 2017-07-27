import numpy as np
arrays = [np.random.randn(3, 4) for _ in range(10)]
print arrays.__len__()