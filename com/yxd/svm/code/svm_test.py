import numpy as np
arrays = [np.random.randn(3, 4) for _ in range(10)]
print arrays.__len__()


xx, yy = np.meshgrid(np.linspace(-4, 5, 500), np.linspace(-4, 5, 500))

print xx.ravel()
