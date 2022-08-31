import numpy as np
import sys

d = np.loadtxt(sys.argv[1])
print(np.mean(d[:,2], axis=0))
