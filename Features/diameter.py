#Numeric python library
import numpy as np

from itertools import combinations

def diamater(contour):
    dist_val = []
    for pair in combinations(contour,2):
        dist = np.linalg.norm(pair[1]-pair[0])
        dist_val.append(dist)
    max_dist = np.max(dist_val)
    return(max_dist)
