import numpy as np

def RMSE(pred, DOA):
    pred = np.sort(pred)
    DOA = np.sort(DOA)
    error = (pred - DOA) ** 2
    return np.mean(error)