import numpy as np


def length(v):
    if v.ndim == 1:
        return np.sqrt(np.sum(v ** 2))
    elif v.ndim == 2:
        return np.array([length(a) for a in v])
    raise ValueError('A very specific bad thing happened.')


def dot_product(v, w):
    if v.ndim == 1:
        return np.sum(v * w)
    elif v.ndim == 2:
        return np.array([dot_product(a, b) for a, b in zip(v, w)])
    raise ValueError('A very specific bad thing happened.')


def angle(v, w):
    v = np.array(v)
    w = np.array(w)
    length_v = length(v)
    length_w = length(w)
    product = dot_product(v, w)
    cosx = product / (length_v * length_w)
    cosx = np.clip(cosx, -1.0, 1.0)
    rad = np.arccos(cosx) % (2 * np.pi)
    return rad.mean()


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
