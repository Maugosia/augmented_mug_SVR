import numpy as np


def mat_rx(fi):
    mat = np.array([
        [1, 0, 0, 0],
        [0, np.cos(fi), -np.sin(fi), 0],
        [0, np.sin(fi), np.cos(fi), 0],
        [0, 0, 0, 1]
    ])
    return mat


def mat_ry(fi):
    mat = np.array([
        [np.cos(fi), 0, np.sin(fi), 0],
        [0, 1, 0, 0],
        [-np.sin(fi), 0, np.cos(fi), 0],
        [0, 0, 0, 1]
    ])
    return mat


def mat_rz(fi):
    mat = np.array([
        [np.cos(fi), -np.sin(fi), 0, 0],
        [np.sin(fi), np.cos(fi), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    return mat