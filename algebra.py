import numpy as np
import scipy
import ldpc

def gauss(X):
    m, n = X.shape

    full_ind = np.arange(m)
    w_ind = np.zeros(n, dtype=bool)
    A = np.copy(X)

    for i in range(0, m):
        if np.sum(A[i:, ]) == 0:
            raise ldpc.DegenerateMatrixError
            break

        ones_ind = np.argwhere(A[i:, ])
        ind = np.argmin(ones_ind[:, 1])
        col = np.min(ones_ind[:, 1])
        w_ind[col] = True
        first_in = ones_ind[ind, 0] + i

        t = np.copy(A[first_in, :])
        A[first_in, :] = np.copy(A[i, :])
        A[i, :] = t

        w = (A[:, col] == 1) & (full_ind != i)
        A[w, i:] = (A[w, i:] + A[i, i:]) % 2
        A[:i, col] = 0

    return A, w_ind


def trimmer(a, trim):
    t_a = np.copy(a)
    t_a[t_a < trim] = trim
    t_a[t_a > (1 - trim)] = 1 - trim
    return t_a
