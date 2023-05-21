import numpy as np


def gf2x_mod_mul_sparse(APosOnes, BPosOnes, INVALID_POS_VALUE):
    sizeA = APosOnes.shape[0]
    sizeB = BPosOnes.shape[0]
    sizeR = sizeA * sizeB
    RPosOnesTemp = np.ones((sizeA, sizeB)) * INVALID_POS_VALUE

    for i in range(sizeA):
        RPosOnesTemp[i, :] = (APosOnes[i] + BPosOnes) % INVALID_POS_VALUE
        RPosOnesTemp[i, BPosOnes == INVALID_POS_VALUE] = INVALID_POS_VALUE

    RPosOnesTemp = np.sort(RPosOnesTemp.reshape(-1))
    RPosOnes = np.ones(sizeR) * INVALID_POS_VALUE

    lastPlaced = 0
    pos = 0
    while pos < sizeR and RPosOnesTemp[pos] != INVALID_POS_VALUE:
        val = RPosOnesTemp[pos]
        pos += 1
        count = 1
        while pos < sizeR and RPosOnesTemp[pos] == val:
            pos += 1
            count += 1

        if count % 2 == 1:
            RPosOnes[lastPlaced] = val
            lastPlaced += 1
    return RPosOnes


def gf2x_mod_add_sparse(APosOnes, BPosOnes, sizeR, INVALID_POS_VALUE):
    sizeA = APosOnes.shape[0]
    sizeB = BPosOnes.shape[0]
    RPosOnes = np.ones(sizeR) * INVALID_POS_VALUE

    idxA = 0
    idxB = 0
    idxR = 0

    while idxA < sizeA and idxB < sizeB and APosOnes[idxA] != INVALID_POS_VALUE and BPosOnes[idxB] != INVALID_POS_VALUE:
        if APosOnes[idxA] == BPosOnes[idxB]:
            idxA += 1
            idxB += 1
        else:
            if APosOnes[idxA] < BPosOnes[idxB]:
                RPosOnes[idxR] = APosOnes[idxA]
                idxA += 1
            else:
                RPosOnes[idxR] = BPosOnes[idxB]
                idxB += 1
            idxR += 1

    while idxA < sizeA and APosOnes[idxA] != INVALID_POS_VALUE:
        RPosOnes[idxR] = APosOnes[idxA]
        idxA += 1
        idxR += 1

    while idxB < sizeB and BPosOnes[idxB] != INVALID_POS_VALUE:
        RPosOnes[idxR] = BPosOnes[idxB]
        idxB += 1
        idxR += 1

    while idxR < sizeR:
        RPosOnes[idxR] = INVALID_POS_VALUE
        idxR += 1

    return RPosOnes


def left_bit_shift(A):
    A = np.roll(A, -1)
    A[-1] = 0
    return A


def rotate_bit_left(A):
    return np.roll(A, -1)


def rotate_bit_right(A):
    return np.roll(A, 1)


def gf2x_add(A, B):
    return A ^ B


def gf2x_mod_add(A, B):
    return A ^ B


def gf2x_mod_mul(A, B):
    res = np.zeros(A.shape[0], dtype=bool)
    for i in range(A.shape[0]):
        if A[-i - 1] == 1:
            res = gf2x_mod_add(res, np.roll(B, -i))

    return res


def gf2x_swap(A, B):
    return B.copy(), A.copy()


def gf2x_mod_inverse(Ln0Dense):
    ln0Size = Ln0Dense.shape[0]
    u = np.zeros(ln0Size, dtype=bool)
    v = np.zeros(ln0Size, dtype=bool)
    s = np.zeros(ln0Size+1, dtype=bool)
    r = np.zeros(ln0Size+1, dtype=bool)

    u[-1] = 1
    s[-1] = 1
    s[0] = 1
    r[1:] = Ln0Dense.copy()
    delta = 0

    for i in range(2*ln0Size):
        if r[0] == 0:
            r = left_bit_shift(r)
            u = rotate_bit_left(u)
            delta += 1
        else:
            if s[0] == 1:
                s = gf2x_add(s, r)
                v = gf2x_mod_add(v, u)

            s = left_bit_shift(s)

            if delta == 0:
                r, s = s, r
                u, v = v, u
                u = rotate_bit_left(u)
                delta = 1
            else:
                u = rotate_bit_right(u)
                delta -= 1
    return u


def gf2x_mod_mul_dense_to_sparse(Dense, Sparse, INVALID_POS_VALUE):
    aux = np.zeros(2*Dense.shape[0], dtype=bool)
    res = np.zeros(2*Dense.shape[0], dtype=bool)

    aux[-Dense.shape[0]:] = Dense.copy()
    res[-Dense.shape[0]:] = Dense.copy()

    if Sparse[0] != INVALID_POS_VALUE:
        aux = np.roll(aux, -Sparse[0])
        res = np.roll(res, -Sparse[0])

        for i in range(1, Sparse.shape[0]):
            if Sparse[i] != INVALID_POS_VALUE:
                aux = np.roll(aux, -(Sparse[i] - Sparse[i-1]))
                res = gf2x_add(res, aux)

    return gf2x_add(res[-Dense.shape[0]:], res[:Dense.shape[0]])


def gf2x_transpose(A):
    return np.roll(A, 1)[::-1]

