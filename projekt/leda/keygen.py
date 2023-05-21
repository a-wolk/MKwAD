from random_utils import TRNG, RNG
from gf2x import gf2x_mod_mul_sparse, gf2x_mod_add_sparse, gf2x_mod_inverse, gf2x_mod_mul_dense_to_sparse, gf2x_transpose
import numpy as np
from utils import pack_bytes


def randCirculantSparseBlock(countOnes, rng, maxIndex):
    r = np.zeros(countOnes)

    placedOnes = 0
    duplicated = False
    while placedOnes < countOnes:
        p = rng.randrange(maxIndex)
        duplicated = False
        for j in range(placedOnes):
            if r[j] == p:
                duplicated = True
        if not duplicated:
            r[placedOnes] = p
            placedOnes += 1

    return r


def generateHPosOnes(rng, leda):
    HT = np.zeros((leda.N0, leda.DV), dtype=np.int32)
    H = np.zeros((leda.N0, leda.DV), dtype=np.int32)

    for i in range(leda.N0):
        HT[i, :] = randCirculantSparseBlock(leda.DV, rng, leda.P)
        H[i, :] = (leda.P - HT[i, :]) % leda.P

    return H, HT


def generateQPosOnes(rng, leda):
    Q = np.zeros((leda.N0, leda.M))

    for i in range(leda.N0):
        placed_ones = 0
        for j in range(leda.N0):
            countOnes = leda.q_block_weight(i, j)
            Q[i, placed_ones:placed_ones+countOnes] = randCirculantSparseBlock(countOnes, rng, leda.P)
            placed_ones += countOnes

    return Q


def calcLPosOnes(HPosOnes, QPosOnes, leda):
    LPosOnes = np.ones((leda.N0, leda.DV * leda.M), dtype=np.int32) * leda.P
    processedQOnes = [0] * leda.N0
    for i in range(leda.N0):
        for j in range(leda.N0):
            countOnesQji = leda.q_block_weight(j, i)
            mul = gf2x_mod_mul_sparse(HPosOnes[j, :], QPosOnes[j, processedQOnes[j]:processedQOnes[j] + countOnesQji], leda.P)
            prod = gf2x_mod_add_sparse(LPosOnes[i, :], mul, leda.DV * leda.M, leda.P)
            LPosOnes[i, :] = prod
            processedQOnes[j] += countOnesQji
    return LPosOnes


def sparseToDense(sparse, leda):
    dense = np.zeros(leda.P, dtype=bool)
    validPos = filter(lambda x: x != leda.INVALID_POS_VALUE, sparse.tolist())
    validReversedPos = np.array(list(map(lambda x: leda.P - x - 1, validPos)), dtype=np.int32)
    dense[validReversedPos] = 1
    return dense


def keygen(leda):
    #TRNG
    rndPrivateMatricesSeed = TRNG(leda.TRNG_BYTE_LENGTH)
    rng = RNG(rndPrivateMatricesSeed)

    # Generate H, Q
    HPosOnes, _ = generateHPosOnes(rng, leda)
    QPosOnes = generateQPosOnes(rng, leda)

    # L
    LPosOnes = calcLPosOnes(HPosOnes, QPosOnes, leda)
    Ln0Dense = sparseToDense(LPosOnes[leda.N0-1, :], leda)
    Ln0Inv = gf2x_mod_inverse(Ln0Dense)

    M = np.zeros(leda.K, dtype=bool)
    for i in range(leda.N0-1):
        M[leda.getBlockSlice(i)] = gf2x_mod_mul_dense_to_sparse(Ln0Inv, LPosOnes[i, :], leda)
        M[leda.getBlockSlice(i)] = gf2x_transpose(leda.getBlock(M, i))

    return rndPrivateMatricesSeed.to_bytes(leda.TRNG_BYTE_LENGTH, "big"), pack_bytes(M)