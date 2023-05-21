import numpy as np

from random_utils import RNG
from keygen import generateHPosOnes, generateQPosOnes, calcLPosOnes
from gf2x import gf2x_transpose, gf2x_mod_mul_dense_to_sparse, gf2x_mod_add
from utils import unpack_bytes, remove_padding, bytes_to_bitarray, bitarray_to_int
from cw import constant_weight_to_binary


def bf_decoding(HT, QT, privateSyndrome, leda):
    currQ_pos = np.zeros(leda.M, dtype=np.int32)
    imax = 15 #ITERATIONS_MAX
    synd_corrt_vec = leda.SYND_TRESH_LOOKUP_TABLE

    out = np.zeros(leda.N0*leda.P, dtype=bool)
    while True:
        currSyndrome = privateSyndrome.copy()
        upc = np.zeros(leda.N0 * leda.P, dtype=np.int32)
        for i in range(leda.N0):
            valueIdx = np.arange(leda.P).reshape(1, -1)
            upc[i*leda.P + valueIdx] += (currSyndrome[-((HT[i, :].reshape(-1, 1) + valueIdx) % leda.P) - 1] == 1).sum(axis=0)

        syndrome_wt = currSyndrome.sum()
        min_idx = 0
        max_idx = len(synd_corrt_vec) - 1
        tresh_table_idx = (min_idx + max_idx) // 2
        while min_idx < max_idx:
            if synd_corrt_vec[tresh_table_idx][0] <= syndrome_wt:
                min_idx = tresh_table_idx + 1
            else:
                max_idx = tresh_table_idx - 1
            tresh_table_idx = (min_idx + max_idx) // 2
        corrt_syndrome_based = synd_corrt_vec[tresh_table_idx][1]

        for i in range(leda.N0):
            for j in range(leda.P):
                currQoneIdx = 0
                endQblockIdx = 0
                correlation = 0

                for blockIdx in range(leda.N0):
                    endQblockIdx += leda.q_block_weight(blockIdx, i)
                    while currQoneIdx < endQblockIdx:
                        currQ_pos[currQoneIdx] = (QT[i, currQoneIdx] + j) % leda.P + blockIdx * leda.P
                        correlation += upc[currQ_pos[currQoneIdx]]
                        currQoneIdx += 1

                if correlation > corrt_syndrome_based:
                    out[(i+1)*leda.P - j - 1] = not out[(i+1)*leda.P - j - 1]
                    for v in range(leda.M):
                        posFlip = currQ_pos[v] % leda.P
                        blockIndex = currQ_pos[v] // leda.P
                        syndromePosToFlip = (HT[blockIndex, :] + posFlip) % leda.P
                        privateSyndrome[-syndromePosToFlip - 1] = (1 - privateSyndrome[-syndromePosToFlip - 1])

        imax -= 1
        if imax == 0 or privateSyndrome.sum() == 0:
            break

    return out, privateSyndrome.sum() == 0


def decrypt_mceliece(ctx, sk, leda):
    rng = RNG(sk)
    H, HT = generateHPosOnes(rng, leda)
    Q = generateQPosOnes(rng, leda)
    L = calcLPosOnes(H, Q, leda)

    codewordPoly = ctx.copy()
    for i in range(leda.N0):
        codewordPoly[leda.getBlockSlice(i)] = gf2x_transpose(leda.getBlock(codewordPoly, i))

    privateSyndrome = np.zeros(leda.P, dtype=bool)
    for i in range(leda.N0):
        aux = gf2x_mod_mul_dense_to_sparse(leda.getBlock(codewordPoly, i), L[i, :], leda.P)
        privateSyndrome = gf2x_mod_add(aux, privateSyndrome)
    privateSyndrome = gf2x_transpose(privateSyndrome)

    QT = np.zeros(Q.shape, dtype=np.int32)
    transposed_ones_idx = np.zeros(leda.N0, dtype=np.uint32)
    for source_row_idx in range(leda.N0):
        currQoneIdx = 0
        endQblockIdx = 0
        for blockIdx in range(leda.N0):
            endQblockIdx += leda.q_block_weight(source_row_idx, blockIdx)
            while currQoneIdx < endQblockIdx:
                QT[blockIdx, transposed_ones_idx[blockIdx]] = (leda.P - Q[source_row_idx, currQoneIdx]) % leda.P
                transposed_ones_idx[blockIdx] += 1
                currQoneIdx += 1

    decoded_err, success = bf_decoding(HT, QT, privateSyndrome, leda)
    if not success:
        raise Exception("Decoding failed")

    correct_codeword = np.zeros(codewordPoly.shape, dtype=bool)
    for i in range(leda.N0):
        correct_codeword[leda.getBlockSlice(i)] = gf2x_mod_add(leda.getBlock(ctx, i), leda.getBlock(decoded_err, i))

    return correct_codeword, decoded_err


def decode(ctx, sk, leda):
    ctx = unpack_bytes(ctx)
    sk = int.from_bytes(sk, byteorder='big')

    codeword, err = decrypt_mceliece(ctx, sk, leda)

    yBufferBitLength = leda.CONSTANT_WEIGHT_ENCODED_DATA_ACTUAL_BIT_LENGTH + leda.K
    yBuffer = constant_weight_to_binary(err, yBufferBitLength, leda)
    for i in range(leda.N0 - 1):
        yBuffer[leda.CONSTANT_WEIGHT_ENCODED_DATA_ACTUAL_BIT_LENGTH:][leda.getBlockSlice(i)] = leda.getBlock(codeword, i)[::-1]

    hash = leda.HASH_FUNC(yBuffer[8*leda.HASH_BYTE_LENGTH:])
    secretSeed = yBuffer[:8*leda.TRNG_BYTE_LENGTH] ^ hash[0, :8*leda.TRNG_BYTE_LENGTH]
    secretSeed = bitarray_to_int(secretSeed)
    prngSeq = bytes_to_bitarray(RNG(secretSeed).randombytes(yBufferBitLength // 8 - leda.HASH_BYTE_LENGTH))

    ptx = yBuffer[8*leda.HASH_BYTE_LENGTH:] ^ prngSeq
    ptx = remove_padding(ptx)
    return np.packbits(ptx).tobytes()
