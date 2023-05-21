import numpy as np
from gf2x import gf2x_mod_add, gf2x_mod_mul
from random_utils import TRNG, RNG
from utils import unpack_bytes, pack_bytes, int_to_bitarray, bytes_to_bitarray
from cw import binary_to_constant_weight_approximate


def plaintext_constant_pad(msgArr, yBufferBitLength, HASH_BYTE_LENGTH):
    padded = np.zeros(yBufferBitLength, dtype=bool)
    msgBitLen = msgArr.shape[0]

    padded[8*HASH_BYTE_LENGTH:8*HASH_BYTE_LENGTH + msgBitLen] = msgArr.copy()
    padded[8*HASH_BYTE_LENGTH + msgBitLen] = 0x1
    padded[-1] = 0x1

    return padded


def encrypt(pk, informationWord, encodedError, leda):
    codeword = np.zeros(leda.N0*leda.P, dtype=bool)
    codeword[:leda.K] = informationWord.copy()

    for i in range(leda.N0-1):
        mul = gf2x_mod_mul(leda.getBlock(pk, i), leda.getBlock(informationWord, i))
        codeword[leda.K:] = gf2x_mod_add(codeword[leda.K:], mul)

    for i in range(leda.N0):
        codeword[leda.getBlockSlice(i)] = gf2x_mod_add(leda.getBlock(codeword, i), leda.getBlock(encodedError, i))

    return codeword


def encode(msg, pk, leda):
    msgArr = bytes_to_bitarray(msg)
    pk = unpack_bytes(pk)

    yBufferBitLength = leda.CONSTANT_WEIGHT_ENCODED_DATA_ACTUAL_BIT_LENGTH + leda.K
    encodedError = None
    while True:
        secretSeed = TRNG(leda.TRNG_BYTE_LENGTH)
        rng = RNG(secretSeed)

        prngSeqByteLen = yBufferBitLength // 8 - leda.HASH_BYTE_LENGTH
        prngSequence = bytes_to_bitarray(rng.randombytes(prngSeqByteLen))

        yBuffer = plaintext_constant_pad(msgArr, yBufferBitLength, leda.HASH_BYTE_LENGTH)

        yBuffer[8*leda.HASH_BYTE_LENGTH:] ^= prngSequence

        yBuffer[:8*leda.HASH_BYTE_LENGTH] = leda.HASH_FUNC(yBuffer[8*leda.HASH_BYTE_LENGTH:])

        secretSeedBits = int_to_bitarray(secretSeed, leda.TRNG_BYTE_LENGTH)
        yBuffer[:8*leda.TRNG_BYTE_LENGTH] ^= secretSeedBits[:]

        informationWord = np.zeros(leda.K, dtype=bool)
        for i in range(leda.N0-1):
            informationWord[leda.getBlockSlice(i)] = leda.getBlock(yBuffer[leda.CONSTANT_WEIGHT_ENCODED_DATA_ACTUAL_BIT_LENGTH:], i)[::-1]
        encodedError = binary_to_constant_weight_approximate(yBuffer[:leda.CONSTANT_WEIGHT_ENCODED_DATA_ACTUAL_BIT_LENGTH], leda)

        if encodedError is not None:
            break

    ctx = encrypt(pk, informationWord, encodedError, leda)
    return pack_bytes(ctx)
