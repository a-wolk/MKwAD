import numpy as np


def remove_padding(arr):
    arr[-1] = 0
    i = -1
    while arr[i] == 0:
        i -= 1
    return arr[:i]


def add_padding(arr):
    paddingLen = (8 - arr.shape[0] % 8) % 8 + 8
    padding = np.zeros(paddingLen, dtype=bool)
    padding[0] = 1
    padding[-1] = 1
    return np.concatenate((arr, padding))


def unpack_bytes(arr):
    return remove_padding(np.unpackbits(np.array(list(arr), dtype=np.uint8)))


def pack_bytes(arr):
    return np.packbits(add_padding(arr)).tobytes()


def int_to_bitarray(x, byteLength):
    return np.array([x & (0x1 << i) for i in reversed(range(8*byteLength))], dtype=bool)


BIG_ENDIAN_MASK = np.array([128, 64, 32, 16, 8, 4, 2, 1], dtype=np.uint8)


def bytes_to_bitarray(arr):
    return (np.array(list(arr), dtype=np.uint8).reshape(-1, 1) & BIG_ENDIAN_MASK).astype(bool).reshape(-1)


def bitarray_to_int(x):
    r = 0
    for i,v in enumerate(x.tolist()[::-1]):
        if v == 1:
            r += 2**i
    return r