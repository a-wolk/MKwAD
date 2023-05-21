import hashlib
import numpy as np


class LEDA:
    def __init__(self, N0, P, DV, MS, NUM_ERRORS_T, MAX_ENCODABLE_BIT_SIZE_CW_ENCODING,
                 HASH_FUNC, HASH_BYTE_LENGTH, TRNG_BYTE_LENGTH, SYND_TRESH_LOOKUP_TABLE):
        self.N0 = N0
        self.P = P
        self.DV = DV
        self.MS = MS
        self.M = sum(MS)
        self.NUM_ERRORS_T = NUM_ERRORS_T
        self.MAX_ENCODABLE_BIT_SIZE_CW_ENCODING = MAX_ENCODABLE_BIT_SIZE_CW_ENCODING
        self.HASH_FUNC = HASH_FUNC
        self.HASH_BYTE_LENGTH = HASH_BYTE_LENGTH
        self.TRNG_BYTE_LENGTH = TRNG_BYTE_LENGTH
        self.SYND_TRESH_LOOKUP_TABLE = SYND_TRESH_LOOKUP_TABLE
        self.K = (N0-1)*P
        self.INVALID_POS_VALUE = P

        self.SAFE_BIT_SIZE_CW_ENCODING = MAX_ENCODABLE_BIT_SIZE_CW_ENCODING - MAX_ENCODABLE_BIT_SIZE_CW_ENCODING // 30
        self.CONSTANT_WEIGHT_ENCODED_DATA_ACTUAL_BIT_LENGTH = self.SAFE_BIT_SIZE_CW_ENCODING - (self.SAFE_BIT_SIZE_CW_ENCODING + self.K) % 8
        self.KOBARA_IMAI_DOMAIN_SEPARATION_CONSTANT_MIN_BIT_LENGTH = 2
        self.KOBARA_IMAI_MAX_PTX_BIT_LENGTH = self.K + self.CONSTANT_WEIGHT_ENCODED_DATA_ACTUAL_BIT_LENGTH - 8 * HASH_BYTE_LENGTH - self.KOBARA_IMAI_DOMAIN_SEPARATION_CONSTANT_MIN_BIT_LENGTH

    def q_block_weight(self, r, c):
        return self.MS[c - r]

    def getBlock(self, A, i):
        return A[self.P*i:self.P*(i+1)]

    def getBlockSlice(self, i):
        return slice(self.P*i, self.P*(i+1))


def sha3_256(bitarray):
    bintxt = bytes((bitarray.reshape(-1, 8) * np.array([128, 64, 32, 16, 8, 4, 2, 1]).reshape(1, 8)).sum(axis=1).tolist())
    h = hashlib.sha3_256(bintxt).digest()
    return np.array(list(map(lambda v: [v & (0x1 << i) for i in reversed(range(8))], h)), dtype=bool).reshape(1, -1)


def sha3_384(bitarray):
    bintxt = bytes((bitarray.reshape(-1, 8) * np.array([128, 64, 32, 16, 8, 4, 2, 1]).reshape(1, 8)).sum(axis=1).tolist())
    h = hashlib.sha3_384(bintxt).digest()
    return np.array(list(map(lambda v: [v & (0x1 << i) for i in reversed(range(8))], h)), dtype=bool).reshape(1, -1)


def sha3_512(bitarray):
    bintxt = bytes((bitarray.reshape(-1, 8) * np.array([128, 64, 32, 16, 8, 4, 2, 1]).reshape(1, 8)).sum(axis=1).tolist())
    h = hashlib.sha3_512(bintxt).digest()
    return np.array(list(map(lambda v: [v & (0x1 << i) for i in reversed(range(8))], h)), dtype=bool).reshape(1, -1)


class LEDACat:
    __synd_tresh_lookup_tables_1 = {
        2: [
            [0, 43],
            [2843, 44],
            [4392, 45],
            [5193, 46],
            [5672, 47]
        ],
        3: [
            [0, 49],
            [2509, 50],
            [3124, 51],
            [3478, 52],
            [3695, 53],
            [3878, 54]
        ],
        4: [
            [0, 53],
            [2021, 54],
            [2611, 55],
            [2957, 56],
            [3181, 57],
            [3345, 58],
            [3447, 59]
        ]
    }

    __synd_tresh_lookup_tables_2 = {
        2: [
            [0, 61],
            [3957, 62],
            [6698, 63],
            [8128, 64],
            [8978, 65],
            [9578, 66],
            [9981, 67],
            [10286, 68],
            [10533, 69]
        ],
        3: [
            [0, 71],
            [4255, 72],
            [5492, 73],
            [6203, 74],
            [6666, 75],
            [7021, 76],
            [7271, 77],
            [7466, 78],
            [7617, 79]
        ],
        4: [
            [0, 71],
            [3244, 72],
            [4359, 73],
            [5006, 74],
            [5408, 75],
            [5712, 76],
            [5915, 77],
            [6094, 78],
            [6230, 79]
        ]
    }

    __synd_tresh_lookup_tables_4 = {
        2: [
            [0, 74],
            [5742, 75],
            [10032, 76],
            [12263, 77],
            [13621, 78],
            [14538, 79],
            [15211, 80],
            [15706, 81],
            [16091, 82],
            [16391, 83],
            [16640, 84]
        ],
        3: [
            [0, 88],
            [6551, 89],
            [8560, 90],
            [9789, 91],
            [10536, 92],
            [11123, 93],
            [11519, 94],
            [11837, 95],
            [12091, 96],
            [12319, 97]
        ],
        4: [
            [0, 88],
            [4788, 89],
            [6581, 90],
            [7620, 91],
            [8300, 92],
            [8782, 93],
            [9121, 94],
            [9386, 95],
            [9593, 96],
            [9780, 97]
        ]
    }

    __instances_1 = {
        2: LEDA(2, 15013, 9, [5, 4], 143, 1304, sha3_256, 32, 24, __synd_tresh_lookup_tables_1[2]),
        3: LEDA(3, 9643, 13, [3, 2, 2], 90, 874, sha3_256, 32, 24, __synd_tresh_lookup_tables_1[3]),
        4: LEDA(4, 8467, 11, [3, 2, 2, 2], 72, 738, sha3_256, 32, 24, __synd_tresh_lookup_tables_1[4])
    }

    __instances_2 = {
        2: LEDA(2, 24533, 13, [5, 4], 208, 1933, sha3_384, 48, 32, __synd_tresh_lookup_tables_2[2]),
        3: LEDA(3, 17827, 15, [4, 3, 2], 129, 1302, sha3_384, 48, 32, __synd_tresh_lookup_tables_2[3]),
        4: LEDA(4, 14717, 15, [3, 2, 2, 2], 104, 1096, sha3_384, 48, 32, __synd_tresh_lookup_tables_2[4])
    }

    __instances_4 = {
        2: LEDA(2, 37619, 11, [7, 6], 272, 2592, sha3_512, 64, 40, __synd_tresh_lookup_tables_4[2]),
        3: LEDA(3, 28477, 13, [5, 4, 4], 172, 1783, sha3_512, 64, 40, __synd_tresh_lookup_tables_4[3]),
        4: LEDA(4, 22853, 13, [4, 3, 3, 3], 135, 1459, sha3_512, 64, 40, __synd_tresh_lookup_tables_4[4])
    }

    __categories_instances_map = {
        1: __instances_1,
        2: __instances_2,
        3: __instances_2,
        4: __instances_4,
        5: __instances_4
    }

    @staticmethod
    def get(CATEGORY, N0):
        if CATEGORY < 1 or CATEGORY > 5:
            raise Exception("CATEGORY must be 1, 2, 3, 4 or 5")
        if N0 < 2 or N0 > 4:
            raise Exception("N0 must be 2, 3 or 4")
        return LEDACat.__categories_instances_map[CATEGORY][N0]