import numpy as np
from utils import bitarray_to_int


def estimate_d_u(availablePositions, onesToPlace):
    d = int(0.69315 * (availablePositions - (onesToPlace - 1.0)/2.0) / onesToPlace)

    u = 0
    tmp = d
    while tmp != 0:
        tmp >>= 1
        u += 1
    return d, u


def read_bitstream(stream, cursor, amount):
    if cursor.val + amount > stream.shape[0]:
        diff = cursor.val + amount - stream.shape[0]
        ret = np.concatenate((stream[cursor.val:], np.zeros(diff, dtype=bool))).astype(bool)
        cursor.val = stream.shape[0]
        return ret
    ret = stream[cursor.val:cursor.val+amount]
    cursor.val += amount
    return ret


class Cursor:
    def __init__(self):
        self.val = 0


def binary_to_constant_weight_approximate(buff, leda):
    distanceBetweenOnes = np.zeros(leda.NUM_ERRORS_T, dtype=np.uint32)
    idxDistances = 0
    onesStillToPlace = leda.NUM_ERRORS_T
    outPositionsAvailable = leda.N0*leda.P
    bitStreamCursor = Cursor()

    while idxDistances < leda.NUM_ERRORS_T and outPositionsAvailable > onesStillToPlace:
        d, u = estimate_d_u(outPositionsAvailable, onesStillToPlace)

        quotient = 0
        while read_bitstream(buff, bitStreamCursor, 1)[0] == 1:
            quotient += 1

        distanceToBeComputed = bitarray_to_int(read_bitstream(buff, bitStreamCursor, u-1)) if u > 0 else 0

        if distanceToBeComputed >= ((1 << u) - d):
            distanceToBeComputed *= 2
            distanceToBeComputed += read_bitstream(buff, bitStreamCursor, 1)[0]
            distanceToBeComputed -= ((1 << u) - d)

        distanceBetweenOnes[idxDistances] = distanceToBeComputed + quotient*d
        outPositionsAvailable -= distanceBetweenOnes[idxDistances] + 1
        onesStillToPlace -= 1
        idxDistances += 1

    if outPositionsAvailable == onesStillToPlace:
        distanceBetweenOnes[idxDistances:] = 0

    if outPositionsAvailable < onesStillToPlace:
        return None

    cw = np.zeros(leda.N0*leda.P, dtype=bool)
    current_one_position = -1
    for i in range(leda.NUM_ERRORS_T):
        current_one_position += distanceBetweenOnes[i] + 1
        if current_one_position >= leda.N0*leda.P:
            return None
        polyIdx = current_one_position // leda.P
        exponent = current_one_position % leda.P
        cw[leda.P*(polyIdx+1) - exponent - 1] = 1

    return cw


def constant_weight_to_binary(constantWeight, yBufferLen, leda):
    distanceBetweenOnes = np.zeros(leda.NUM_ERRORS_T, dtype=np.uint32)

    last_one_position = -1
    idxDistances = 0

    for current_inspected_position in range(leda.N0*leda.P):
        current_inspected_exponent = current_inspected_position % leda.P
        current_inspected_poly = current_inspected_position // leda.P

        if constantWeight[(current_inspected_poly+1)*leda.P - current_inspected_exponent - 1] == 1:
            distanceBetweenOnes[idxDistances] = current_inspected_position - last_one_position - 1
            last_one_position = current_inspected_position
            idxDistances += 1

    onesStillToPlace = leda.NUM_ERRORS_T
    posStillAvailable = leda.N0*leda.P

    out = np.zeros(yBufferLen, dtype=bool)
    outputBitCursor = 0
    for idxDistances in range(leda.NUM_ERRORS_T):
        d, u = estimate_d_u(posStillAvailable, onesStillToPlace)

        quotient = distanceBetweenOnes[idxDistances] // d
        out[outputBitCursor:outputBitCursor+quotient] = 1
        out[outputBitCursor+quotient] = 0
        outputBitCursor += quotient + 1

        remainder = distanceBetweenOnes[idxDistances] % d
        if remainder < ((1 << u) - d):
            u = u-1 if u > 0 else 0
            val = np.array([(remainder >> i) & 0x1 for i in reversed(range(u))], dtype=bool)
            out[outputBitCursor:outputBitCursor+u] = val
        else:
            remainder += ((1 << u) - d)
            val = np.array([(remainder >> i) & 0x1 for i in reversed(range(u))], dtype=bool)
            out[outputBitCursor:outputBitCursor+u] = val
        outputBitCursor += u
        posStillAvailable -= distanceBetweenOnes[idxDistances] + 1
        onesStillToPlace -= 1

    return out
