from random import getrandbits, Random


def TRNG(n):
    return getrandbits(n*8)


class RNG:
    def __init__(self, seed):
        self.rng = Random(seed)

    def randrange(self, stop):
        return self.rng.randrange(0, stop)

    def randombytes(self, n):
        return self.rng.getrandbits(n*8).to_bytes(n, byteorder='big')