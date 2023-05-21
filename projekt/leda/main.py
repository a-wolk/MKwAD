from keygen import keygen
from encode import encode
from decode import decode
from LEDA import LEDACat

if __name__ == '__main__':
    leda = LEDACat.get(4, 4)
    sk, pk = keygen(leda)
    print("SK:", sk)
    print("PK:", pk)

    e = encode("AAAABBBBCCCCDDDD 1111222233334444".encode("utf-8"), pk, leda)
    print("Ciphertext:", e)
    d = decode(e, sk, leda)
    print("Plaintext:", d.decode("utf-8"))