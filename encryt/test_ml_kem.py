import unittest
import json
from src.kyber_py.ml_kem import ML_KEM_512, ML_KEM_768, ML_KEM_1024
import time

class TestML_KEM(unittest.TestCase):
    """
    Test ML_KEM levels for internal
    consistency by generating key pairs
    and shared secrets.
    """

    def generic_test_ML_KEM(self, ML_KEM, count):
        for _ in range(count):
            (ek, dk) = ML_KEM.keygen()
            for _ in range(count):
                (K, c) = ML_KEM.encaps(ek)
                K_prime = ML_KEM.decaps(dk, c)
                self.assertEqual(K, K_prime)

    def test_ML_KEM_512(self):
        self.generic_test_ML_KEM(ML_KEM_512, 5)

    def test_ML_KEM_768(self):
        self.generic_test_ML_KEM(ML_KEM_768, 5)

    def test_ML_KEM_1024(self):
        self.generic_test_ML_KEM(ML_KEM_1024, 5)

    def test_encaps_type_check_failure(self):
        """
        Send an ecaps key of the wrong length
        """
        self.assertRaises(ValueError, lambda: ML_KEM_512.encaps(b"1"))

    def test_encaps_modulus_check_failure(self):
        """
        We create a vector of polynomials with non-canonical values for
        coefficents to fail the modulus check
        """
        (ek, _) = ML_KEM_512.keygen()
        rho = ek[-32:]

        bad_f_hat = ML_KEM_512.R([3329] * 256)
        bad_t_hat = ML_KEM_512.M.vector([bad_f_hat, bad_f_hat])
        bad_t_hat_bytes = bad_t_hat.encode(12)

        bad_ek = bad_t_hat_bytes + rho

        self.assertEqual(len(bad_ek), len(ek))
        self.assertRaises(ValueError, lambda: ML_KEM_512.encaps(bad_ek))

    def test_xof_failure(self):
        self.assertRaises(
            ValueError, lambda: ML_KEM_512._xof(b"1", b"2", b"3")
        )

    def test_prf_failure(self):
        self.assertRaises(ValueError, lambda: ML_KEM_512._prf(2, b"1", b"2"))

    def test_decaps_ct_type_check_failure(self):
        """
        Send a ciphertext of the wrong length
        """
        ek, dk = ML_KEM_512.keygen()
        K, c = ML_KEM_512.encaps(ek)
        self.assertRaises(ValueError, lambda: ML_KEM_512.decaps(dk, b"1"))

    def test_decaps_dk_type_check_failure(self):
        """
        Send a ciphertext of the wrong length
        """
        ek, dk = ML_KEM_512.keygen()
        K, c = ML_KEM_512.encaps(ek)
        self.assertRaises(ValueError, lambda: ML_KEM_512.decaps(b"1", c))

    def test_decaps_hash_check_failure(self):
        """
        Send a ciphertext of the wrong length
        """
        ek, dk = ML_KEM_512.keygen()
        K, c = ML_KEM_512.encaps(ek)
        dk_bad = b"0" * len(dk)
        self.assertRaises(ValueError, lambda: ML_KEM_512.decaps(dk_bad, c))


class TestML_KEM_KAT(unittest.TestCase):
    """
    Test ML-KEM against test vectors collected from
    https://github.com/usnistgov/ACVP-Server/releases/tag/v1.1.0.35
    """

    def generic_keygen_kat(self, ML_KEM, index):
        with open("assets/ML-KEM-keyGen-FIPS203/internalProjection.json") as f:
            data = json.load(f)
        kat_data = data["testGroups"][index]["tests"]

        for test in kat_data:
            d_kat = bytes.fromhex(test["d"])
            z_kat = bytes.fromhex(test["z"])
            ek_kat = bytes.fromhex(test["ek"])
            dk_kat = bytes.fromhex(test["dk"])

            ek, dk = ML_KEM._keygen_internal(d_kat, z_kat)
            self.assertEqual(ek, ek_kat)
            self.assertEqual(dk, dk_kat)

    def generic_encap_decap_kat(self, ML_KEM, index):
        with open(
            "assets/ML-KEM-encapDecap-FIPS203/internalProjection.json"
        ) as f:
            data = json.load(f)
        kat_data = data["testGroups"][index]["tests"]

        for test in kat_data:
            ek_kat = bytes.fromhex(test["ek"])
            dk_kat = bytes.fromhex(test["dk"])
            c_kat = bytes.fromhex(test["c"])
            k_kat = bytes.fromhex(test["k"])
            m_kat = bytes.fromhex(test["m"])

            K, c = ML_KEM._encaps_internal(ek_kat, m_kat)
            self.assertEqual(K, k_kat)
            self.assertEqual(c, c_kat)

            K_prime = ML_KEM.decaps(dk_kat, c_kat)
            self.assertEqual(K_prime, k_kat)

    def test_ML_KEM_512_keygen(self):
        self.generic_keygen_kat(ML_KEM_512, 0)

    def test_ML_KEM_768_keygen(self):
        self.generic_keygen_kat(ML_KEM_768, 1)

    def test_ML_KEM_1024_keygen(self):
        self.generic_keygen_kat(ML_KEM_1024, 2)

    def test_ML_KEM_512_encap_decap(self):
        self.generic_encap_decap_kat(ML_KEM_512, 0)

    def test_ML_KEM_768_encap_decap(self):
        self.generic_encap_decap_kat(ML_KEM_768, 1)

    def test_ML_KEM_1024_encap_decap(self):
        self.generic_encap_decap_kat(ML_KEM_1024, 2)

if __name__ == '__main__':
    T0 = time.time()
    """Parameters for the :py:obj:`.ML_KEM` objects."""

    ek, dk = ML_KEM_512.keygen()
    print("ek = ", ek)
    print("ek = ", bytes.hex(ek))
    print("\n")
    print("dk = ", bytes.hex(dk))
    print("\n")
    T1 = time.time()
    key, ct = ML_KEM_512.encaps(ek)
    print("key = ", key)
    print("zhuan-key = ", int(bytes.hex(key), 16))
    key_ten = int(bytes.hex(key), 16)
    key_ten = str(key_ten)
    key_ten1 = key_ten[:3]
    key_ten2 = key_ten[4:7]
    key_ten3 = key_ten[8:11]
    key_ten4=key_ten1+key_ten2
    print("key4:",key_ten4)
    print("key123:", key_ten1, key_ten2, key_ten3)

    print("ct = ", bytes.hex(ct))
    _key = ML_KEM_512.decaps(dk, ct)
    print("_key = ", int(bytes.hex(_key), 16))
    assert key == _key
    # test = TestML_KEM
    # test512 = test.test_ML_KEM_512



    T2 = time.time()
    print('程序运行时间:%s秒' % ((T2 - T0)))
    print('程序运行时间:%s秒' % ((T2 - T1)))
