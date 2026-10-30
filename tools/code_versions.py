"""Code hashes (src/utils/common.py:code_fingerprint) of the code versions in this repository.

Runs record the hash of the code that produced them. The released code computes
the same results as each executed version (tools/verify_release.py); the batch
runners check that they run the released code.
"""
RELEASED = "112f9c4c9616"
EXECUTED = {
    "de756535ee20": "protocols A-G, C0-C24, T",
    "598ac132e99f": "protocols V, N, P, Q, R, L, M, W, S50, S100",
    "c73ef066fa6e": "protocols Z20, Z50, Z100, ZN, ZM",
}
