# initialization.py
import numpy as np

def initialize(n=100):
    alice_bits = np.random.randint(2, size=n)
    alice_bases = np.random.randint(2, size=n)
    bob_bases = np.random.randint(2, size=n)
    return alice_bits, alice_bases, bob_bases
