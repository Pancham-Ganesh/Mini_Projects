# error_estimation.py
import random

def estimate_qber(sifted_alice, sifted_bob):
    sample_size = max(1, len(sifted_alice) // 5)
    sample_indices = random.sample(range(len(sifted_alice)), sample_size)

    sample_alice = [sifted_alice[i] for i in sample_indices]
    sample_bob = [sifted_bob[i] for i in sample_indices]

    errors = sum(a != b for a, b in zip(sample_alice, sample_bob))
    qber = errors / sample_size

    final_key = [bit for i, bit in enumerate(sifted_alice) if i not in sample_indices]

    return qber, final_key
