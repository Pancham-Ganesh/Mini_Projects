# sifting.py
def sift_and_correct(alice_bits, alice_bases, bob_bases, bob_results):
    matching_indices = [i for i in range(len(alice_bases)) if alice_bases[i] == bob_bases[i]]
    
    if not matching_indices:
        raise ValueError("No matching bases found. Cannot establish a shared key.")
    
    alice_key = [alice_bits[i] for i in matching_indices]
    bob_key = [bob_results[i] for i in matching_indices]
    
    errors = sum(1 for a, b in zip(alice_key, bob_key) if a != b)
    qber = errors / len(alice_key)

    # If QBER is too high, consider the key compromised
    if qber > 0.2:
        raise ValueError(f"High QBER detected ({qber:.2f}). Possible eavesdropping!")

    # Only keep bits that match
    secret_key = [a for a, b in zip(alice_key, bob_key) if a == b]
    
    return matching_indices, secret_key, qber
