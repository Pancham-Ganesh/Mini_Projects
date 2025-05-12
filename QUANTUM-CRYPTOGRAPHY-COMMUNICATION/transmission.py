# transmission.py
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
import numpy as np

def transmit(alice_bits, alice_bases, bob_bases, eavesdrop=False):
    bob_results = []
    simulator = AerSimulator()

    for i in range(len(alice_bits)):
        qc = QuantumCircuit(1, 1)

        # Alice prepares the qubit
        if alice_bases[i] == 0:
            if alice_bits[i] == 1:
                qc.x(0)
        else:
            if alice_bits[i] == 1:
                qc.x(0)
            qc.h(0)

        # Simulate Eve (eavesdropper)
        if eavesdrop:
            eve_basis = np.random.randint(2)
            if eve_basis == 1:
                qc.h(0)
            qc.measure(0, 0)

            # Run Eve's measurement
            eve_job = simulator.run(qc, shots=1, memory=True)
            eve_result = int(eve_job.result().get_memory()[0])

            # Reset qubit and re-encode based on what Eve "measured"
            qc = QuantumCircuit(1, 1)
            if eve_result == 1:
                qc.x(0)
            if eve_basis == 1:
                qc.h(0)

        # Bob's measurement
        if bob_bases[i] == 1:
            qc.h(0)
        qc.measure(0, 0)

        job = simulator.run(qc, shots=1, memory=True)
        measured_bit = int(job.result().get_memory()[0])
        bob_results.append(measured_bit)

    return bob_results
