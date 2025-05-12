# Quantum Key Distribution Simulation
A Python-based simulation of **Quantum Key Distribution (QKD)** using **IBM Qiskit** to enhance secure communication through quantum cryptography principles.

This project demonstrates a simple BB84 protocol flow — generating secret keys, encrypting a message via XOR, and transmitting securely between Alice and Bob.

---

## Features

- Quantum key generation using simulated qubits.
- Secret key agreement based on BB84 protocol.
- XOR-based message encryption & decryption.
- Quantum Bit Error Rate (QBER) calculation.
- Clear console outputs for learning and demonstration.

---

## Project Structure

```
Quantum-Crypto-Simulation/
├── initialization.py      # Random bit and basis selection
├── transmission.py        # Quantum state preparation and measurement
├── key_sifting.py         # Matching bases, filtering, and QBER calculation
├── encryption.py          # Text encryption/decryption using XOR
├── main.py                # Full simulation runner
├── requirements.txt       # Dependency list
└── README.md              # Project documentation
```

---

## How It Works

1. **Alice** randomly chooses bits and quantum bases.
2. **Bob** chooses random bases for measurement.
3. They exchange basis information and discard mismatched measurements.
4. The shared key is used to encrypt a message using XOR.
5. The same key is used for decryption on Bob’s side.

Example Output:
```
Alice's bits:    [0 1 0 0 0 1 0 0 0 1]
Alice's bases:   ['Z' 'Z' 'Z' 'Z' 'X' 'Z' 'X' 'X' 'X' 'Z']
Bob's bases:     ['X' 'Z' 'X' 'X' 'X' 'X' 'X' 'X' 'X' 'X']
Bob's results:   [0, 1, 0, 1, 0, 1, 0, 0, 0, 1]
Matching indices:[1, 4, 6, 7, 8]
Shared key:      [1, 0, 0, 0, 0]

Original Message:  HELLO
Encrypted Message: IELLO
Decrypted Message: HELLO
```

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/Quantum-Crypto-Simulation.git
cd Quantum-Crypto-Simulation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the simulation:
```bash
python run_protocol.py
```

---

## Requirements

- Python 3.8+
- Qiskit  
- numpy

Your `requirements.txt` should contain:
```
qiskit
qiskit-aer
numpy
```

---

## Learning Purpose

This project is designed for:
- Students exploring quantum computing basics.
- Security enthusiasts understanding quantum cryptography.
- Demonstrating BB84 and XOR encryption in an educational way.

---

## Contributing

Pull requests, suggestions, and improvements are welcome!  
Fork the repo, create a new branch, and submit your PR.

---

## License

MIT License.  
For academic, educational, and demonstration purposes.

---

