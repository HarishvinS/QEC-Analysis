# Quantum Error Correction & Algorithms Analysis

A platform for simulating 2D surface codes, noise modeling, and fundamental quantum algorithms. This project leverages Stim, PyMatching, and Sinter to provide high-precision analysis of fault-tolerant quantum computing systems.

## What this does

- **2D Rotated Surface Codes**: Implementation of industry-standard topological codes with integrated X and Z stabilizer measurements.
- **Fault-Tolerant Memory**: Simulation of quantum memory experiments to determine logical error suppression.
- **Quantum Algorithms**: 
    - **Shor's Algorithm**: Efficient circuit for integer factorization (factoring 15).
    - **Grover's Search**: Quadratic speedup for unstructured database searching.
- **Noise Modeling**: Integrated depolarizing noise, measurement flips, and reset errors based on physical hardware benchmarks.
- **Threshold Analysis**: Automated detection of error correction thresholds and code-distance scaling.
- **Visualizations**: Comprehensive plotting suite for error rates, resource scaling, and algorithm success probabilities.

## Project Structure

- `surface_code.py`: Generation of rotated surface code circuits using Stim.
- `quantum_algorithms.py`: Implementation of Shor's (QFT-based) and Grover's search.
- `noise_models.py`: Support for custom noise channels and error rate estimations.
- `validation.py`: Statistical analysis, PyMatching integration, and threshold calculation.
- `visualization.py`: Grahpical visualization feature.
- `main.py`: Full-scale parameter sweeps and analysis.

## Quick Start

### 1. Installation

Ensure you have the required dependencies installed:

```bash
pip install stim pymatching sinter numpy matplotlib
```

### 2. Run Analysis

Execute the main pipeline to generate a full suite of analysis plots and statistics:

```bash
python main.py
```

For a faster, focused test of the core features, run:

```bash
python main_simple.py
```

## Technical Details

### Surface Code Performance
The project uses the Rotated Surface Code. By scaling the Code Distance ($d$), the system exhibits how logical errors are exponentially suppressed below the Error Correction Threshold (typically observed at ~0.7%).

### Algorithm Implementations
- **Shor's Algorithm (N=15)**: A mathematically rigorous implementation of the $a^x \pmod{15}$ circuit using controlled-swap permutations (Fredkin gates).
- **Grover's Search**: Features a proper Multi-Controlled Z (MCZ) oracle and diffusion operator, decomposed into Toffoli gates with ancilla management.
 **Notes**: 
 - More rigourous implementation of Shor's algorithm is possible using QFT and controlled rotations. 
 - More algorithms which may have ore real-world applications are possible, such as quantum chemistry, quantum machine learning, etc. My most immediate goal is to implement quantum chemistry algorithms and Quantum Neural Networks.

### Simulation Limitations & Constraints
> [!IMPORTANT]
> This project uses Stim, a high-performance Stabilizer Simulator. While this allows for simulating thousands of qubits in surface codes, it introduces specific constraints for general algorithms:

1. **Clifford Approximation**: Stim only natively supports Clifford gates (H, S, CNOT, etc.). Non-Clifford gates (like the T-gate or CCZ) are implemented using Clifford-based decompositions that work correctly for computational basis states but do not capture universal quantum behavior.

2. **Approximate QFT**: The QFT used in Shor's algorithm is an *Approximate QFT* (AQFT). It ignores small-angle rotations that fall outside the Clifford set, which is a standard technique in fault-tolerant quantum computing design.

3. **Performance Expectations**: 
    - **Memory experiments** reach near-perfect logical error suppression below threshold.
    - **Grover's Search** success rate is significantly higher than random but may be affected by the Clifford-only approximation of the oracle phase.

## Generated Analysis

The following plots are generated automatically in the project directory:

- `logical_vs_physical_errors.png`: Scaling performance across different code distances.
- `threshold_analysis.png`: Identification of the crossover point for error correction benefit.
- `protected_vs_unprotected.png`: Direct comparison between raw and error-corrected logical qubits.
- `resource_scaling.png`: Analysis of qubit overhead ($O(d^2)$ scaling).

---
If you have any questions or suggestions, please feel free to contact me at harishsasi17@gmail.com
