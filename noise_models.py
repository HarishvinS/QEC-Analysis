"""
Noise Models for Quantum Circuits

This module provides noise models for realistic quantum simulation.
"""

import stim
from typing import Dict, List


def add_depolarizing_noise(circuit: stim.Circuit, error_rate: float) -> stim.Circuit:
    """
    Add depolarizing noise to a quantum circuit.
    
    Args:
        circuit: The quantum circuit
        error_rate: The probability of error for each gate
    
    Returns:
        A circuit with depolarizing noise
    """
    noisy_circuit = stim.Circuit()
    
    for instruction in circuit:
        # Add the original instruction
        noisy_circuit.append(instruction)
        
        # Add noise after each gate operation (not after measurements/resets/TICK)
        if instruction.name in ["H", "X", "Y", "Z", "S", "SQRT_X", "SQRT_Y"]:
            # Single-qubit gates - apply single-qubit depolarizing noise
            targets = instruction.targets_copy()
            for target in targets:
                if hasattr(target, 'value'):  # It's a qubit target
                    noisy_circuit.append("DEPOLARIZE1", [target], error_rate)
        
        elif instruction.name in ["CNOT", "CZ", "CY", "SWAP"]:
            # Two-qubit gates - apply two-qubit depolarizing noise
            targets = instruction.targets_copy()
            if len(targets) >= 2:
                # Apply two-qubit depolarizing noise to pairs
                for i in range(0, len(targets), 2):
                    if i + 1 < len(targets):
                        noisy_circuit.append("DEPOLARIZE2", 
                                           [targets[i], targets[i+1]], error_rate)
    
    return noisy_circuit


def add_measurement_noise(circuit: stim.Circuit, readout_error: float) -> stim.Circuit:
    """
    Add measurement (readout) errors to a circuit.
    
    Args:
        circuit: The quantum circuit
        readout_error: Probability of bit flip during measurement
    
    Returns:
        Circuit with measurement noise
    """
    noisy_circuit = stim.Circuit()
    
    for instruction in circuit:
        noisy_circuit.append(instruction)
        
        # Add X_ERROR before measurements to simulate readout errors
        if instruction.name == "M":
            targets = instruction.targets_copy()
            for target in targets:
                if hasattr(target, 'value'):
                    # Apply X error before measurement
                    noisy_circuit.append("X_ERROR", [target], readout_error)
    
    return noisy_circuit


def add_combined_noise(circuit: stim.Circuit, 
                       gate_error_rate: float,
                       readout_error_rate: float = 0.001) -> stim.Circuit:
    """
    Add both gate errors and readout errors to a circuit.
    
    Args:
        circuit: The quantum circuit
        gate_error_rate: Error rate for gate operations
        readout_error_rate: Error rate for measurements
    
    Returns:
        Circuit with combined noise
    """
    # First add gate noise
    noisy_circuit = add_depolarizing_noise(circuit, gate_error_rate)
    
    # Then add readout noise
    noisy_circuit = add_measurement_noise(noisy_circuit, readout_error_rate)
    
    return noisy_circuit


def estimate_circuit_error_rate(circuit: stim.Circuit, 
                                gate_error_rate: float) -> float:
    """
    Estimate the total error probability for a circuit.
    
    Args:
        circuit: The quantum circuit
        gate_error_rate: Per-gate error rate
    
    Returns:
        Estimated total error probability
    """
    # Count gates
    num_gates = 0
    for instruction in circuit:
        if instruction.name in ["H", "X", "Y", "Z", "S", "CNOT", "CZ"]:
            num_gates += 1
    
    # Approximate total error (assuming independent errors)
    # P(at least one error) ≈ 1 - (1 - p)^n ≈ n*p for small p
    total_error_rate = min(1.0, num_gates * gate_error_rate)
    
    return total_error_rate


if __name__ == "__main__":
    # Test noise models
    print("Testing noise models...")
    
    # Create a simple test circuit
    test_circuit = stim.Circuit()
    test_circuit.append_operation("R", [0, 1])
    test_circuit.append_operation("H", [0])
    test_circuit.append_operation("CNOT", [0, 1])
    test_circuit.append_operation("M", [0, 1])
    
    print(f"Original circuit: {len(list(test_circuit))} instructions")
    
    # Add noise
    noisy_circuit = add_depolarizing_noise(test_circuit, 0.01)
    print(f"With depolarizing noise: {len(list(noisy_circuit))} instructions")
    
    # Add combined noise
    combined_noisy = add_combined_noise(test_circuit, 0.01, 0.001)
    print(f"With combined noise: {len(list(combined_noisy))} instructions")
    
    # Estimate error rate
    error_est = estimate_circuit_error_rate(test_circuit, 0.01)
    print(f"Estimated circuit error rate: {error_est:.4f}")
    
    print("Noise models working correctly")
