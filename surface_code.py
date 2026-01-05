"""
True 2D Surface Code Implementation

This module implements a rotated surface code for quantum error correction.
Unlike 1D repetition codes, this is a proper topological code with:
- Data qubits arranged in a 2D square lattice
- X-type stabilizers (star/vertex operators)
- Z-type stabilizers (plaquette/face operators)
- Logical X and Z operators
- Multi-round syndrome extraction
"""

import stim
from typing import List, Tuple, Dict
import numpy as np


class SurfaceCodeLayout:
    """Represents the qubit layout for a surface code."""
    
    def __init__(self, distance: int):
        """
        Create a surface code layout.
        
        Args:
            distance: Code distance (must be odd, >= 3)
        """
        if distance < 3 or distance % 2 == 0:
            raise ValueError("Code distance must be odd and >= 3")
        
        self.distance = distance
        self.data_qubits: List[Tuple[int, int]] = []
        self.x_ancillas: List[Tuple[int, int]] = []  # X-type stabilizer ancillas
        self.z_ancillas: List[Tuple[int, int]] = []  # Z-type stabilizer ancillas
        self._create_layout()
    
    def _create_layout(self):
        """Create the rotated surface code layout."""
        # For a rotated surface code with distance d:
        # - Data qubits: d^2 total
        # - X-stabilizers: (d^2 - 1) / 2
        # - Z-stabilizers: (d^2 - 1) / 2
        
        # Create data qubits in a checkerboard pattern
        for i in range(self.distance):
            for j in range(self.distance):
                self.data_qubits.append((i, j))
        
        # Create X-type ancillas (measure star operators)
        # These are at positions (i+0.5, j+0.5) for even i+j
        for i in range(self.distance - 1):
            for j in range(self.distance - 1):
                if (i + j) % 2 == 0:
                    self.x_ancillas.append((i, j))
        
        # Create Z-type ancillas (measure plaquette operators)
        # These are at positions (i+0.5, j+0.5) for odd i+j
        for i in range(self.distance - 1):
            for j in range(self.distance - 1):
                if (i + j) % 2 == 1:
                    self.z_ancillas.append((i, j))
    
    def get_qubit_index(self, coord: Tuple[int, int], qubit_type: str = 'data') -> int:
        """Get the qubit index for a given coordinate."""
        if qubit_type == 'data':
            return self.data_qubits.index(coord)
        elif qubit_type == 'x_ancilla':
            offset = len(self.data_qubits)
            return offset + self.x_ancillas.index(coord)
        elif qubit_type == 'z_ancilla':
            offset = len(self.data_qubits) + len(self.x_ancillas)
            return offset + self.z_ancillas.index(coord)
        else:
            raise ValueError(f"Unknown qubit type: {qubit_type}")
    
    def get_x_stabilizer_qubits(self, ancilla_coord: Tuple[int, int]) -> List[int]:
        """Get data qubits involved in an X-stabilizer measurement."""
        i, j = ancilla_coord
        neighbors = []
        
        # X-stabilizer acts on 4 surrounding data qubits (in a star pattern)
        potential_neighbors = [
            (i, j), (i+1, j), (i, j+1), (i+1, j+1)
        ]
        
        for coord in potential_neighbors:
            if coord in self.data_qubits:
                neighbors.append(self.get_qubit_index(coord, 'data'))
        
        return neighbors
    
    def get_z_stabilizer_qubits(self, ancilla_coord: Tuple[int, int]) -> List[int]:
        """Get data qubits involved in a Z-stabilizer measurement."""
        i, j = ancilla_coord
        neighbors = []
        
        # Z-stabilizer acts on 4 surrounding data qubits (in a plaquette pattern)
        potential_neighbors = [
            (i, j), (i+1, j), (i, j+1), (i+1, j+1)
        ]
        
        for coord in potential_neighbors:
            if coord in self.data_qubits:
                neighbors.append(self.get_qubit_index(coord, 'data'))
        
        return neighbors
    
    @property
    def total_qubits(self) -> int:
        """Total number of qubits (data + ancillas)."""
        return len(self.data_qubits) + len(self.x_ancillas) + len(self.z_ancillas)


def create_surface_code_circuit(distance: int, rounds: int = 3, 
                                error_rate: float = 0.0,
                                logical_state: str = '0') -> stim.Circuit:
    """
    Create a complete surface code circuit with syndrome extraction.
    
    Uses Stim's built-in surface code generator which properly implements:
    - Rotated surface code layout
    - X and Z stabilizers
    - Proper boundary conditions
    - Correct detector placement
    - Logical observables
    
    Args:
        distance: Code distance (odd, >= 3)
        rounds: Number of syndrome extraction rounds
        error_rate: Depolarizing error rate (0 for noiseless)
        logical_state: Logical state ('0' for Z-basis, '+' for X-basis)
    
    Returns:
        Stim circuit implementing the surface code
    """
    if distance < 3 or distance % 2 == 0:
        raise ValueError("Code distance must be odd and >= 3")
    
    # Use Stim's built-in surface code generator
    # This is a properly constructed rotated surface code with:
    # - Correct stabilizer measurements
    # - Proper detector definitions
    # - Logical observable tracking
    
    if logical_state == '0':
        # Z-basis memory experiment
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=distance,
            rounds=rounds,
            after_clifford_depolarization=error_rate,
            after_reset_flip_probability=error_rate * 0.1 if error_rate > 0 else 0,
            before_measure_flip_probability=error_rate * 0.1 if error_rate > 0 else 0,
        )
    else:
        # X-basis memory experiment
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_x",
            distance=distance,
            rounds=rounds,
            after_clifford_depolarization=error_rate,
            after_reset_flip_probability=error_rate * 0.1 if error_rate > 0 else 0,
            before_measure_flip_probability=error_rate * 0.1 if error_rate > 0 else 0,
        )
    
    return circuit


def create_noisy_surface_code(distance: int, rounds: int, 
                               error_rate: float) -> stim.Circuit:
    """
    Create a noisy surface code circuit for error correction testing.
    
    Args:
        distance: Code distance
        rounds: Number of syndrome extraction rounds
        error_rate: Physical error rate per gate
    
    Returns:
        Noisy surface code circuit with proper noise injection
    """
    return create_surface_code_circuit(distance, rounds, error_rate=error_rate)


def validate_surface_code(distance: int) -> bool:
    """
    Validate that the surface code implementation is correct.
    
    Args:
        distance: Code distance to test
    
    Returns:
        True if validation passes
    """
    print(f"Validating surface code with distance {distance}...")
    
    try:
        # Create circuit
        circuit = create_surface_code_circuit(distance, rounds=3)
        
        # Check that we can create a detector error model
        dem = circuit.detector_error_model(decompose_errors=True)
        
        # Validate structure
        layout = SurfaceCodeLayout(distance)
        expected_qubits = layout.total_qubits
        actual_qubits = circuit.num_qubits
        
        print(f"  Expected qubits: {expected_qubits}")
        print(f"  Actual qubits: {actual_qubits}")
        print(f"  Data qubits: {len(layout.data_qubits)}")
        print(f"  X ancillas: {len(layout.x_ancillas)}")
        print(f"  Z ancillas: {len(layout.z_ancillas)}")
        print(f"  Detectors: {dem.num_detectors}")
        print(f"  Observables: {dem.num_observables}")
        
        assert actual_qubits == expected_qubits, "Qubit count mismatch"
        assert dem.num_detectors > 0, "No detectors found"
        assert dem.num_observables > 0, "No observables found"
        
        print(f" Surface code validation passed for distance {distance}")
        return True
        
    except Exception as e:
        print(f" Surface code validation failed: {e}")
        return False


if __name__ == "__main__":
    # Test the surface code implementation
    for d in [3, 5]:
        validate_surface_code(d)
        print()
