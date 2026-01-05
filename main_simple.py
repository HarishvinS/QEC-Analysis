"""
Simplified Main Analysis - Focused on Core Functionality

Runs a smaller parameter sweep for faster testing and debugging.
"""

import stim
import numpy as np
from typing import List, Dict
import time

from surface_code import create_surface_code_circuit, validate_surface_code
from quantum_algorithms import test_shor_circuit, test_grover_circuit
from noise_models import add_depolarizing_noise
from validation import test_error_correction, compare_protected_unprotected, find_threshold
import matplotlib.pyplot as plt

print("="*70)
print(" "*15 + "QUANTUM ERROR CORRECTION PROJECT v2.0")
print(" "*20 + "Simplified Analysis")
print("="*70 + "\n")

# 1. Quick validation
print("Validating surface code (d=3)...")
validate_surface_code(3)
print()

# 2. Test algorithms
print("Testing quantum algorithms...")
test_shor_circuit()
print()
test_grover_circuit()
print()

# 3. Run smaller parameter sweep
print("="*60)
print("RUNNING ANALYSIS")
print("="*60)

code_distances = [3, 5]  # Just 2 distances
error_rates = [0.001, 0.005, 0.01, 0.02]  # Just 4 error rates
shots = 500  # Fewer shots for speed

results = {'protected': {}}

for distance in code_distances:
    results['protected'][distance] = {}
    
    for error_rate in error_rates:
        print(f"\\nTesting d={distance}, p={error_rate:.4f}...")
        result = test_error_correction(distance, error_rate, shots)
        results['protected'][distance][error_rate] = result

# 4. Create simple plot
print("\\n" + "="*60)
print("CREATING VISUALIZATION")
print("="*60)

plt.figure(figsize=(10, 7))

for distance in sorted(results['protected'].keys()):
    distance_results = results['protected'][distance]
    
    physical_errors = sorted(distance_results.keys())
    logical_errors = [distance_results[p]['logical_error_rate'] for p in physical_errors]
    
    # Filter out zeros for log plot
    valid_indices = [i for i, val in enumerate(logical_errors) if val > 0]
    if valid_indices:
        physical_errors_filtered = [physical_errors[i] for i in valid_indices]
        logical_errors_filtered = [logical_errors[i] for i in valid_indices]
        
        plt.plot(physical_errors_filtered, logical_errors_filtered, marker='o', 
                label=f"d = {distance}", linewidth=2, markersize=8)

# Reference line
p_range = np.linspace(min(error_rates), max(error_rates), 100)
plt.plot(p_range, p_range, 'k--', alpha=0.5, linewidth=2, label="No improvement")

plt.xlabel("Physical Error Rate", fontsize=12, fontweight='bold')
plt.ylabel("Logical Error Rate", fontsize=12, fontweight='bold')
plt.title("Quantum Error Correction Performance", fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig("simple_qec_results.png", dpi=300, bbox_inches='tight')
plt.close()

print("\\n Saved plot to simple_qec_results.png")

# 5. Print summary
print("\\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"\\nTested configurations: {len(code_distances) * len(error_rates)}")
print(f"Code distances: {code_distances}")
print(f"Error rates: {error_rates}")

# Find best config
best_config = {'distance': None, 'p': None, 'logical_err': float('inf')}

for distance in results['protected']:
    for p in results['protected'][distance]:
        result = results['protected'][distance][p]
        logical_err = result['logical_error_rate']
        
        if logical_err < best_config['logical_err']:
            best_config = {
                'distance': distance,
                'p': p,
                'logical_err': logical_err,
                'improvement': result['improvement_factor']
            }

if best_config['distance']:
    print(f"\\nBest Configuration:")
    print(f"  Distance: {best_config['distance']}")
    print(f"  Physical Error Rate: {best_config['p']:.4f}")
    print(f"  Logical Error Rate: {best_config['logical_err']:.6f}")
    print(f"  Improvement: {best_config['improvement']:.2f}×")

print("\\n Analysis complete")
print("="*70 + "\\n")
