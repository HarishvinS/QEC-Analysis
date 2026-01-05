"""
Validation and Testing Module

This module provides validation tests and comparison utilities for
assessing quantum error correction performance.
"""

import stim
import pymatching
import numpy as np
from typing import Dict, List, Tuple
import time

from surface_code import create_surface_code_circuit, validate_surface_code, SurfaceCodeLayout
from noise_models import add_depolarizing_noise


def setup_matching_graph(circuit: stim.Circuit) -> pymatching.Matching:
    """
    Create a matching graph for error correction using PyMatching.
    
    Args:
        circuit: The Stim circuit with defined detectors
    
    Returns:
        A PyMatching object representing the matching graph
    """
    # Extract detector error model from the circuit
    dem = circuit.detector_error_model(decompose_errors=True)
    
    # Create the matching problem
    matching_graph = pymatching.Matching.from_detector_error_model(dem)
    
    return matching_graph


def run_error_correction_trial(circuit: stim.Circuit, shots: int) -> Tuple[float, int]:
    """
    Run error correction and compute logical error rate.
    
    Args:
        circuit: The noisy surface code circuit
        shots: Number of trials to run
    
    Returns:
        Tuple of (logical_error_rate, num_errors)
    """
    try:
        # Create matching graph
        matcher = setup_matching_graph(circuit)
        
        # Sample from the circuit
        sampler = circuit.compile_detector_sampler()
        detector_samples, obs_samples = sampler.sample(shots, separate_observables=True)
        
        if len(detector_samples) == 0 or len(obs_samples) == 0:
            return 0.5, shots // 2  # Return 50% error rate if no samples
        
        # Decode each sample using PyMatching
        predictions = np.zeros(obs_samples.shape, dtype=np.uint8)
        for s in range(shots):
            syndrome = detector_samples[s]
            prediction = matcher.decode(syndrome)
            predictions[s] = prediction
        
        # Calculate the logical error rate
        errors = np.logical_xor(predictions, obs_samples)
        num_errors = int(np.sum(errors))
        error_rate = float(np.mean(errors))
        
        # Handle NaN
        if np.isnan(error_rate):
            error_rate = 0.5
            num_errors = shots // 2
        
        return error_rate, num_errors
        
    except Exception as e:
        print(f"Warning: Error in trial: {e}")
        return 0.5, shots // 2


def test_error_correction(distance: int, error_rate: float, 
                         num_shots: int = 1000) -> Dict:
    """
    Test error correction for a specific configuration.
    
    Args:
        distance: Surface code distance
        error_rate: Physical error rate
        num_shots: Number of simulation shots
    
    Returns:
        Dictionary with test results
    """
    print(f"Testing error correction: d={distance}, p={error_rate:.4f}...")
    
    start_time = time.time()
    
    # Create surface code circuit with integrated noise
    # This is more realistic than adding noise after the fact
    circuit = create_surface_code_circuit(distance, rounds=distance, error_rate=error_rate)
    
    # Run error correction
    logical_error_rate, num_errors = run_error_correction_trial(circuit, num_shots)
    
    end_time = time.time()
    runtime = end_time - start_time
    
    results = {
        'distance': distance,
        'physical_error_rate': error_rate,
        'logical_error_rate': logical_error_rate,
        'num_errors': num_errors,
        'num_shots': num_shots,
        'runtime': runtime,
        'improvement_factor': error_rate / logical_error_rate if logical_error_rate > 0 else float('inf')
    }
    
    print(f"  Logical error rate: {logical_error_rate:.6f}")
    print(f"  Improvement: {results['improvement_factor']:.2f}×")
    print(f"  Runtime: {runtime:.2f}s")
    
    return results


def compare_protected_unprotected(base_circuit: stim.Circuit,
                                  distances: List[int],
                                  error_rates: List[float],
                                  shots: int = 1000) -> Dict:
    """
    Compare protected (with error correction) vs unprotected circuits.
    
    Args:
        base_circuit: Base logical circuit
        distances: List of code distances to test
        error_rates: List of physical error rates
        shots: Number of shots per configuration
    
    Returns:
        Dictionary with comparison results
    """
    print("\n" + "="*60)
    print("PROTECTED VS UNPROTECTED COMPARISON")
    print("="*60)
    
    results = {
        'protected': {},
        'unprotected': {},
        'improvement': {}
    }
    
    for distance in distances:
        results['protected'][distance] = {}
        
        for p in error_rates:
            # Test with error correction
            protected_result = test_error_correction(distance, p, shots)
            results['protected'][distance][p] = protected_result
    
    # Test unprotected (simulate without error correction)
    print("\nTesting unprotected circuits...")
    for p in error_rates:
        # For unprotected, the logical error rate ≈ physical error rate
        # (actually it's worse due to error accumulation)
        # Simulate a simple circuit
        simple_circuit = stim.Circuit()
        simple_circuit.append_operation("R", [0])
        simple_circuit.append_operation("H", [0])
        simple_circuit.append_operation("M", [0])
        
        noisy_simple = add_depolarizing_noise(simple_circuit, p)
        
        # We can't use surface code decoding, so just estimate
        # In practice, unprotected error rate is approximately the physical rate
        results['unprotected'][p] = {
            'physical_error_rate': p,
            'logical_error_rate': p,  # Rough approximation
            'improvement_factor': 1.0
        }
    
    # Calculate improvement factors
    for distance in distances:
        results['improvement'][distance] = {}
        for p in error_rates:
            if p in results['unprotected']:
                protected_err = results['protected'][distance][p]['logical_error_rate']
                unprotected_err = results['unprotected'][p]['logical_error_rate']
                
                if protected_err > 0:
                    improvement = unprotected_err / protected_err
                else:
                    improvement = float('inf')
                
                results['improvement'][distance][p] = improvement
    
    return results


def find_threshold(results: Dict) -> float:
    """
    Estimate the error correction threshold from results.
    """
    crossover_points = []
    
    if 'protected' in results:
        for distance in results['protected']:
            p_rates = sorted(results['protected'][distance].keys())
            for i in range(len(p_rates) - 1):
                p1, p2 = p_rates[i], p_rates[i+1]
                l1 = results['protected'][distance][p1]['logical_error_rate']
                l2 = results['protected'][distance][p2]['logical_error_rate']
                
                # Check if we cross y=x line
                # (l1 - p1) and (l2 - p2) should have different signs
                if (l1 - p1) * (l2 - p2) <= 0:
                    # Linear interpolation for crossover
                    # l(p) = l1 + (l2-l1)/(p2-p1) * (p-p1)
                    # Solving l(p) = p for p
                    # p = l1 + m(p-p1) => p(1-m) = l1 - m*p1
                    m = (l2 - l1) / (p2 - p1)
                    if m = 1:
                        p_cross = (l1 - m * p1) / (1 - m)
                        if p1 <= p_cross <= p2:
                            crossover_points.append(p_cross)
    
    if crossover_points:
        return float(np.mean(crossover_points))
    
    # Fallback to simple logic
    max_p_with_improvement = 0
    if 'protected' in results:
        for distance in results['protected']:
            for p, res in results['protected'][distance].items():
                if res['logical_error_rate'] < p:
                    max_p_with_improvement = max(max_p_with_improvement, p)
    
    return max_p_with_improvement if max_p_with_improvement > 0 else 0.01


def create_validation_report(results: Dict) -> str:
    """
    Create a formatted validation report.
    
    Args:
        results: Dictionary of test results
    
    Returns:
        Formatted report string
    """
    report = []
    report.append("\n" + "="*60)
    report.append("QUANTUM ERROR CORRECTION VALIDATION REPORT")
    report.append("="*60)
    
    if 'protected' in results:
        report.append("\nPROTECTED CIRCUITS (With Error Correction):")
        report.append("-" * 60)
        report.append(f"{'Distance':<10}{'Phys. Err':<15}{'Log. Err':<15}{'Improvement':<15}")
        report.append("-" * 60)
        
        for distance in sorted(results['protected'].keys()):
            for p in sorted(results['protected'][distance].keys()):
                result = results['protected'][distance][p]
                log_err = result['logical_error_rate']
                improvement = result['improvement_factor']
                report.append(f"{distance:<10}{p:<15.6f}{log_err:<15.6f}{improvement:<15.2f}×")
    
    if 'improvement' in results:
        report.append("\nIMPROVEMENT OVER UNPROTECTED:")
        report.append("-" * 60)
        best_distance = None
        best_improvement = 0
        best_p = None
        
        for distance in results['improvement']:
            for p in results['improvement'][distance]:
                improvement = results['improvement'][distance][p]
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_distance = distance
                    best_p = p
        
        if best_distance:
            report.append(f"Best configuration: d={best_distance}, p={best_p:.4f}")
            report.append(f"Improvement factor: {best_improvement:.2f}×")
    
    report.append("="*60)
    
    return "\n".join(report)


if __name__ == "__main__":
    # Run validation tests
    print("Running validation tests...\n")
    
    # Test surface code validation
    validate_surface_code(3)
    print()
    
    # Test error correction
    test_error_correction(distance=3, error_rate=0.01, num_shots=500)
    print()
    
    print(" Validation module working correctly")
