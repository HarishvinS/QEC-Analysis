"""
Comprehensive Visualization Suite

This module creates publication-quality plots for analyzing
quantum error correction performance.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List

# Set style for publication-quality plots
plt.style.use('default')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True


def plot_logical_vs_physical_errors(results: Dict, output_file: str = "error_rates.png"):
    """
    Plot logical error rate vs physical error rate for different code distances.
    
    Args:
        results: Dictionary of simulation results
        output_file: Output filename
    """
    plt.figure(figsize=(10, 7))
    
    if 'protected' in results:
        for distance in sorted(results['protected'].keys()):
            distance_results = results['protected'][distance]
            
            # Extract data
            physical_errors = sorted(distance_results.keys())
            logical_errors = [distance_results[p]['logical_error_rate'] for p in physical_errors]
            
            # Plot
            plt.plot(physical_errors, logical_errors, marker='o', 
                    label=f"d = {distance}", linewidth=2, markersize=8)
    
    # Add reference line (logical = physical)
    max_p = max([max(results['protected'][d].keys()) for d in results['protected']])
    min_p = min([min(results['protected'][d].keys()) for d in results['protected']])
    p_range = np.logspace(np.log10(min_p), np.log10(max_p), 100)
    plt.plot(p_range, p_range, 'k--', alpha=0.5, linewidth=2, label="No improvement")
    
    plt.xlabel("Physical Error Rate (p)", fontsize=12, fontweight='bold')
    plt.ylabel("Logical Error Rate", fontsize=12, fontweight='bold')
    plt.title("Quantum Error Correction Performance\nLogical vs Physical Error Rates", 
             fontsize=14, fontweight='bold')
    
    # Try log scale, fall back to linear if it fails
    try:
        plt.xscale('log')
        plt.yscale('log')
    except Exception as e:
        print(f"  Note: Using linear scale due to: {e}")
    
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f" Saved error rate plot to {output_file}")


def plot_threshold_analysis(results: Dict, threshold: float, 
                            output_file: str = "threshold.png"):
    """
    Create threshold plot highlighting the error correction threshold.
    
    Args:
        results: Dictionary of simulation results
        threshold: Estimated threshold value
        output_file: Output filename
    """
    plt.figure(figsize=(10, 7))
    
    if 'protected' in results:
        for distance in sorted(results['protected'].keys()):
            distance_results = results['protected'][distance]
            
            physical_errors = sorted(distance_results.keys())
            improvement_factors = [
                distance_results[p]['improvement_factor'] 
                for p in physical_errors
            ]
            
            plt.plot(physical_errors, improvement_factors, marker='s', 
                    label=f"d = {distance}", linewidth=2, markersize=8)
    
    # Add threshold line
    plt.axvline(x=threshold, color='red', linestyle='--', linewidth=2.5, 
               label=f"Threshold ≈ {threshold:.3f}", alpha=0.7)
    
    # Add reference line at 1 (no improvement)
    plt.axhline(y=1, color='gray', linestyle=':', linewidth=1.5, alpha=0.5)
    
    plt.xlabel("Physical Error Rate (p)", fontsize=12, fontweight='bold')
    plt.ylabel("Improvement Factor (p / p_logical)", fontsize=12, fontweight='bold')
    plt.title("Error Correction Threshold Analysis\nImprovement vs Physical Error Rate", 
             fontsize=14, fontweight='bold')
    
    # Try log scale, fall back to linear if it fails
    try:
        plt.xscale('log')
        plt.yscale('log')
    except:
        pass
    
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f" Saved threshold plot to {output_file}")


def plot_protected_vs_unprotected(comparison_results: Dict, 
                                 output_file: str = "protected_vs_unprotected.png"):
    """
    Compare protected and unprotected circuits.
    
    Args:
        comparison_results: Comparison results dictionary
        output_file: Output filename
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Error rates
    has_data_ax1 = False
    if 'protected' in comparison_results and comparison_results['protected']:
        # Get a representative distance
        distances = sorted(comparison_results['protected'].keys())
        representative_d = distances[len(distances)//2]  # Middle distance
        
        distance_results = comparison_results['protected'][representative_d]
        physical_errors = sorted(distance_results.keys())
        logical_errors_protected = [distance_results[p]['logical_error_rate'] 
                                   for p in physical_errors]
        
        # Filter out zeros for plotting
        valid_data = [(p, l) for p, l in zip(physical_errors, logical_errors_protected) if l > 0]
        if valid_data:
            physical_errors_filtered, logical_errors_filtered = zip(*valid_data)
            ax1.plot(physical_errors_filtered, logical_errors_filtered, marker='o', 
                    label=f"Protected (d={representative_d})", linewidth=2.5, markersize=8,
                    color='green')
            has_data_ax1 = True
    
    if 'unprotected' in comparison_results:
        physical_errors = sorted(comparison_results['unprotected'].keys())
        logical_errors_unprotected = [comparison_results['unprotected'][p]['logical_error_rate'] 
                                     for p in physical_errors]
        
        ax1.plot(physical_errors, logical_errors_unprotected, marker='x', 
                label="Unprotected", linewidth=2.5, markersize=10,
                color='red', linestyle='--')
        has_data_ax1 = True
    
    ax1.set_xlabel("Physical Error Rate", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Logical Error Rate", fontsize=11, fontweight='bold')
    ax1.set_title("Error Correction Benefit", fontsize=12, fontweight='bold')
    
    # Try log scale only if we have valid data
    if has_data_ax1:
        try:
            ax1.set_xscale('log')
            ax1.set_yscale('log')
        except:
            pass  # Fall back to linear
    
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Plot 2: Improvement factor
    has_data_ax2 = False
    if 'improvement' in comparison_results and comparison_results['improvement']:
        for distance in sorted(comparison_results['improvement'].keys()):
            physical_errors = sorted(comparison_results['improvement'][distance].keys())
            improvements = [comparison_results['improvement'][distance][p] 
                          for p in physical_errors]
            
            # Filter out infinities and zeros for plotting
            valid_data = [(p, i) for p, i in zip(physical_errors, improvements) 
                         if 0 < i < 1000]  # Filter infinities and zeros
            
            if valid_data:
                physical_errors_filtered, improvements_filtered = zip(*valid_data)
                ax2.plot(physical_errors_filtered, improvements_filtered, marker='o', 
                        label=f"d = {distance}", linewidth=2, markersize=8)
                has_data_ax2 = True
        
        if has_data_ax2:
            ax2.axhline(y=1, color='gray', linestyle=':', linewidth=1.5, alpha=0.5,
                       label="No benefit")
        
        ax2.set_xlabel("Physical Error Rate", fontsize=11, fontweight='bold')
        ax2.set_ylabel("Improvement Factor", fontsize=11, fontweight='bold')
        ax2.set_title("Protected vs Unprotected", fontsize=12, fontweight='bold')
        
        # Only use log scale if we have valid data
        if has_data_ax2:
            try:
                ax2.set_xscale('log')
                ax2.set_yscale('log')
            except:
                pass  # Fall back to linear
        
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
    else:
        # No improvement data, add a note
        ax2.text(0.5, 0.5, 'No valid improvement data\n(check error correction results)', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_xlabel("Physical Error Rate", fontsize=11, fontweight='bold')
        ax2.set_ylabel("Improvement Factor", fontsize=11, fontweight='bold')
        ax2.set_title("Protected vs Unprotected", fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f" Saved comparison plot to {output_file}")


def plot_resource_scaling(results: Dict, output_file: str = "resource_scaling.png"):
    """
    Plot resource requirements vs code distance.
    
    Args:
        results: Dictionary of simulation results
        output_file: Output filename
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    if 'protected' in results:
        distances = sorted(results['protected'].keys())
        
        # Calculate resources
        # For rotated surface code: data qubits = d^2
        data_qubits = [d**2 for d in distances]
        # Ancilla qubits ≈ d^2 - 1
        ancilla_qubits = [d**2 - 1 for d in distances]
        total_qubits = [d + a for d, a in zip(data_qubits, ancilla_qubits)]
        
        # Plot qubit count
        ax1.plot(distances, data_qubits, marker='o', label="Data Qubits", 
                linewidth=2.5, markersize=8)
        ax1.plot(distances, ancilla_qubits, marker='s', label="Ancilla Qubits", 
                linewidth=2.5, markersize=8)
        ax1.plot(distances, total_qubits, marker='^', label="Total Qubits", 
                linewidth=2.5, markersize=8, color='red')
        
        ax1.set_xlabel("Code Distance", fontsize=11, fontweight='bold')
        ax1.set_ylabel("Number of Qubits", fontsize=11, fontweight='bold')
        ax1.set_title("Qubit Requirements", fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        ax1.set_xticks(distances)
        
        # Plot runtime
        # Extract runtimes for a representative error rate
        if distances and results['protected'][distances[0]]:
            rep_error_rate = sorted(results['protected'][distances[0]].keys())[0]
            
            runtimes = []
            for d in distances:
                if rep_error_rate in results['protected'][d]:
                    runtimes.append(results['protected'][d][rep_error_rate]['runtime'])
                else:
                    runtimes.append(0)
            
            ax2.plot(distances, runtimes, marker='o', linewidth=2.5, 
                    markersize=8, color='purple')
            ax2.set_xlabel("Code Distance", fontsize=11, fontweight='bold')
            ax2.set_ylabel("Runtime (seconds)", fontsize=11, fontweight='bold')
            ax2.set_title(f"Computational Cost (p={rep_error_rate:.4f})", 
                         fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.set_xticks(distances)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f" Saved resource scaling plot to {output_file}")


def plot_algorithm_performance(algorithm_results: Dict, 
                               output_file: str = "algorithm_performance.png"):
    """
    Plot performance of different quantum algorithms under error correction.
    
    Args:
        algorithm_results: Results for different algorithms
        output_file: Output filename
    """
    plt.figure(figsize=(10, 7))
    
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    
    for i, (algo_name, results) in enumerate(algorithm_results.items()):
        if 'protected' in results:
            # Get data for first distance
            distances = sorted(results['protected'].keys())
            if distances:
                distance_results = results['protected'][distances[0]]
                
                physical_errors = sorted(distance_results.keys())
                logical_errors = [distance_results[p]['logical_error_rate'] 
                                for p in physical_errors]
                
                plt.plot(physical_errors, logical_errors, marker='o', 
                        label=algo_name, linewidth=2.5, markersize=8,
                        color=colors[i % len(colors)])
    
    plt.xlabel("Physical Error Rate", fontsize=12, fontweight='bold')
    plt.ylabel("Logical Error Rate", fontsize=12, fontweight='bold')
    plt.title("Algorithm Performance Comparison\nError Correction Effectiveness", 
             fontsize=14, fontweight='bold')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f" Saved algorithm performance plot to {output_file}")


def create_comprehensive_plots(all_results: Dict, comparison_results: Dict, 
                               threshold: float):
    """
    Create all visualization plots.
    
    Args:
        all_results: Complete simulation results
        comparison_results: Protected vs unprotected comparison
        threshold: Estimated error threshold
    """
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60 + "\n")
    
    # Main error rate plot
    plot_logical_vs_physical_errors(all_results, "logical_vs_physical_errors.png")
    
    # Threshold analysis
    plot_threshold_analysis(all_results, threshold, "threshold_analysis.png")
    
    # Protected vs unprotected comparison
    if comparison_results:
        plot_protected_vs_unprotected(comparison_results, "protected_vs_unprotected.png")
    
    # Resource scaling
    plot_resource_scaling(all_results, "resource_scaling.png")
    
    print("\n All visualizations created successfully")


if __name__ == "__main__":
    # Test visualization with mock data
    print("Testing visualization module with mock data...\n")
    
    mock_results = {
        'protected': {
            3: {
                0.001: {'logical_error_rate': 0.0005, 'improvement_factor': 2.0, 'runtime': 1.5},
                0.01: {'logical_error_rate': 0.008, 'improvement_factor': 1.25, 'runtime': 1.6},
                0.05: {'logical_error_rate': 0.06, 'improvement_factor': 0.83, 'runtime': 1.7},
            },
            5: {
                0.001: {'logical_error_rate': 0.0001, 'improvement_factor': 10.0, 'runtime': 3.2},
                0.01: {'logical_error_rate': 0.002, 'improvement_factor': 5.0, 'runtime': 3.5},
                0.05: {'logical_error_rate': 0.055, 'improvement_factor': 0.91, 'runtime': 3.8},
            }
        }
    }
    
    plot_logical_vs_physical_errors(mock_results, "test_error_rates.png")
    plot_threshold_analysis(mock_results, 0.015, "test_threshold.png")
    plot_resource_scaling(mock_results, "test_resources.png")
    
    print("\n Visualization module working correctly")
