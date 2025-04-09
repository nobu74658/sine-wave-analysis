#!/usr/bin/env python3
"""
Sine Wave Probability Density Function and Cumulative Distribution Function Analysis

This script calculates and visualizes the probability density function (PDF)
and cumulative distribution function (CDF) of a sinusoidal AC signal by random sampling.

Requirements:
- numpy
- matplotlib
- scipy
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import argparse
from typing import Tuple, List, Dict, Any

# Default parameters
DEFAULT_PARAMS = {
    'amplitude': 141,        # Amplitude of the sine wave (V)
    'num_samples': 100000,   # Number of samples for the main analysis
    'num_bins': 50,          # Number of bins for the histogram
    'num_demo_samples': 20,  # Number of samples to show in the demonstration
    'output_path': '/workspace/sine_wave_pdf_and_cdf.png',  # Output file path
    'random_seed': 42,       # Random seed for reproducibility
    'dpi': 300               # DPI for the output image
}

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Calculate and visualize PDF and CDF of a sinusoidal signal'
    )
    parser.add_argument('--amplitude', type=float, default=DEFAULT_PARAMS['amplitude'],
                        help=f'Amplitude of the sine wave in volts (default: {DEFAULT_PARAMS["amplitude"]})')
    parser.add_argument('--num-samples', type=int, default=DEFAULT_PARAMS['num_samples'],
                        help=f'Number of samples for analysis (default: {DEFAULT_PARAMS["num_samples"]})')
    parser.add_argument('--num-bins', type=int, default=DEFAULT_PARAMS['num_bins'],
                        help=f'Number of bins for histogram (default: {DEFAULT_PARAMS["num_bins"]})')
    parser.add_argument('--num-demo-samples', type=int, default=DEFAULT_PARAMS['num_demo_samples'],
                        help=f'Number of demonstration samples (default: {DEFAULT_PARAMS["num_demo_samples"]})')
    parser.add_argument('--output', type=str, default=DEFAULT_PARAMS['output_path'],
                        help=f'Output file path (default: {DEFAULT_PARAMS["output_path"]})')
    parser.add_argument('--seed', type=int, default=DEFAULT_PARAMS['random_seed'],
                        help=f'Random seed (default: {DEFAULT_PARAMS["random_seed"]})')
    parser.add_argument('--dpi', type=int, default=DEFAULT_PARAMS['dpi'],
                        help=f'DPI for output image (default: {DEFAULT_PARAMS["dpi"]})')
    
    return parser.parse_args()

def generate_sine_wave_samples(
    amplitude: float, 
    num_samples: int, 
    random_seed: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate random samples from a sine wave.
    
    Args:
        amplitude: Amplitude of the sine wave
        num_samples: Number of samples to generate
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple containing:
            - Array of random time points
            - Array of corresponding voltage values
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        
    # Generate random time points uniformly distributed between 0 and 1
    random_times = np.random.uniform(0, 1, num_samples)
    
    # Calculate voltage values at these random times (sine wave with period 1)
    voltage_values = amplitude * np.sin(2 * np.pi * random_times)
    
    return random_times, voltage_values

def calculate_histogram(
    values: np.ndarray, 
    num_bins: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Calculate histogram from values.
    
    Args:
        values: Array of values to create histogram from
        num_bins: Number of bins for the histogram
        
    Returns:
        Tuple containing:
            - Histogram values (normalized)
            - Bin edges
            - Bin centers
            - Bin width
    """
    hist, bin_edges = np.histogram(values, bins=num_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
    
    return hist, bin_edges, bin_centers, bin_width

def calculate_theoretical_pdf(
    x: np.ndarray, 
    amplitude: float
) -> np.ndarray:
    """
    Calculate theoretical PDF for a sine wave.
    
    Args:
        x: Array of voltage values
        amplitude: Amplitude of the sine wave
        
    Returns:
        Array of PDF values
    """
    # Formula: f(V) = 1/(π√(A²-V²)) for -A ≤ V ≤ A
    return 1 / (np.pi * np.sqrt(amplitude**2 - x**2))

def calculate_theoretical_cdf(
    x: np.ndarray, 
    amplitude: float
) -> np.ndarray:
    """
    Calculate theoretical CDF for a sine wave.
    
    Args:
        x: Array of voltage values (sorted)
        amplitude: Amplitude of the sine wave
        
    Returns:
        Array of CDF values
    """
    # Formula: F(V) = (1/π)·arcsin(V/A) + 0.5 for -A ≤ V ≤ A
    return (1/np.pi) * np.arcsin(x/amplitude) + 0.5

def calculate_numerical_cdf(
    hist: np.ndarray, 
    bin_centers: np.ndarray, 
    bin_width: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate numerical CDF from histogram data.
    
    Args:
        hist: Histogram values
        bin_centers: Centers of histogram bins
        bin_width: Width of histogram bins
        
    Returns:
        Tuple containing:
            - Sorted bin centers
            - Normalized CDF values
    """
    # Sort bin centers and corresponding histogram values
    sorted_indices = np.argsort(bin_centers)
    sorted_bin_centers = bin_centers[sorted_indices]
    
    # Calculate cumulative sum and normalize
    sorted_cdf = np.cumsum(hist[sorted_indices]) * bin_width
    sorted_cdf = sorted_cdf / sorted_cdf[-1]  # Normalize to ensure it ends at 1.0
    
    return sorted_bin_centers, sorted_cdf

def plot_sine_wave_with_samples(
    ax: plt.Axes,
    amplitude: float,
    times: np.ndarray,
    voltages: np.ndarray
) -> None:
    """
    Plot sine wave with random samples.
    
    Args:
        ax: Matplotlib axes to plot on
        amplitude: Amplitude of the sine wave
        times: Array of random time points
        voltages: Array of corresponding voltage values
    """
    # Generate sine wave for one period
    t = np.linspace(0, 1, 1000)
    v = amplitude * np.sin(2 * np.pi * t)
    
    # Plot sine wave
    ax.plot(t, v, 'b-', linewidth=2, label='Sine Wave')
    
    # Plot random samples
    ax.scatter(times, voltages, color='red', s=50, label=f'Random Samples (n={len(times)})')
    
    # Add vertical lines to show sampling
    for i in range(len(times)):
        ax.plot([times[i], times[i]], [0, voltages[i]], 'r--', alpha=0.3)
    
    # Set labels and limits
    ax.set_title('Sinusoidal AC Signal with Random Time Sampling')
    ax.set_xlabel('Time (normalized)')
    ax.set_ylabel('Voltage (V)')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(-amplitude*1.1, amplitude*1.1)
    ax.legend()

def plot_histogram(
    ax: plt.Axes,
    bin_centers: np.ndarray,
    hist: np.ndarray,
    bin_width: float,
    amplitude: float,
    title: str,
    color: str = 'green'
) -> None:
    """
    Plot histogram of voltage values.
    
    Args:
        ax: Matplotlib axes to plot on
        bin_centers: Centers of histogram bins
        hist: Histogram values
        bin_width: Width of histogram bins
        amplitude: Amplitude of the sine wave
        title: Title for the plot
        color: Color for the histogram bars
    """
    ax.bar(bin_centers, hist, width=bin_width, alpha=0.6, color=color)
    ax.set_title(title)
    ax.set_xlabel('Voltage (V)')
    ax.set_ylabel('Probability Density')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-amplitude*1.05, amplitude*1.05)

def plot_pdf_comparison(
    ax: plt.Axes,
    bin_centers: np.ndarray,
    hist: np.ndarray,
    bin_width: float,
    x: np.ndarray,
    pdf: np.ndarray,
    amplitude: float
) -> None:
    """
    Plot comparison of numerical and theoretical PDFs.
    
    Args:
        ax: Matplotlib axes to plot on
        bin_centers: Centers of histogram bins
        hist: Histogram values
        bin_width: Width of histogram bins
        x: Array of voltage values for theoretical PDF
        pdf: Theoretical PDF values
        amplitude: Amplitude of the sine wave
    """
    ax.bar(bin_centers, hist, width=bin_width, alpha=0.6, 
           label='Numerical PDF (Histogram)', color='green')
    ax.plot(x, pdf, 'r-', linewidth=2, label='Theoretical PDF: 1/(π√(A²-V²))')
    ax.set_title('Probability Density Function Comparison')
    ax.set_xlabel('Voltage (V)')
    ax.set_ylabel('Probability Density')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(-amplitude*1.05, amplitude*1.05)
    ax.set_ylim(0, max(pdf)*1.1)

def plot_cdf_comparison(
    ax: plt.Axes,
    sorted_bin_centers: np.ndarray,
    sorted_cdf: np.ndarray,
    x_sorted: np.ndarray,
    theoretical_cdf: np.ndarray,
    amplitude: float
) -> None:
    """
    Plot comparison of numerical and theoretical CDFs.
    
    Args:
        ax: Matplotlib axes to plot on
        sorted_bin_centers: Sorted centers of histogram bins
        sorted_cdf: Numerical CDF values
        x_sorted: Sorted array of voltage values for theoretical CDF
        theoretical_cdf: Theoretical CDF values
        amplitude: Amplitude of the sine wave
    """
    ax.plot(sorted_bin_centers, sorted_cdf, 'g-', linewidth=2, label='Numerical CDF')
    ax.plot(x_sorted, theoretical_cdf, 'r--', linewidth=2, 
            label='Theoretical CDF: (1/π)arcsin(V/A) + 0.5')
    ax.set_title('Cumulative Distribution Function (CDF)')
    ax.set_xlabel('Voltage (V)')
    ax.set_ylabel('Cumulative Probability')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(-amplitude*1.05, amplitude*1.05)
    ax.set_ylim(0, 1.05)

def create_plots(params: Dict[str, Any]) -> plt.Figure:
    """
    Create all plots for sine wave PDF and CDF analysis.
    
    Args:
        params: Dictionary of parameters
        
    Returns:
        Matplotlib figure containing all plots
    """
    # Extract parameters
    amplitude = params['amplitude']
    num_samples = params['num_samples']
    num_bins = params['num_bins']
    num_demo_samples = params['num_demo_samples']
    random_seed = params['random_seed']
    
    # Generate samples
    random_times, voltage_values = generate_sine_wave_samples(
        amplitude, num_samples, random_seed)
    demo_times, demo_voltages = generate_sine_wave_samples(
        amplitude, num_demo_samples, random_seed + 1)  # Use different seed for demo
    
    # Calculate histogram for main samples
    hist, bin_edges, bin_centers, bin_width = calculate_histogram(
        voltage_values, num_bins)
    
    # Calculate histogram for demo samples
    demo_hist, demo_bin_edges, demo_bin_centers, demo_bin_width = calculate_histogram(
        demo_voltages, 10)  # Fewer bins for demo
    
    # Calculate theoretical PDF
    x = np.linspace(-amplitude+0.001, amplitude-0.001, 1000)  # Avoid division by zero
    pdf = calculate_theoretical_pdf(x, amplitude)
    
    # Calculate numerical CDF
    sorted_bin_centers, sorted_cdf = calculate_numerical_cdf(
        hist, bin_centers, bin_width)
    
    # Calculate theoretical CDF
    x_sorted = np.sort(x)
    theoretical_cdf = calculate_theoretical_cdf(x_sorted, amplitude)
    
    # Create figure with 5 subplots
    fig = plt.figure(figsize=(14, 16))
    
    # Plot 1: Sine wave with samples
    ax1 = fig.add_subplot(5, 1, 1)
    plot_sine_wave_with_samples(ax1, amplitude, demo_times, demo_voltages)
    
    # Plot 2: Histogram of demo samples
    ax2 = fig.add_subplot(5, 1, 2)
    plot_histogram(
        ax2, demo_bin_centers, demo_hist, demo_bin_width, amplitude,
        f'Histogram of Demonstration Samples (n={num_demo_samples})', 'orange')
    
    # Plot 3: Histogram of all samples
    ax3 = fig.add_subplot(5, 1, 3)
    plot_histogram(
        ax3, bin_centers, hist, bin_width, amplitude,
        f'Histogram of Voltage Values (n={num_samples}, bins={num_bins})')
    
    # Plot 4: PDF comparison
    ax4 = fig.add_subplot(5, 1, 4)
    plot_pdf_comparison(ax4, bin_centers, hist, bin_width, x, pdf, amplitude)
    
    # Plot 5: CDF comparison
    ax5 = fig.add_subplot(5, 1, 5)
    plot_cdf_comparison(
        ax5, sorted_bin_centers, sorted_cdf, x_sorted, theoretical_cdf, amplitude)
    
    plt.tight_layout()
    
    return fig, voltage_values, bin_width

def print_statistics_and_explanation(
    voltage_values: np.ndarray, 
    bin_width: float, 
    amplitude: float
) -> None:
    """
    Print statistical analysis and explanation of the results.
    
    Args:
        voltage_values: Array of voltage values
        bin_width: Width of histogram bins
        amplitude: Amplitude of the sine wave
    """
    print("Statistical Analysis:")
    print(f"Min voltage: {voltage_values.min():.2f} V")
    print(f"Max voltage: {voltage_values.max():.2f} V")
    print(f"Mean voltage: {voltage_values.mean():.2f} V")
    print(f"Standard deviation: {voltage_values.std():.2f} V")
    print(f"Bin width: {bin_width:.2f} V")

    print("\nExplanation of the Probability Density Function and CDF of a Sinusoidal Signal:")
    print("1. Mathematical Background:")
    print("   - For a sine wave V(t) = A·sin(2πt), the PDF follows the arcsine distribution")
    print("   - The PDF formula is: f(V) = 1/(π√(A²-V²)) for -A ≤ V ≤ A")
    print("   - The CDF formula is: F(V) = (1/π)·arcsin(V/A) + 0.5 for -A ≤ V ≤ A")
    print("   - This is because the time spent at any voltage is inversely proportional")
    print("     to the rate of change (derivative) of the sine function at that point")

    print("\n2. Key Characteristics:")
    print("   - The PDF is symmetric around V = 0")
    print("   - It approaches infinity at the extremes (V = ±A) because the sine wave")
    print("     slows down and spends more time near its peaks")
    print("   - It has its minimum at V = 0 where the sine wave crosses zero most rapidly")
    print("   - The mean value is 0V (for a complete cycle)")
    print("   - The standard deviation is A/√2 ≈ 0.707A")
    print("   - The CDF is S-shaped, starting at 0 for V = -A and ending at 1 for V = A")
    print("   - The CDF has its steepest slope at V = 0, where the PDF has its minimum")

    print("\n3. Sampling Process:")
    print("   - We randomly selected time points from a uniform distribution over one period")
    print("   - At each time point, we calculated the corresponding voltage")
    print("   - The histogram of these voltage values approximates the theoretical PDF")
    print("   - With more samples, the histogram converges to the theoretical distribution")

    print("\n4. Practical Implications:")
    print("   - In AC circuits, the voltage spends more time near peak values than near zero")
    print("   - This affects measurements, power calculations, and component stress")
    print("   - The RMS (root mean square) value of a sine wave is A/√2 ≈ 0.707A")
    print(f"   - For our amplitude of {amplitude}V, the RMS value is approximately {amplitude/np.sqrt(2):.1f}V")
    print("   - The CDF is useful for determining the probability that the voltage")
    print("     will be less than or equal to a specific value")
    print("   - For example, the probability that the voltage is less than or equal to 0V is 0.5")

def main():
    """Main function to run the analysis."""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Create parameter dictionary
        params = {
            'amplitude': args.amplitude,
            'num_samples': args.num_samples,
            'num_bins': args.num_bins,
            'num_demo_samples': args.num_demo_samples,
            'output_path': args.output,
            'random_seed': args.seed,
            'dpi': args.dpi
        }
        
        # Create plots
        fig, voltage_values, bin_width = create_plots(params)
        
        # Save figure
        output_dir = os.path.dirname(params['output_path'])
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        fig.savefig(params['output_path'], dpi=params['dpi'], bbox_inches='tight')
        print(f"グラフを '{params['output_path']}' に保存しました。")
        
        # Print statistics and explanation
        print_statistics_and_explanation(voltage_values, bin_width, params['amplitude'])
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)