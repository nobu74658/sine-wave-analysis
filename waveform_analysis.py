#!/usr/bin/env python3
"""
Waveform Probability Density Function and Cumulative Distribution Function Analysis

This script calculates and visualizes the probability density function (PDF)
and cumulative distribution function (CDF) of various waveforms (sine, triangle, square)
by random sampling.

Requirements:
- numpy
- matplotlib
- scipy
- pandas (for CSV export)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import argparse
import pandas as pd
from typing import Tuple, List, Dict, Any, Callable

# Waveform functions
def sine_wave(t: np.ndarray, amplitude: float) -> np.ndarray:
    """Generate sine wave values."""
    return amplitude * np.sin(2 * np.pi * t)

def triangle_wave(t: np.ndarray, amplitude: float) -> np.ndarray:
    """Generate triangle wave values."""
    return amplitude * (2 * np.abs(2 * (t - np.floor(t + 0.5))) - 1)

def square_wave(t: np.ndarray, amplitude: float) -> np.ndarray:
    """Generate square wave values."""
    return amplitude * np.sign(np.sin(2 * np.pi * t))

# Waveform theoretical PDFs
def sine_wave_pdf(x: np.ndarray, amplitude: float) -> np.ndarray:
    """Theoretical PDF for sine wave."""
    # Avoid division by zero at endpoints
    mask = (x > -amplitude) & (x < amplitude)
    result = np.zeros_like(x, dtype=float)
    result[mask] = 1 / (np.pi * np.sqrt(amplitude**2 - x[mask]**2))
    return result

def triangle_wave_pdf(x: np.ndarray, amplitude: float) -> np.ndarray:
    """Theoretical PDF for triangle wave."""
    # Triangle wave has constant PDF within its range
    result = np.zeros_like(x, dtype=float)
    mask = (x >= -amplitude) & (x <= amplitude)
    result[mask] = 1 / (2 * amplitude)
    return result

def square_wave_pdf(x: np.ndarray, amplitude: float) -> np.ndarray:
    """Theoretical PDF for square wave."""
    # Square wave has two delta functions at ±amplitude
    # For visualization, we use narrow Gaussians
    result = np.zeros_like(x, dtype=float)
    sigma = amplitude * 0.01  # Width of the Gaussian approximation
    result += 0.5 * stats.norm.pdf(x, -amplitude, sigma)
    result += 0.5 * stats.norm.pdf(x, amplitude, sigma)
    return result

# Waveform theoretical CDFs
def sine_wave_cdf(x: np.ndarray, amplitude: float) -> np.ndarray:
    """Theoretical CDF for sine wave."""
    result = np.zeros_like(x, dtype=float)
    # Values below -amplitude have CDF = 0
    mask_below = x < -amplitude
    result[mask_below] = 0
    # Values above amplitude have CDF = 1
    mask_above = x > amplitude
    result[mask_above] = 1
    # Values in between follow the formula
    mask_between = ~(mask_below | mask_above)
    result[mask_between] = (1/np.pi) * np.arcsin(x[mask_between]/amplitude) + 0.5
    return result

def triangle_wave_cdf(x: np.ndarray, amplitude: float) -> np.ndarray:
    """Theoretical CDF for triangle wave."""
    result = np.zeros_like(x, dtype=float)
    # Values below -amplitude have CDF = 0
    mask_below = x < -amplitude
    result[mask_below] = 0
    # Values above amplitude have CDF = 1
    mask_above = x > amplitude
    result[mask_above] = 1
    # Values in between follow the formula (linear increase)
    mask_between = ~(mask_below | mask_above)
    result[mask_between] = (x[mask_between] + amplitude) / (2 * amplitude)
    return result

def square_wave_cdf(x: np.ndarray, amplitude: float) -> np.ndarray:
    """Theoretical CDF for square wave."""
    result = np.zeros_like(x, dtype=float)
    # Values below -amplitude have CDF = 0
    mask_below = x < -amplitude
    result[mask_below] = 0
    # Values above amplitude have CDF = 1
    mask_above = x > amplitude
    result[mask_above] = 1
    # Values between -amplitude and amplitude have CDF = 0.5
    mask_between = (x >= -amplitude) & (x < amplitude)
    result[mask_between] = 0.5
    return result

# Waveform types dictionary
WAVEFORM_TYPES = {
    'sine': {
        'function': sine_wave,
        'pdf': sine_wave_pdf,
        'cdf': sine_wave_cdf,
        'name': 'Sine Wave'
    },
    'triangle': {
        'function': triangle_wave,
        'pdf': triangle_wave_pdf,
        'cdf': triangle_wave_cdf,
        'name': 'Triangle Wave'
    },
    'square': {
        'function': square_wave,
        'pdf': square_wave_pdf,
        'cdf': square_wave_cdf,
        'name': 'Square Wave'
    }
}

# Default parameters
DEFAULT_PARAMS = {
    'amplitude': 141,        # Amplitude of the wave (V)
    'num_samples': 100000,   # Number of samples for the main analysis
    'num_bins': 50,          # Number of bins for the histogram
    'num_demo_samples': 20,  # Number of samples to show in the demonstration
    'output_path': '/workspace/waveform_pdf_and_cdf.png',  # Output image file path
    'csv_output': '/workspace/waveform_data.xlsx',  # Output Excel file path
    'waveform': 'sine',      # Type of waveform (sine, triangle, square)
    'random_seed': 42,       # Random seed for reproducibility
    'dpi': 300,              # DPI for the output image
    'export_csv': False      # Whether to export data to CSV
}

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Calculate and visualize PDF and CDF of various waveforms'
    )
    parser.add_argument('--amplitude', type=float, default=DEFAULT_PARAMS['amplitude'],
                        help=f'Amplitude of the waveform in volts (default: {DEFAULT_PARAMS["amplitude"]})')
    parser.add_argument('--num-samples', type=int, default=DEFAULT_PARAMS['num_samples'],
                        help=f'Number of samples for analysis (default: {DEFAULT_PARAMS["num_samples"]})')
    parser.add_argument('--num-bins', type=int, default=DEFAULT_PARAMS['num_bins'],
                        help=f'Number of bins for histogram (default: {DEFAULT_PARAMS["num_bins"]})')
    parser.add_argument('--num-demo-samples', type=int, default=DEFAULT_PARAMS['num_demo_samples'],
                        help=f'Number of demonstration samples (default: {DEFAULT_PARAMS["num_demo_samples"]})')
    parser.add_argument('--output', type=str, default=DEFAULT_PARAMS['output_path'],
                        help=f'Output image file path (default: {DEFAULT_PARAMS["output_path"]})')
    parser.add_argument('--excel', type=str, default=DEFAULT_PARAMS['csv_output'],
                        help=f'Output Excel file path (default: {DEFAULT_PARAMS["csv_output"]})')
    parser.add_argument('--export-excel', action='store_true', default=DEFAULT_PARAMS['export_csv'],
                        help='Export data to Excel file')
    parser.add_argument('--waveform', type=str, choices=list(WAVEFORM_TYPES.keys()), 
                        default=DEFAULT_PARAMS['waveform'],
                        help=f'Type of waveform (default: {DEFAULT_PARAMS["waveform"]})')
    parser.add_argument('--seed', type=int, default=DEFAULT_PARAMS['random_seed'],
                        help=f'Random seed (default: {DEFAULT_PARAMS["random_seed"]})')
    parser.add_argument('--dpi', type=int, default=DEFAULT_PARAMS['dpi'],
                        help=f'DPI for output image (default: {DEFAULT_PARAMS["dpi"]})')
    
    return parser.parse_args()

def generate_waveform_samples(
    waveform_function: Callable,
    amplitude: float, 
    num_samples: int, 
    random_seed: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate random samples from a waveform.
    
    Args:
        waveform_function: Function to generate waveform values
        amplitude: Amplitude of the waveform
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
    
    # Calculate voltage values at these random times
    voltage_values = waveform_function(random_times, amplitude)
    
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

def export_to_csv(
    file_path: str,
    voltage_values: np.ndarray,
    bin_centers: np.ndarray,
    hist: np.ndarray,
    sorted_bin_centers: np.ndarray,
    sorted_cdf: np.ndarray,
    waveform_type: str,
    amplitude: float
) -> None:
    """
    Export analysis results to Excel file.
    
    Args:
        file_path: Path to save the Excel file
        voltage_values: Array of voltage values
        bin_centers: Centers of histogram bins
        hist: Histogram values (PDF)
        sorted_bin_centers: Sorted centers of histogram bins
        sorted_cdf: Numerical CDF values
        waveform_type: Type of waveform
        amplitude: Amplitude of the waveform
    """
    # Create a DataFrame with the raw voltage samples
    samples_df = pd.DataFrame({
        'voltage': voltage_values
    })
    
    # Create a DataFrame with the PDF data
    pdf_df = pd.DataFrame({
        'bin_center': bin_centers,
        'pdf': hist
    })
    
    # Create a DataFrame with the CDF data
    cdf_df = pd.DataFrame({
        'voltage': sorted_bin_centers,
        'cdf': sorted_cdf
    })
    
    # Create a DataFrame with metadata
    metadata_df = pd.DataFrame({
        'parameter': ['waveform_type', 'amplitude', 'num_samples', 'num_bins'],
        'value': [waveform_type, amplitude, len(voltage_values), len(bin_centers)]
    })
    
    # Create a dictionary of DataFrames to save to different sheets
    dfs = {
        'metadata': metadata_df,
        'samples': samples_df,
        'pdf': pdf_df,
        'cdf': cdf_df
    }
    
    # Save to Excel file with multiple sheets
    with pd.ExcelWriter(file_path) as writer:
        for sheet_name, df in dfs.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            
    print(f"データを '{file_path}' に保存しました。")

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

def plot_waveform_with_samples(
    ax: plt.Axes,
    waveform_function: Callable,
    waveform_name: str,
    amplitude: float,
    times: np.ndarray,
    voltages: np.ndarray
) -> None:
    """
    Plot waveform with random samples.
    
    Args:
        ax: Matplotlib axes to plot on
        waveform_function: Function to generate waveform values
        waveform_name: Name of the waveform for the plot title
        amplitude: Amplitude of the waveform
        times: Array of random time points
        voltages: Array of corresponding voltage values
    """
    # Generate waveform for one period
    t = np.linspace(0, 1, 1000)
    v = waveform_function(t, amplitude)
    
    # Plot waveform
    ax.plot(t, v, 'b-', linewidth=2, label=waveform_name)
    
    # Plot random samples
    ax.scatter(times, voltages, color='red', s=50, label=f'Random Samples (n={len(times)})')
    
    # Add vertical lines to show sampling
    for i in range(len(times)):
        ax.plot([times[i], times[i]], [0, voltages[i]], 'r--', alpha=0.3)
    
    # Set labels and limits
    ax.set_title(f'{waveform_name} with Random Time Sampling')
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
    pdf_function: Callable,
    amplitude: float,
    waveform_name: str
) -> None:
    """
    Plot comparison of numerical and theoretical PDFs.
    
    Args:
        ax: Matplotlib axes to plot on
        bin_centers: Centers of histogram bins
        hist: Histogram values
        bin_width: Width of histogram bins
        x: Array of voltage values for theoretical PDF
        pdf_function: Function to calculate theoretical PDF
        amplitude: Amplitude of the waveform
        waveform_name: Name of the waveform for the plot title
    """
    # Calculate theoretical PDF
    pdf = pdf_function(x, amplitude)
    
    # Plot numerical PDF (histogram)
    ax.bar(bin_centers, hist, width=bin_width, alpha=0.6, 
           label='Numerical PDF (Histogram)', color='green')
    
    # Plot theoretical PDF
    ax.plot(x, pdf, 'r-', linewidth=2, label=f'Theoretical PDF')
    
    ax.set_title(f'Probability Density Function Comparison - {waveform_name}')
    ax.set_xlabel('Voltage (V)')
    ax.set_ylabel('Probability Density')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(-amplitude*1.05, amplitude*1.05)
    
    # Set y-limit based on the maximum PDF value, with some padding
    max_pdf = max(np.max(pdf), np.max(hist)) if len(pdf) > 0 and len(hist) > 0 else 1
    ax.set_ylim(0, max_pdf*1.1)

def plot_cdf_comparison(
    ax: plt.Axes,
    sorted_bin_centers: np.ndarray,
    sorted_cdf: np.ndarray,
    x_sorted: np.ndarray,
    cdf_function: Callable,
    amplitude: float,
    waveform_name: str
) -> None:
    """
    Plot comparison of numerical and theoretical CDFs.
    
    Args:
        ax: Matplotlib axes to plot on
        sorted_bin_centers: Sorted centers of histogram bins
        sorted_cdf: Numerical CDF values
        x_sorted: Sorted array of voltage values for theoretical CDF
        cdf_function: Function to calculate theoretical CDF
        amplitude: Amplitude of the waveform
        waveform_name: Name of the waveform for the plot title
    """
    # Calculate theoretical CDF
    theoretical_cdf = cdf_function(x_sorted, amplitude)
    
    # Plot numerical CDF
    ax.plot(sorted_bin_centers, sorted_cdf, 'g-', linewidth=2, label='Numerical CDF')
    
    # Plot theoretical CDF
    ax.plot(x_sorted, theoretical_cdf, 'r--', linewidth=2, label='Theoretical CDF')
    
    ax.set_title(f'Cumulative Distribution Function (CDF) - {waveform_name}')
    ax.set_xlabel('Voltage (V)')
    ax.set_ylabel('Cumulative Probability')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(-amplitude*1.05, amplitude*1.05)
    ax.set_ylim(0, 1.05)

def create_plots(params: Dict[str, Any]) -> Tuple[plt.Figure, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create all plots for waveform PDF and CDF analysis.
    
    Args:
        params: Dictionary of parameters
        
    Returns:
        Tuple containing:
            - Matplotlib figure containing all plots
            - Array of voltage values
            - Bin width
            - Bin centers
            - Histogram values
            - Sorted bin centers
            - Sorted CDF values
    """
    # Extract parameters
    amplitude = params['amplitude']
    num_samples = params['num_samples']
    num_bins = params['num_bins']
    num_demo_samples = params['num_demo_samples']
    random_seed = params['random_seed']
    waveform_type = params['waveform']
    
    # Get waveform functions
    waveform_info = WAVEFORM_TYPES[waveform_type]
    waveform_function = waveform_info['function']
    pdf_function = waveform_info['pdf']
    cdf_function = waveform_info['cdf']
    waveform_name = waveform_info['name']
    
    # Generate samples
    random_times, voltage_values = generate_waveform_samples(
        waveform_function, amplitude, num_samples, random_seed)
    demo_times, demo_voltages = generate_waveform_samples(
        waveform_function, amplitude, num_demo_samples, random_seed + 1)  # Use different seed for demo
    
    # Calculate histogram for main samples
    hist, bin_edges, bin_centers, bin_width = calculate_histogram(
        voltage_values, num_bins)
    
    # Calculate histogram for demo samples
    demo_hist, demo_bin_edges, demo_bin_centers, demo_bin_width = calculate_histogram(
        demo_voltages, 10)  # Fewer bins for demo
    
    # Calculate numerical CDF
    sorted_bin_centers, sorted_cdf = calculate_numerical_cdf(
        hist, bin_centers, bin_width)
    
    # Create x values for theoretical functions
    x = np.linspace(-amplitude+0.001, amplitude-0.001, 1000)  # Avoid division by zero
    x_sorted = np.sort(x)
    
    # Create figure with 5 subplots
    fig = plt.figure(figsize=(14, 16))
    
    # Plot 1: Waveform with samples
    ax1 = fig.add_subplot(5, 1, 1)
    plot_waveform_with_samples(ax1, waveform_function, waveform_name, 
                              amplitude, demo_times, demo_voltages)
    
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
    plot_pdf_comparison(ax4, bin_centers, hist, bin_width, x, 
                       pdf_function, amplitude, waveform_name)
    
    # Plot 5: CDF comparison
    ax5 = fig.add_subplot(5, 1, 5)
    plot_cdf_comparison(
        ax5, sorted_bin_centers, sorted_cdf, x_sorted, 
        cdf_function, amplitude, waveform_name)
    
    plt.tight_layout()
    
    return fig, voltage_values, bin_width, bin_centers, hist, sorted_bin_centers, sorted_cdf

def print_statistics_and_explanation(
    voltage_values: np.ndarray, 
    bin_width: float, 
    amplitude: float,
    waveform_type: str
) -> None:
    """
    Print statistical analysis and explanation of the results.
    
    Args:
        voltage_values: Array of voltage values
        bin_width: Width of histogram bins
        amplitude: Amplitude of the waveform
        waveform_type: Type of waveform
    """
    waveform_name = WAVEFORM_TYPES[waveform_type]['name']
    
    print("Statistical Analysis:")
    print(f"Waveform type: {waveform_name}")
    print(f"Min voltage: {voltage_values.min():.2f} V")
    print(f"Max voltage: {voltage_values.max():.2f} V")
    print(f"Mean voltage: {voltage_values.mean():.2f} V")
    print(f"Standard deviation: {voltage_values.std():.2f} V")
    print(f"Bin width: {bin_width:.2f} V")

    print(f"\nExplanation of the Probability Density Function and CDF of a {waveform_name}:")
    
    if waveform_type == 'sine':
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
        
        print("\n3. Practical Implications:")
        print("   - In AC circuits, the voltage spends more time near peak values than near zero")
        print("   - This affects measurements, power calculations, and component stress")
        print("   - The RMS (root mean square) value of a sine wave is A/√2 ≈ 0.707A")
        print(f"   - For our amplitude of {amplitude}V, the RMS value is approximately {amplitude/np.sqrt(2):.1f}V")
        
    elif waveform_type == 'triangle':
        print("1. Mathematical Background:")
        print("   - For a triangle wave, the PDF is uniform across its range")
        print("   - The PDF formula is: f(V) = 1/(2A) for -A ≤ V ≤ A")
        print("   - The CDF formula is: F(V) = (V+A)/(2A) for -A ≤ V ≤ A")
        print("   - This is because the triangle wave changes at a constant rate")

        print("\n2. Key Characteristics:")
        print("   - The PDF is flat (uniform distribution) across the entire voltage range")
        print("   - All voltage values are equally likely to occur")
        print("   - The mean value is 0V (for a complete cycle)")
        print("   - The standard deviation is A/√3 ≈ 0.577A")
        print("   - The CDF is a straight line from (−A,0) to (A,1)")
        
        print("\n3. Practical Implications:")
        print("   - In circuits with triangle waves, the voltage is distributed evenly")
        print("   - The RMS value of a triangle wave is A/√3 ≈ 0.577A")
        print(f"   - For our amplitude of {amplitude}V, the RMS value is approximately {amplitude/np.sqrt(3):.1f}V")
        
    elif waveform_type == 'square':
        print("1. Mathematical Background:")
        print("   - For a square wave, the PDF consists of two delta functions at ±A")
        print("   - The PDF formula is: f(V) = 0.5·δ(V+A) + 0.5·δ(V-A)")
        print("   - The CDF formula is: F(V) = 0 for V < -A, 0.5 for -A ≤ V < A, and 1 for V ≥ A")
        print("   - This is because the square wave only takes on values of ±A")

        print("\n2. Key Characteristics:")
        print("   - The PDF has non-zero values only at V = ±A")
        print("   - The probability is equally divided between the two possible values")
        print("   - The mean value is 0V (for a complete cycle)")
        print("   - The standard deviation is exactly A")
        print("   - The CDF is a step function with a jump of 0.5 at V = -A and V = A")
        
        print("\n3. Practical Implications:")
        print("   - In circuits with square waves, the voltage is always at its extreme values")
        print("   - The RMS value of a square wave is exactly A")
        print(f"   - For our amplitude of {amplitude}V, the RMS value is exactly {amplitude:.1f}V")
    
    print("\nSampling Process:")
    print("   - We randomly selected time points from a uniform distribution over one period")
    print("   - At each time point, we calculated the corresponding voltage")
    print("   - The histogram of these voltage values approximates the theoretical PDF")
    print("   - With more samples, the histogram converges to the theoretical distribution")
    
    print("\nCDF Interpretation:")
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
            'csv_output': args.excel,
            'export_csv': args.export_excel,
            'waveform': args.waveform,
            'random_seed': args.seed,
            'dpi': args.dpi
        }
        
        # Create plots
        fig, voltage_values, bin_width, bin_centers, hist, sorted_bin_centers, sorted_cdf = create_plots(params)
        
        # Save figure
        output_dir = os.path.dirname(params['output_path'])
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        fig.savefig(params['output_path'], dpi=params['dpi'], bbox_inches='tight')
        print(f"グラフを '{params['output_path']}' に保存しました。")
        
        # Export data to CSV if requested
        if params['export_csv']:
            export_to_csv(
                params['csv_output'],
                voltage_values,
                bin_centers,
                hist,
                sorted_bin_centers,
                sorted_cdf,
                params['waveform'],
                params['amplitude']
            )
        
        # Print statistics and explanation
        print_statistics_and_explanation(
            voltage_values, 
            bin_width, 
            params['amplitude'],
            params['waveform']
        )
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)