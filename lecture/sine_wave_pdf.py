#!/usr/bin/env python3
"""
Simplified Sine Wave PDF and CDF Analysis

This script generates random samples from a sine wave,
calculates the Probability Density Function (PDF) and
Cumulative Distribution Function (CDF), and saves the plots separately.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Sine Wave PDF and CDF Analysis')
    parser.add_argument('--output', type=str, default='./sine_wave_pdf_and_cdf.png', help='Base output image path')
    parser.add_argument('--num-bins', type=int, default=1000, help='Number of bins for histogram')
    return parser.parse_args()

def generate_samples(num_samples=100000, amplitude=141, seed=42):
    """Generate random voltage samples from a sine wave."""
    np.random.seed(seed)
    t = np.random.uniform(0, 1, num_samples)
    voltage = amplitude * np.sin(2 * np.pi * t)
    return voltage

def plot_histogram(voltage, num_bins, amplitude, output_path, dpi=300):
    """Plot and save the histogram (PDF) of voltage values."""
    plt.figure(figsize=(10, 6))
    plt.hist(voltage, bins=num_bins, density=True, alpha=0.6, color='g', label='Histogram (PDF)')

    # Theoretical PDF
    x = np.linspace(-amplitude, amplitude, 1000, endpoint=False)
    with np.errstate(divide='ignore', invalid='ignore'):
        theoretical_pdf = 1 / (np.pi * np.sqrt(amplitude**2 - x**2))
        theoretical_pdf[np.isnan(theoretical_pdf)] = 0  # Replace NaNs resulting from division by zero
    plt.plot(x, theoretical_pdf, 'r-', label='Theoretical PDF')

    plt.xlabel('Voltage (V)')
    plt.ylabel('Probability Density')
    plt.title('Sine Wave Probability Density Function (PDF)')
    plt.legend()
    plt.grid(True)
    plt.xlim(-amplitude, amplitude)
    plt.savefig(output_path.replace('.png', '_pdf.png'), dpi=dpi)
    plt.close()

def plot_cdf(voltage, output_path, dpi=300):
    """Plot and save the Cumulative Distribution Function (CDF) of voltage values."""
    plt.figure(figsize=(10, 6))

    # Empirical CDF using interpolation
    sorted_voltage = np.sort(voltage)
    cdf = np.arange(1, len(sorted_voltage)+1) / len(sorted_voltage)
    
    # Define x from -150 to 150
    x = np.linspace(-150, 150, 1000)
    
    # Compute empirical CDF at these x values using interpolation
    empirical_cdf = np.interp(x, sorted_voltage, cdf, left=0, right=1)
    
    plt.plot(x, empirical_cdf, 'b-', label='Empirical CDF')

    # Theoretical CDF
    theoretical_cdf = np.piecewise(
        x,
        [x < -141, x > 141, (x >= -141) & (x <= 141)],
        [0, 1, lambda v: (1/np.pi) * np.arcsin(v/141) + 0.5]
    )
    plt.plot(x, theoretical_cdf, 'r--', label='Theoretical CDF')

    plt.xlabel('Voltage (V)')
    plt.ylabel('Cumulative Probability')
    plt.title('Sine Wave Cumulative Distribution Function (CDF)')
    plt.legend()
    plt.grid(True)
    plt.xlim(-150, 150)
    plt.ylim(0, 1)
    plt.savefig(output_path.replace('.png', '_cdf.png'), dpi=dpi)
    plt.close()

def main():
    """Main function to run the analysis."""
    args = parse_args()
    voltage = generate_samples()
    plot_histogram(voltage, args.num_bins, 141, args.output)
    plot_cdf(voltage, args.output)
    print(f"Plots saved to {args.output.replace('.png', '_pdf.png')} and {args.output.replace('.png', '_cdf.png')}")

if __name__ == "__main__":
    main()
