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

def main():
    # Set random seed for reproducibility
    np.random.seed(42)

    # Parameters
    amplitude = 141  # Amplitude of the sine wave (V)
    num_samples = 100000  # Number of samples for the main analysis
    num_bins = 50  # Number of bins for the histogram
    num_demo_samples = 20  # Number of samples to show in the demonstration

    # Generate random time points uniformly distributed between 0 and 1
    random_times = np.random.uniform(0, 1, num_samples)
    demo_times = np.random.uniform(0, 1, num_demo_samples)  # For demonstration

    # Calculate voltage values at these random times (sine wave with period 1)
    voltage_values = amplitude * np.sin(2 * np.pi * random_times)
    demo_voltages = amplitude * np.sin(2 * np.pi * demo_times)

    # Create histogram
    hist, bin_edges = np.histogram(voltage_values, bins=num_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    # Theoretical PDF for comparison
    x = np.linspace(-amplitude+0.001, amplitude-0.001, 1000)  # Avoid division by zero at endpoints
    pdf = 1 / (np.pi * np.sqrt(amplitude**2 - x**2))

    # Create a figure with 5 subplots (added CDF)
    fig = plt.figure(figsize=(14, 16))

    # Plot 1: The sine wave for one period with sampling demonstration
    ax1 = fig.add_subplot(5, 1, 1)
    t = np.linspace(0, 1, 1000)
    v = amplitude * np.sin(2 * np.pi * t)
    ax1.plot(t, v, 'b-', linewidth=2, label='Sine Wave')
    ax1.scatter(demo_times, demo_voltages, color='red', s=50, label=f'Random Samples (n={num_demo_samples})')

    # Add vertical lines to show sampling
    for i in range(num_demo_samples):
        ax1.plot([demo_times[i], demo_times[i]], [0, demo_voltages[i]], 'r--', alpha=0.3)

    ax1.set_title('Sinusoidal AC Signal with Random Time Sampling')
    ax1.set_xlabel('Time (normalized)')
    ax1.set_ylabel('Voltage (V)')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(-amplitude*1.1, amplitude*1.1)
    ax1.legend()

    # Plot 2: Histogram of the demonstration samples
    ax2 = fig.add_subplot(5, 1, 2)
    demo_hist, demo_bins = np.histogram(demo_voltages, bins=10, density=True)
    demo_bin_centers = (demo_bins[:-1] + demo_bins[1:]) / 2
    demo_bin_width = demo_bins[1] - demo_bins[0]

    ax2.bar(demo_bin_centers, demo_hist, width=demo_bin_width, alpha=0.6, color='orange')
    ax2.set_title(f'Histogram of Demonstration Samples (n={num_demo_samples})')
    ax2.set_xlabel('Voltage (V)')
    ax2.set_ylabel('Probability Density')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-amplitude*1.05, amplitude*1.05)

    # Plot 3: Histogram of all sampled voltage values
    ax3 = fig.add_subplot(5, 1, 3)
    ax3.bar(bin_centers, hist, width=bin_width, alpha=0.6, color='green')
    ax3.set_title(f'Histogram of Voltage Values (n={num_samples}, bins={num_bins})')
    ax3.set_xlabel('Voltage (V)')
    ax3.set_ylabel('Probability Density')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-amplitude*1.05, amplitude*1.05)

    # Plot 4: Comparison of numerical and theoretical PDFs
    ax4 = fig.add_subplot(5, 1, 4)
    
    # Calculate the CDF from the histogram data
    cdf_values = np.cumsum(hist) * bin_width
    ax4.bar(bin_centers, hist, width=bin_width, alpha=0.6, label='Numerical PDF (Histogram)', color='green')
    ax4.plot(x, pdf, 'r-', linewidth=2, label='Theoretical PDF: 1/(π√(A²-V²))')
    ax4.set_title('Probability Density Function Comparison')
    ax4.set_xlabel('Voltage (V)')
    ax4.set_ylabel('Probability Density')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_xlim(-amplitude*1.05, amplitude*1.05)
    ax4.set_ylim(0, max(pdf)*1.1)
    
    # Plot 5: Cumulative Distribution Function (CDF)
    ax5 = fig.add_subplot(5, 1, 5)
    
    # Numerical CDF from histogram
    sorted_indices = np.argsort(bin_centers)
    sorted_bin_centers = bin_centers[sorted_indices]
    sorted_cdf = np.cumsum(hist[sorted_indices]) * bin_width
    # Normalize CDF to ensure it ends at 1.0
    sorted_cdf = sorted_cdf / sorted_cdf[-1]
    
    # Theoretical CDF for a sine wave: F(V) = (1/π) * arcsin(V/A) + 0.5 for -A ≤ V ≤ A
    x_sorted = np.sort(x)
    theoretical_cdf = (1/np.pi) * np.arcsin(x_sorted/amplitude) + 0.5
    
    ax5.plot(sorted_bin_centers, sorted_cdf, 'g-', linewidth=2, label='Numerical CDF')
    ax5.plot(x_sorted, theoretical_cdf, 'r--', linewidth=2, label='Theoretical CDF: (1/π)arcsin(V/A) + 0.5')
    ax5.set_title('Cumulative Distribution Function (CDF)')
    ax5.set_xlabel('Voltage (V)')
    ax5.set_ylabel('Cumulative Probability')
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    ax5.set_xlim(-amplitude*1.05, amplitude*1.05)
    ax5.set_ylim(0, 1.05)

    plt.tight_layout()
    
    # Save the figure as an image file instead of displaying it
    plt.savefig('/workspace/sine_wave_pdf_and_cdf.png', dpi=300, bbox_inches='tight')
    print("グラフを '/workspace/sine_wave_pdf_and_cdf.png' に保存しました。")

    # Print statistics and explanation
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
    print("   - For our amplitude of 141V, the RMS value is approximately 99.7V")
    print("   - The CDF is useful for determining the probability that the voltage")
    print("     will be less than or equal to a specific value")
    print("   - For example, the probability that the voltage is less than or equal to 0V is 0.5")

if __name__ == "__main__":
    main()