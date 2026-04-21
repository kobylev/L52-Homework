import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from config import ASSETS_DIR, TARGET_POS

def save_plot(filename):
    if not os.path.exists(ASSETS_DIR):
        os.makedirs(ASSETS_DIR)
    path = os.path.join(ASSETS_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

def plot_pe_matrix(pe_matrix):
    plt.figure(figsize=(12, 8))
    sns.heatmap(pe_matrix, cmap='RdBu', center=0)
    plt.title("Positional Encoding Matrix")
    plt.xlabel("Embedding Dimension (d_model)")
    plt.ylabel("Position in Sequence")
    save_plot("pe_heatmap.png")

def plot_wavelengths(wavelengths):
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(0, len(wavelengths) * 2, 2), wavelengths)
    plt.yscale('log')
    plt.title("Wavelength Progression (Log Scale)")
    plt.xlabel("Dimension Index (2i)")
    plt.ylabel("Wavelength (Tokens)")
    plt.grid(True, which="both", ls="-", alpha=0.5)
    save_plot("wavelength_progression.png")

def plot_position_vector(pe_matrix, pos=TARGET_POS):
    plt.figure(figsize=(12, 4))
    plt.plot(pe_matrix[pos, :], label=f'Pos {pos}')
    plt.title(f"Positional Encoding Vector for Position {pos}")
    plt.xlabel("Dimension Index")
    plt.ylabel("Encoding Value")
    plt.legend()
    plt.grid(alpha=0.3)
    save_plot(f"pos_{pos}_vector.png")

def plot_similarity_heatmap(similarity_matrix):
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, cmap='viridis')
    plt.title("Cosine Similarity Matrix between Positions")
    plt.xlabel("Position Index")
    plt.ylabel("Position Index")
    save_plot("similarity_heatmap.png")

def plot_unit_circle(pe_matrix, dim_idx=0):
    """
    Plots the (sin, cos) pairs for a specific dimension to show the unit circle.
    """
    plt.figure(figsize=(6, 6))
    sin_vals = pe_matrix[:, 2*dim_idx]
    cos_vals = pe_matrix[:, 2*dim_idx + 1]
    plt.scatter(sin_vals, cos_vals, alpha=0.5, s=10)
    plt.title(f"2D Subspace (sin, cos) for Dimension Pair {dim_idx}")
    plt.xlabel("Sine Value")
    plt.ylabel("Cosine Value")
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    save_plot("unit_circle.png")

def plot_norm_stability(pe_matrix):
    """
    Plots the L2 norm of the PE vector across all positions.
    """
    norms = np.linalg.norm(pe_matrix, axis=1)
    d_model = pe_matrix.shape[1]
    expected_norm = np.sqrt(d_model / 2)
    
    plt.figure(figsize=(10, 6))
    plt.plot(norms, label="L2 Norm")
    plt.axhline(y=expected_norm, color='r', linestyle='--', label=f"Expected (sqrt(d/2)={expected_norm:.2f})")
    plt.title("L2 Norm Stability across Positions")
    plt.xlabel("Position Index")
    plt.ylabel("L2 Norm")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(expected_norm * 0.9, expected_norm * 1.1)
    save_plot("norm_stability.png")
