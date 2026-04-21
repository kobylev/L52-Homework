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
    save_plot("pe_matrix.png")

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
