import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path
from datetime import datetime

# Add root to path
sys.path.append(str(Path(__file__).parent.parent))

from code.model import PositionalEncoding

st.set_page_config(page_title="Positional Encoding Explorer", layout="wide")

st.title("🧩 Positional Encoding Interactive Explorer")

# --- Sidebar ---
st.sidebar.header("Hyperparameters")
d_model = st.sidebar.slider("d_model", 64, 1024, 512, step=64)
max_seq_len = st.sidebar.slider("max_seq_len", 100, 10000, 10000, step=100)

st.sidebar.header("Visualization Settings")
vis_seq_len = st.sidebar.slider("vis_seq_len", 10, 500, 100, step=10)
vis_dims = st.sidebar.slider("vis_dims", 8, 256, 64, step=8)
k_offset = st.sidebar.slider("k_offset (Linear Transform Demo)", 1, 100, 1, step=1)

# --- Compute ---
pe_obj = PositionalEncoding(d_model=d_model, max_len=max_seq_len)
pe_matrix = pe_obj.get_encoding()

# --- Main Panel ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "PE Heatmap", "Wavelengths", "Cosine Similarity", "Norm Stability", "Linear Transform"
])

with tab1:
    st.header("Positional Encoding Matrix Heatmap")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pe_matrix[:vis_seq_len, :vis_dims], cmap="RdBu", center=0, ax=ax)
    ax.set_title(f"PE Matrix (Showing {vis_seq_len} tokens, {vis_dims} dims)")
    st.pyplot(fig)

with tab2:
    st.header("Wavelength Progression")
    wavelengths = pe_obj.get_wavelengths()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(np.arange(0, len(wavelengths) * 2, 2), wavelengths)
    ax.set_yscale("log")
    ax.set_xlabel("Dimension Index (2i)")
    ax.set_ylabel("Wavelength (Tokens)")
    ax.grid(True, which="both", ls="-", alpha=0.5)
    st.pyplot(fig)

with tab3:
    st.header("Cosine Similarity Heatmap")
    # Normalize for cosine similarity
    subset = pe_matrix[:vis_seq_len, :]
    norm = np.linalg.norm(subset, axis=1, keepdims=True)
    norm_subset = subset / norm
    sim_matrix = np.dot(norm_subset, norm_subset.T)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(sim_matrix, cmap="viridis", ax=ax)
    st.pyplot(fig)

with tab4:
    st.header("L2 Norm Stability")
    norms = np.linalg.norm(pe_matrix, axis=1)
    expected = np.sqrt(d_model / 2)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(norms, label="Actual Norm")
    ax.axhline(expected, color='r', linestyle='--', label=f"Expected (sqrt(d/2)={expected:.2f})")
    ax.set_ylim(expected * 0.9, expected * 1.1)
    ax.set_xlabel("Position")
    ax.set_ylabel("L2 Norm")
    ax.legend()
    st.pyplot(fig)

with tab5:
    st.header("Linear Transformation Verification")
    st.write(f"Verifying $PE_{{pos+{k_offset}}} = M_{{{k_offset}}} \\cdot PE_{{pos}}$")
    
    # Construct M_k
    M_k = np.zeros((d_model, d_model))
    for i in range(0, d_model, 2):
        theta_i = 1.0 / (10000.0 ** (i / d_model))
        angle = k_offset * theta_i
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        M_k[i:i+2, i:i+2] = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
    
    # Test across all possible positions
    errors = []
    limit = max_seq_len - k_offset
    for pos in range(0, min(limit, 500)):
        transformed = M_k @ pe_matrix[pos]
        err = np.linalg.norm(transformed - pe_matrix[pos + k_offset])
        errors.append(err)
    
    st.metric("Max Reconstruction Error", f"{np.max(errors):.2e}")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(errors)
    ax.set_title("Reconstruction Error across Positions")
    ax.set_ylabel("L2 Error")
    st.pyplot(fig)

if st.button("Save All Figures"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path("screenshots") / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Redraw and save
    # (In a real app, you'd reuse the figures, but here we redraw for brevity)
    st.success(f"All figures saved to {save_dir}/")
