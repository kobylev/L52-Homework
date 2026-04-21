# Product Requirements Document (PRD): Positional Encoding Analysis

## 1. Overview
This project provides a comprehensive geometric and analytical study of the sinusoidal Positional Encoding (PE) mechanism introduced in "Attention Is All You Need" (Vaswani et al., 2017). It aims to prove six fundamental mathematical properties of PE through rigorous visualization and linear algebraic verification, serving as both an educational resource and a technical validation suite.

## 2. Objectives
1.  **Prove Linear Relationship**: Verify that $PE_{pos+k}$ can be represented as a linear transformation $M_k \cdot PE_{pos}$ through matrix construction and numerical error quantification.
2.  **Demonstrate Multi-Scale Frequencies**: Visualize the exponential distribution of wavelengths across the $d_{model}$ dimensions.
3.  **Validate Relative Distance Perceivability**: Show that cosine similarity between position vectors decays monotonically as the distance between positions increases.
4.  **Confirm Signal Stability**: Verify that the Euclidean norm of position vectors remains constant across the entire sequence.
5.  **Quantify Extrapolation Limits**: Analyze the behavior of the encoding function when sequence lengths exceed the training bounds ($max\_seq\_len$).

## 3. Scope
*   **In-Scope**:
    *   Sinusoidal Positional Encoding (Fixed).
    *   Geometric interpretation of PE subspaces.
    *   Numerical verification of rotation properties.
    *   Visual analysis of frequency spectra.
*   **Out-of-Scope**:
    *   Learned Positional Embeddings.
    *   Rotary Positional Embeddings (RoPE).
    *   ALiBi or other relative bias mechanisms.
    *   Full Transformer model training.

## 4. Functional Requirements
1.  Generate a full PE matrix of shape $[max\_seq\_len, d\_model]$.
2.  Visualize the PE matrix as a high-resolution heatmap.
3.  Plot wavelength $\lambda_i$ versus dimension index $i$ on a logarithmic scale.
4.  Construct and verify the $M_k$ transformation matrix for arbitrary offsets $k$.
5.  Generate a Cosine Similarity heatmap for a configurable sub-window of positions.
6.  Compute and plot the $L_2$ norm of position vectors for all positions.
7.  Analyze 2D subspaces $(sin, cos)$ to demonstrate unit-circle properties.
8.  Export all analytical results to a structured JSON summary.
9.  Provide an interactive dashboard for parameter exploration.

## 5. Non-Functional Requirements
1.  **Reproducibility**: All calculations must use a fixed seed (default=42).
2.  **Modularity**: Decouple logic into configuration, modeling, and visualization modules.
3.  **Efficiency**: Utilize vectorized NumPy operations for all matrix calculations.
4.  **Stability**: No interactive GUI windows in the core analysis (`plt.show()` is forbidden; use `savefig`).

## 6. Success Criteria
*   **Order Invariance**: Verification that all rows in the PE matrix are unique.
*   **Multi-Scale**: $R^2 > 0.999$ for the logarithmic wavelength progression.
*   **Relative Distance**: Frobenius norm error for linear transformation $< 1e-10$.
*   **Stability**: Standard deviation of vector norms $< 1\%$ of the mean.

## 7. Constraints
*   **Technology Stack**: Python 3.10+, NumPy, Matplotlib, Seaborn.
*   **Dependencies**: No heavy Deep Learning frameworks (PyTorch/TensorFlow) required for core analysis.
