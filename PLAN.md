# Implementation Plan: Positional Encoding Analysis

## Phase 1: Mathematical Engine
1.  **Step 1: PE Matrix Construction**
    *   Compute $PE[pos, 2i] = \sin(pos/10000^{2i/d\_model})$
    *   Compute $PE[pos, 2i+1] = \cos(pos/10000^{2i/d\_model})$
    *   Implement using vectorized NumPy broadasting for shape $[max\_seq\_len, d\_model]$.

## Phase 2: Research Question Validation
2.  **Step 2: RQ 1 - Uniqueness**
    *   Generate a heatmap of $PE[:100, :64]$ using Seaborn.
    *   Perform a pairwise equality check using `np.allclose` to ensure no two rows are identical.
3.  **Step 3: RQ 2 - Multi-Scale**
    *   Compute analytical wavelengths $\lambda_i = 2\pi \cdot 10000^{2i/d\_model}$.
    *   Plot $\lambda_i$ vs $i$ on a log scale; verify the range from $\approx 6.28$ to $\approx 62,831$.
4.  **Step 4: RQ 3 - Linear Transform**
    *   Implement a function to construct $M_k$ as a block-diagonal matrix of $2 \times 2$ rotation matrices $R(k \cdot \theta_i)$.
    *   For $k \in \{1, 5, 10, 50\}$, verify $||PE_{pos+k} - M_k \cdot PE_{pos}||_F < 1e-10$.
5.  **Step 5: RQ 4 - Extrapolation**
    *   Calculate $PE$ for $pos$ up to $50,000$.
    *   Plot the cosine similarity between $PE[0]$ and $PE[pos]$. Verify the signal remains stable and non-divergent.
6.  **Step 6: RQ 5 - Norm Stability**
    *   Calculate $||PE_{pos}||_2 = \sqrt{\sum PE_{pos, d}^2}$ for all positions.
    *   Plot the results; expect a constant line at $\sqrt{d\_model/2}$.
7.  **Step 7: RQ 6 - Sine vs Cosine**
    *   Select a frequency pair $(sin, cos)$ and plot their values across positions.
    *   Verify the trajectory forms a perfect circle (phase quadrature).

## Phase 3: Reporting & Visualization
8.  **Step 8: Export Visual Assets**
    *   Save all generated figures to `docs/assets/` using high DPI. Ensure no `plt.show()` calls exist.
9.  **Step 9: Numerical Summary**
    *   Generate a `results.json` in `output/analysis/` containing the exact error metrics, norm variances, and uniqueness flags.
