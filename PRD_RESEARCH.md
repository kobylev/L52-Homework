# PRD Research: Mathematical Foundations of Positional Encoding

This document details the six core research questions that form the analytical backbone of this project.

---

### 1. Order Invariance
*   **Question**: Does the Positional Encoding uniquely identify each position in a sequence?
*   **Hypothesis**: Every row in the $PE$ matrix is unique, ensuring the model can distinguish any position $i$ from any position $j$.
*   **Experiment**: Calculate the minimum pairwise Euclidean distance between all rows in a sequence of length 10,000.
*   **Success Criterion**: $min(dist(PE_i, PE_j)) > 0$ for all $i \neq j$.
*   **Expected Figure**: A high-contrast heatmap showing the unique patterns across positions.

### 2. Multi-Scale Representation
*   **Question**: How do varying frequencies capture different levels of context?
*   **Hypothesis**: Dimensions with lower indices encode high-frequency (local) information, while higher indices encode low-frequency (global) signals, following an exponential growth in wavelength.
*   **Experiment**: Plot the wavelength $\lambda_i = 2\pi \cdot 10000^{2i/d_{model}}$ against the dimension index $i$.
*   **Success Criterion**: The wavelength should scale from $2\pi$ to $2\pi \cdot 10000$ across the embedding space.
*   **Expected Figure**: A line plot of wavelengths on a logarithmic Y-axis.

### 3. Linear Transformation (Relative Distances)
*   **Question**: Can the model perceive relative distances as linear transformations?
*   **Hypothesis**: For any offset $k$, there exists a fixed matrix $M_k$ such that $PE_{pos+k} = M_k \cdot PE_{pos}$.
*   **Experiment**: Construct a block-diagonal rotation matrix $M_k$ and measure the reconstruction error against the actual $PE_{pos+k}$.
*   **Success Criterion**: Mean Squared Error (MSE) between $PE_{pos+k}$ and $M_k PE_{pos}$ is $< 1e-10$.
*   **Expected Figure**: A verification table or plot showing error magnitude across various $k$ values.

### 4. Extrapolation
*   **Question**: How does the encoding behave for sequences longer than the training maximum?
*   **Hypothesis**: Because the functions are periodic, the model will perceive similarity between extremely distant tokens, but the signals remain bounded and deterministic.
*   **Experiment**: Compute $PE$ for $pos > 10,000$ and visualize the cosine similarity to position 0.
*   **Success Criterion**: Smooth, periodic decay without numerical instability (NaN/Inf).
*   **Expected Figure**: Cosine similarity curve extending significantly beyond the default sequence length.

### 5. Vector Norm Stability
*   **Question**: Does adding Positional Encoding distort the magnitude of token embeddings?
*   **Hypothesis**: The $L_2$ norm of the $PE$ vector is constant across all positions, ensuring consistent signal strength.
*   **Experiment**: Compute $||PE_{pos}||_2$ for every $pos \in [0, 10000]$.
*   **Success Criterion**: $||PE_{pos}||_2 \approx \sqrt{d_{model}/2}$ with negligible variance.
*   **Expected Figure**: A flat line plot of the norm versus sequence position.

### 6. Sine vs. Cosine Structural Choice
*   **Question**: Why are both sine and cosine functions used in the architecture?
*   **Hypothesis**: The sine-cosine pairs form 2D rotation basis vectors, which are essential for the linear transformation property $M_k$.
*   **Experiment**: Plot $(sin(pos \cdot \theta_i), cos(pos \cdot \theta_i))$ for a fixed $i$ across all $pos$.
*   **Success Criterion**: The points must lie exactly on a unit circle in the 2D subspace.
*   **Expected Figure**: A scatter plot showing a perfect circular trajectory for position vectors in 2D space.
