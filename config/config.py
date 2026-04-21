from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    d_model: int = 512          # PE embedding dimension
    max_seq_len: int = 10000    # Maximum sequence length
    n_freq_pairs: int = 256     # Number of sine/cosine frequency pairs (d_model // 2)
    vis_seq_len: int = 100      # Sequence length used in visualizations
    vis_dims: int = 64          # Number of dimensions shown in heatmap
    output_dir: str = "output/analysis"
    plots_dir: str = "docs/assets"
    seed: int = 42

    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.plots_dir).mkdir(parents=True, exist_ok=True)
