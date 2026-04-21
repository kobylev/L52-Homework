import numpy as np
import pytest

def test_pe_shape(pe_matrix, cfg):
    """PE matrix has correct shape [max_seq_len, d_model]"""
    assert pe_matrix.shape == (cfg.max_seq_len, cfg.d_model)

def test_pe_uniqueness(small_pe):
    """No two rows of PE are identical (all pairwise distances > 0)"""
    # Compute pairwise distances
    from scipy.spatial.distance import pdist
    distances = pdist(small_pe, metric='euclidean')
    assert np.all(distances > 1e-6), "Found identical or nearly identical rows in PE"

def test_pe_norm_stability(pe_matrix, cfg):
    """||PE_pos||_2 ≈ sqrt(d_model/2) for all positions, within 1% tolerance"""
    norms = np.linalg.norm(pe_matrix, axis=1)
    expected_norm = np.sqrt(cfg.d_model / 2)
    
    # Check mean and variation
    assert np.allclose(norms, expected_norm, rtol=1e-2), f"Norms are not stable. Expected {expected_norm}"

def test_linear_transformation(cfg):
    """PE_{pos+k} = M_k @ PE_{pos} holds for k, error < 1e-8"""
    from code.model import PositionalEncoding
    d_model = 128
    k = 5
    pe_obj = PositionalEncoding(d_model=d_model, max_len=100)
    pe = pe_obj.get_encoding()
    
    # Construct M_k
    # For each pair (sin, cos), M_k is a 2x2 rotation matrix
    M_k = np.zeros((d_model, d_model))
    for i in range(0, d_model, 2):
        theta_i = 1.0 / (10000.0 ** (i / d_model))
        angle = k * theta_i
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        M_k[i:i+2, i:i+2] = np.array([
            [cos_a, sin_a],
            [-sin_a, cos_a]
        ])
    
    pos = 10
    pe_pos = pe[pos]
    pe_pos_k = pe[pos+k]
    
    # M_k @ [sin(p), cos(p)] = [sin(p)cos(k) + cos(p)sin(k), -sin(p)sin(k) + cos(p)cos(k)]
    # = [sin(p+k), cos(p+k)]
    transformed = M_k @ pe_pos
    
    assert np.allclose(transformed, pe_pos_k, atol=1e-8)

def test_sine_cosine_unit_circle(pe_matrix):
    """For each frequency pair i, (sin vals, cos vals) across positions lie on unit circle"""
    # Check sin^2 + cos^2 = 1 for all (pos, i) pairs
    sin_vals = pe_matrix[:, 0::2]
    cos_vals = pe_matrix[:, 1::2]
    
    unit_check = sin_vals**2 + cos_vals**2
    assert np.allclose(unit_check, 1.0, atol=1e-12)

def test_cosine_similarity_decay(small_pe):
    """cosine_similarity(PE[0], PE[k]) is generally non-increasing for small k"""
    # Note: Not strictly monotonic due to oscillations, but should show general decay
    def cosine_sim(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
    similarities = [cosine_sim(small_pe[0], small_pe[k]) for k in range(1, 20)]
    # Check that it's not increasing overall (simple trend check)
    assert similarities[0] > similarities[-1]

def test_extrapolation_no_nan(cfg):
    """PE computed for pos > 10000 contains no NaN or Inf values"""
    from code.model import PositionalEncoding
    large_pe_obj = PositionalEncoding(d_model=cfg.d_model, max_len=50000)
    pe = large_pe_obj.get_encoding()
    assert not np.any(np.isnan(pe))
    assert not np.any(np.isinf(pe))
