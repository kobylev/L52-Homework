import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from config.config import Config
from code.model import PositionalEncoding  # Using existing model path

@pytest.fixture
def cfg():
    return Config()

@pytest.fixture
def pe_matrix(cfg):
    """Full PE matrix [max_seq_len, d_model]"""
    pe_obj = PositionalEncoding(d_model=cfg.d_model, max_len=cfg.max_seq_len)
    return pe_obj.get_encoding()

@pytest.fixture
def small_pe(cfg):
    """Small PE matrix for fast tests"""
    pe_obj = PositionalEncoding(d_model=64, max_len=100)
    return pe_obj.get_encoding()
