import numpy as np
from config import D_MODEL, MAX_LEN, BASE

class PositionalEncoding:
    """
    Implements the Positional Encoding mechanism as described in
    'Attention Is All You Need' (Vaswani et al., 2017).
    """
    
    def __init__(self, d_model=D_MODEL, max_len=MAX_LEN, base=BASE):
        self.d_model = d_model
        self.max_len = max_len
        self.base = base
        self.encoding = self._generate_encoding()

    def _generate_encoding(self):
        """
        Generates the PE matrix of shape (max_len, d_model).
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        """
        pe = np.zeros((self.max_len, self.d_model))
        position = np.arange(0, self.max_len).reshape(-1, 1)
        
        # Calculate the divisor (wavelength progression)
        # We only need it for even indices because 2i is used for both sin and cos
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(self.base) / self.d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        return pe

    def get_encoding(self):
        return self.encoding

    def get_cosine_similarity(self):
        """
        Calculates the cosine similarity matrix between all positions.
        """
        # Normalize vectors for cosine similarity
        norm = np.linalg.norm(self.encoding, axis=1, keepdims=True)
        normalized_pe = self.encoding / norm
        return np.dot(normalized_pe, normalized_pe.T)

    def get_wavelengths(self):
        """
        Calculates the wavelength for each dimension i.
        Wavelength = 2*pi * 10000^(2i/d_model)
        """
        indices = np.arange(0, self.d_model, 2)
        wavelengths = 2 * np.pi * np.power(self.base, indices / self.d_model)
        return wavelengths
