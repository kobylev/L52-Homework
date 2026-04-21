from model import PositionalEncoding
import visualize
import numpy as np

def main():
    print("Starting Positional Encoding Analysis...")
    
    # Initialize Model
    pe_obj = PositionalEncoding()
    pe_matrix = pe_obj.get_encoding()
    
    # 1. Generate PE Matrix Heatmap
    visualize.plot_pe_matrix(pe_matrix)
    
    # 2. Generate Wavelength Progression
    wavelengths = pe_obj.get_wavelengths()
    visualize.plot_wavelengths(wavelengths)
    
    # 3. Generate Specific Position Line Plot
    visualize.plot_position_vector(pe_matrix)
    
    # 4. Generate Similarity Heatmap
    similarity_matrix = pe_obj.get_cosine_similarity()
    visualize.plot_similarity_heatmap(similarity_matrix)
    
    # 5. Generate Unit Circle Plot
    visualize.plot_unit_circle(pe_matrix)
    
    # 6. Generate Norm Stability Plot
    visualize.plot_norm_stability(pe_matrix)
    
    print("All assets generated successfully in docs/assets/")

if __name__ == "__main__":
    main()
