import numpy as np
import yaml

def check_and_fix_matrix_normalization():
    """Debug script to check and properly fix A matrix normalization"""
    
    # Load the config
    with open("active_inference_loop.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    print("=== Checking A Matrix Normalization ===")
    
    A_config = config.get("A", [])
    
    corrected_A = []
    
    for i, a_matrix in enumerate(A_config):
        print(f"\n--- A Matrix {i} ---")
        a_np = np.asarray(a_matrix, dtype=np.float64)
        print(f"Shape: {a_np.shape}")
        print(f"Original matrix:")
        print(a_np)
        
        # Check normalization along axis 0 (columns should sum to 1)
        column_sums = np.sum(a_np, axis=0)
        print(f"Column sums: {column_sums}")
        
        # Check if normalized
        is_normalized = np.allclose(column_sums, 1.0, rtol=1e-5, atol=1e-8)
        print(f"Is normalized: {is_normalized}")
        
        if not is_normalized:
            print("❌ This matrix is NOT normalized!")
            
            # Fix the matrix
            a_fixed = a_np.copy()
            
            # For each column that doesn't sum to 1, we need to fix it
            for col_idx in range(column_sums.shape[0]):
                for subcol_idx in range(column_sums.shape[1]):
                    col_sum = column_sums[col_idx, subcol_idx]
                    
                    if np.isclose(col_sum, 0.0):
                        print(f"  Column [{col_idx}, {subcol_idx}] sums to 0 - setting to uniform distribution")
                        # Set to uniform distribution
                        a_fixed[:, col_idx, subcol_idx] = 1.0 / a_fixed.shape[0]
                    elif not np.isclose(col_sum, 1.0):
                        print(f"  Column [{col_idx}, {subcol_idx}] sum = {col_sum} - normalizing")
                        # Normalize the column
                        a_fixed[:, col_idx, subcol_idx] = a_fixed[:, col_idx, subcol_idx] / col_sum
            
            # Verify the fix
            new_column_sums = np.sum(a_fixed, axis=0)
            print(f"Fixed column sums: {new_column_sums}")
            print(f"Fixed matrix:")
            print(a_fixed)
            
            corrected_A.append(a_fixed.tolist())
        else:
            print("✅ Matrix is properly normalized")
            corrected_A.append(a_np.tolist())
    
    # Save corrected matrices
    config_corrected = config.copy()
    config_corrected["A"] = corrected_A
    
    with open("active_inference_loop_corrected.yaml", "w") as f:
        yaml.dump(config_corrected, f, sort_keys=False, default_flow_style=False)
    
    print("\n✅ Corrected configuration saved to 'active_inference_loop_corrected.yaml'")
    
    # Verify all matrices are now properly normalized
    print("\n=== Final Verification ===")
    for i, a_matrix in enumerate(corrected_A):
        a_np = np.asarray(a_matrix, dtype=np.float64)
        column_sums = np.sum(a_np, axis=0)
        is_normalized = np.allclose(column_sums, 1.0, rtol=1e-5, atol=1e-8)
        print(f"A[{i}] is normalized: {is_normalized}")
        if not is_normalized:
            print(f"  Column sums: {column_sums}")

if __name__ == "__main__":
    check_and_fix_matrix_normalization()