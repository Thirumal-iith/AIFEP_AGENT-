import numpy as np
import yaml
from pymdp import utils
from pymdp.agent import Agent

# --- Custom YAML representers for compact matrix-like output ---
def list_presenter(dumper, data):
    """Force lists to use flow style (inline [ ... ]) in YAML"""
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

def float_presenter(dumper, data):
    """Format floats to 8 decimal places (like 0.73105858)"""
    text = f"{data:.8f}"
    return dumper.represent_scalar('tag:yaml.org,2002:float', text)

yaml.add_representer(list, list_presenter)
yaml.add_representer(float, float_presenter)


class AIFEPAgent:
    def __init__(self, config):
        self.config = config

    def safe_array_conversion(self, data, target_shape=None):
        """Safely convert data to numpy array with proper dtype"""
        try:
            # First, let's handle nested list structures more carefully
            if isinstance(data, list):
                # Check if this is a deeply nested structure
                def get_max_depth(lst):
                    if not isinstance(lst, list):
                        return 0
                    if not lst:
                        return 1
                    return 1 + max(get_max_depth(item) for item in lst)
                
                depth = get_max_depth(data)
                print(f"Data depth: {depth}")
                
                if depth > 4:  # This suggests a complex tensor structure
                    # For B matrix: expected shape should be (num_states, num_states, num_actions)
                    # Your data appears to be: (1, 5, 4, 4, 4) based on the YAML structure
                    print(f"Handling complex tensor structure with depth {depth}")
                    arr = self._convert_nested_structure(data)
                else:
                    try:
                        arr = np.array(data, dtype=np.float64)
                    except ValueError as ve:
                        print(f"Standard numpy conversion failed: {ve}")
                        arr = self._convert_nested_structure(data)
            else:
                arr = np.array(data, dtype=np.float64)
                
            # Handle NaN and inf values
            if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
            
            # Normalize matrices if they are probability matrices
            if target_shape is None:
                arr = self._normalize_if_needed(arr)
            
            return np.ascontiguousarray(arr)
            
        except Exception as e:
            print(f"Error in safe_array_conversion: {e}")
            print(f"Data type: {type(data)}")
            if isinstance(data, list) and len(data) > 0:
                print(f"First element type: {type(data[0])}")
            raise e

    def _flatten_nested_list(self, nested_list):
        """Recursively flatten a nested list structure"""
        result = []
        for item in nested_list:
            if isinstance(item, list):
                result.extend(self._flatten_nested_list(item))
            elif isinstance(item, str):
                # Handle string representations like "1.0 - 0.0 - 1.0 - 0.0"
                if ' - ' in item:
                    # Split by ' - ' and convert each part to float
                    parts = item.split(' - ')
                    for part in parts:
                        result.append(float(part.strip()))
                else:
                    result.append(float(item))
            else:
                result.append(float(item))
        return result

    def _convert_nested_structure(self, data):
        """Convert nested structure by trying different approaches"""
        try:
            print(f"Converting nested structure with length: {len(data)}")
            
            # Method 1: Handle the specific structure from your YAML
            if len(data) == 1 and isinstance(data[0], list):
                # This looks like it has an extra outer dimension
                inner_data = data[0]
                print(f"Inner data length: {len(inner_data)}")
                
                if len(inner_data) == 5:  # 5 actions
                    # Try to stack the action matrices
                    action_matrices = []
                    for i, action_data in enumerate(inner_data):
                        print(f"Processing action {i}, type: {type(action_data)}, length: {len(action_data) if isinstance(action_data, list) else 'N/A'}")
                        
                        if isinstance(action_data, list):
                            # Check if this is a 4x4 matrix or needs further processing
                            if len(action_data) == 4 and all(isinstance(row, list) and len(row) == 4 for row in action_data):
                                # This is a proper 4x4 matrix
                                matrix = np.array(action_data, dtype=np.float64)
                                action_matrices.append(matrix)
                            elif len(action_data) == 4:
                                # This might be a nested structure, let's check deeper
                                processed_rows = []
                                for row in action_data:
                                    if isinstance(row, list):
                                        if len(row) == 4 and all(isinstance(x, (int, float)) for x in row):
                                            processed_rows.append(row)
                                        else:
                                            # This row needs further flattening
                                            flat_row = self._flatten_nested_list([row])
                                            if len(flat_row) == 4:
                                                processed_rows.append(flat_row)
                                            else:
                                                # Try to take first 4 elements
                                                processed_rows.append(flat_row[:4] if len(flat_row) >= 4 else flat_row + [0.0]*(4-len(flat_row)))
                                    elif isinstance(row, str):
                                        # Handle string like "1.0 - 0.0 - 1.0 - 0.0"
                                        if ' - ' in row:
                                            parts = [float(x.strip()) for x in row.split(' - ')]
                                            if len(parts) == 4:
                                                processed_rows.append(parts)
                                            else:
                                                processed_rows.append(parts[:4] if len(parts) >= 4 else parts + [0.0]*(4-len(parts)))
                                        else:
                                            processed_rows.append([float(row), 0.0, 0.0, 0.0])
                                    else:
                                        processed_rows.append([float(row), 0.0, 0.0, 0.0])  # Fallback
                                
                                if len(processed_rows) == 4:
                                    matrix = np.array(processed_rows, dtype=np.float64)
                                    action_matrices.append(matrix)
                    
                    print(f"Successfully created {len(action_matrices)} action matrices")
                    if len(action_matrices) == 5:
                        # Stack into (4, 4, 5) - (states, states, actions)
                        result = np.stack(action_matrices, axis=2)
                        print(f"Final B matrix shape: {result.shape}")
                        return result
            
            # Method 2: Try direct conversion for simpler structures
            try:
                result = np.array(data, dtype=np.float64)
                return result
            except ValueError:
                pass
            
            # Method 3: Flatten and reshape
            print("Trying flatten and reshape approach...")
            flat_data = self._flatten_nested_list(data)
            total_elements = len(flat_data)
            print(f"Total flattened elements: {total_elements}")
            
            # For a transition matrix, common shapes are (states, states, actions)
            if total_elements == 80:  # 4*4*5
                result = np.array(flat_data, dtype=np.float64).reshape(4, 4, 5)
                print(f"Reshaped to (4, 4, 5): {result.shape}")
                return result
            elif total_elements == 100:  # 5*4*5 or similar
                # Try different arrangements
                try:
                    result = np.array(flat_data, dtype=np.float64).reshape(5, 4, 5)
                    # Transpose to get (4, 4, 5) if needed
                    if result.shape[1] == 4:
                        result = result[:4, :, :]  # Take first 4 for states
                        result = np.transpose(result, (1, 0, 2))  # Make it (4, 4, 5)
                    return result
                except:
                    pass
            elif total_elements == 16:  # 4*4
                result = np.array(flat_data, dtype=np.float64).reshape(4, 4)
                return result
            else:
                # Just return as 1D array and let the caller handle it
                result = np.array(flat_data, dtype=np.float64)
                print(f"Returning 1D array with {len(result)} elements")
                return result
            
        except Exception as e:
            print(f"Error in _convert_nested_structure: {e}")
            # Ultimate fallback: try to create a valid B matrix structure
            try:
                flat_data = self._flatten_nested_list(data)
                if len(flat_data) >= 80:
                    # Take first 80 elements and reshape to (4, 4, 5)
                    result = np.array(flat_data[:80], dtype=np.float64).reshape(4, 4, 5)
                    return result
                else:
                    # Pad with zeros if needed
                    padded_data = flat_data + [0.0] * (80 - len(flat_data))
                    result = np.array(padded_data, dtype=np.float64).reshape(4, 4, 5)
                    return result
            except:
                # Final fallback: return identity matrices
                print("Using fallback identity matrices for B")
                result = np.zeros((4, 4, 5))
                for i in range(5):
                    result[:, :, i] = np.eye(4)
                return result

    def _normalize_if_needed(self, arr):
        """Normalize probability matrices (A, B matrices should be normalized)"""
        if arr.ndim == 2:
            # For 2D matrices (like A matrix), normalize columns
            col_sums = arr.sum(axis=0)
            col_sums[col_sums == 0] = 1.0  # Avoid division by zero
            return arr / col_sums[np.newaxis, :]
        elif arr.ndim == 3:
            # For 3D matrices (like B matrix), normalize over first axis for each action
            normalized = np.zeros_like(arr)
            for i in range(arr.shape[2]):  # For each action
                col_sums = arr[:, :, i].sum(axis=0)
                col_sums[col_sums == 0] = 1.0  # Avoid division by zero
                normalized[:, :, i] = arr[:, :, i] / col_sums[np.newaxis, :]
            return normalized
        else:
            return arr

    def generative_model_construction(self):
        """Initialize the A, B, C, D, E, pA, pB, pD matrices"""

        # --- A matrix ---
        if self.config.get("A") is not None:
            print("Processing A matrix...")
            A_list = [self.safe_array_conversion(a_matrix) for a_matrix in self.config["A"]]
            self.A = utils.obj_array([len(A_list)])
            for i, a_arr in enumerate(A_list):
                # Ensure A matrix is properly shaped and normalized
                if a_arr.ndim == 3 and a_arr.shape[0] == 1:
                    # Remove the first dimension if it's 1
                    a_arr = a_arr[0]
                self.A[i] = a_arr
                print(f"A[{i}] shape: {a_arr.shape}")
                print(f"A[{i}] column sums: {a_arr.sum(axis=0)}")
        elif all(k in self.config for k in ["num_states", "num_obs"]):  
            A_factor_list = self.config.get("A_factor_list", None)
            self.A = utils.random_A_matrix(
                self.config["num_obs"],
                self.config["num_states"],
                A_factor_list=A_factor_list
            )
        else:
            raise ValueError("Either 'A' or both 'num_states' and 'num_obs' must be specified.")

        # --- B matrix ---
        if self.config.get("B") is not None:
            print("Processing B matrix...")
            B_list = [self.safe_array_conversion(b_matrix) for b_matrix in self.config["B"]]
            self.B = utils.obj_array([len(B_list)])
            for i, b_arr in enumerate(B_list):
                self.B[i] = b_arr
                print(f"B[{i}] shape: {b_arr.shape}")
        elif all(k in self.config for k in ["num_states", "num_controls"]): 
            self.B = utils.random_B_matrix(
                self.config["num_states"],
                self.config["num_controls"]
            )
        else:
            raise ValueError("Either 'B' or both 'num_states' and 'num_controls' must be specified.")

        # --- C matrix ---
        if self.config.get("C") is not None:
            print("Processing C matrix...")
            C_list = [self.safe_array_conversion(c_vector) for c_vector in self.config["C"]]
            self.C = utils.obj_array([len(C_list)])
            for i, c_arr in enumerate(C_list):
                self.C[i] = c_arr
                print(f"C[{i}] shape: {c_arr.shape}")
        elif "num_obs" in self.config:  
            obs_shapes = [(o,) for o in self.config["num_obs"]]
            self.C = utils.obj_array_uniform(obs_shapes)
        else:
            raise ValueError("Either 'C' or 'num_obs' must be specified.")

        # --- D matrix ---
        if self.config.get("D") is not None:
            print("Processing D matrix...")
            D_list = [self.safe_array_conversion(d_vector) for d_vector in self.config["D"]]
            self.D = utils.obj_array([len(D_list)])
            for i, d_arr in enumerate(D_list):
                self.D[i] = d_arr
                print(f"D[{i}] shape: {d_arr.shape}")
        elif self.B is not None:   
            num_states = [b.shape[0] for b in self.B]
            self.D = utils.obj_array_uniform([(s,) for s in num_states])
        else:
            raise ValueError("D or B matrices must be provided in config")

        # --- Priors ---
        pA_config = self.config.get("pA")
        if pA_config:
            print("Processing pA matrix...")
            pA_list = [self.safe_array_conversion(pa_matrix) for pa_matrix in pA_config]
            self.pA = utils.obj_array([len(pA_list)])
            for i, pa_arr in enumerate(pA_list):
                # Ensure pA matrix is properly shaped 
                if pa_arr.ndim == 3 and pa_arr.shape[0] == 1:
                    # Remove the first dimension if it's 1
                    pa_arr = pa_arr[0]
                self.pA[i] = pa_arr
                print(f"pA[{i}] shape: {pa_arr.shape}")
        else:
            self.pA = utils.obj_array_ones([a.shape for a in self.A])

        pB_config = self.config.get("pB")
        if pB_config:
            print("Processing pB matrix...")
            pB_list = [self.safe_array_conversion(pb_matrix) for pb_matrix in pB_config]
            self.pB = utils.obj_array([len(pB_list)])
            for i, pb_arr in enumerate(pB_list):
                self.pB[i] = pb_arr
                print(f"pB[{i}] shape: {pb_arr.shape}")
        else:
            self.pB = utils.obj_array_ones([b.shape for b in self.B])

        pD_config = self.config.get("pD")
        if pD_config:
            print("Processing pD matrix...")
            pD_list = [self.safe_array_conversion(pd_vector) for pd_vector in pD_config]
            self.pD = utils.obj_array([len(pD_list)])
            for i, pd_arr in enumerate(pD_list):
                self.pD[i] = pd_arr
                print(f"pD[{i}] shape: {pd_arr.shape}")
        else:
            self.pD = utils.obj_array_ones([d.shape for d in self.D])

        self.E = self.config.get("E", None)
        
        print("\nMatrix construction completed!")
        print(f"A matrices: {len(self.A)}")
        print(f"B matrices: {len(self.B)}")
        print(f"C matrices: {len(self.C)}")
        print(f"D matrices: {len(self.D)}")

    def active_inference_loop(self, obs):
        """Perception, policy evaluation, action sampling"""
        if not (isinstance(obs, list) and all(isinstance(x, int) for x in obs)):
            obs = [np.argmax(o) if isinstance(o, (list, np.ndarray)) else o for o in obs]
        qs = self.agent.infer_states(obs)
        q_pi, neg_efe = self.agent.infer_policies()
        action = self.agent.sample_action()
        return qs, q_pi, action, neg_efe

    def learning_update(self, qs_prev, obs, qs_current):
        """Update A, B, D parameters"""
        self.agent.update_A(obs)
        self.agent.update_B(qs_prev)
        self.agent.update_D()

    def run_AIFEP(self):
        """Run the full active inference loop"""
        T = self.config["T"]
        init_obs = self.config["initial_observations"]

        print(f"Running AIFEP for {T} timesteps")
        print(f"Initial observations: {init_obs}")
        print(f"Number of initial observations: {len(init_obs)}")

        results = {"states": [], "policies": [], "actions": [], "G": []}
        self.generative_model_construction()

        # Get learning rates from config
        lr_pA = self.config.get("lr_pA", 1.0)
        lr_pB = self.config.get("lr_pB", 1.0)
        lr_pD = self.config.get("lr_pD", 1.0)
        
        self.agent = Agent(
            A=self.A,
            B=self.B,
            C=self.C,
            D=self.D,
            E=self.E,
            pA=self.pA,
            pB=self.pB,
            pD=self.pD,
            save_belief_hist=self.config.get("save_belief_hist", True),
            inference_algo=self.config.get("inference_algo", "VANILLA"),
            lr_pA=lr_pA,
            lr_pB=lr_pB,
            lr_pD=lr_pD
        )

        qs = self.agent.D
        
        # Handle the case where we have fewer observations than timesteps
        if len(init_obs) < T:
            print(f"Warning: Only {len(init_obs)} observations provided for {T} timesteps")
            print("Extending observations by repeating the last observation")
            # Extend with the last observation
            extended_obs = init_obs + [init_obs[-1]] * (T - len(init_obs))
            init_obs = extended_obs
        elif len(init_obs) > T:
            print(f"Warning: {len(init_obs)} observations provided for {T} timesteps")
            print("Truncating observations to match timesteps")
            init_obs = init_obs[:T]
        
        print(f"Using observations: {init_obs}")
        
        for t in range(T):
            print(f"\n--- Timestep {t} ---")
            obs = init_obs[t]
            print(f"Observation: {obs}")
            
            qs_prev = qs
            qs, q_pi, action, G = self.active_inference_loop(obs)
            
            print(f"Inferred states: {[q.tolist() for q in qs]}")
            print(f"Selected action: {action}")
            
            self.learning_update(qs_prev, obs, qs)
            results["states"].append([q.tolist() for q in qs])
            results["policies"].append(q_pi.tolist())
            results["actions"].append(action.tolist() if isinstance(action, np.ndarray) else action)
            results["G"].append(G.tolist())

        results["final_parameters"] = {
            "A": [a.tolist() for a in self.agent.A],
            "B": [b.tolist() for b in self.agent.B],
            "C": [c.tolist() for c in self.agent.C],
            "D": [d.tolist() for d in self.agent.D],
            "pA": [pa.tolist() for pa in self.agent.pA],
            "pB": [pb.tolist() for pb in self.agent.pB],
            "pD": [pd.tolist() for pd in self.agent.pD],
        }
        
        print(f"\nAIFEP completed successfully!")
        return results


if __name__ == "__main__":
    with open("active_inference_loop_corrected.yaml", "r") as f:
        config = yaml.safe_load(f)

    agent = AIFEPAgent(config)
    output = agent.run_AIFEP()

    with open("grid_output.yaml", "w") as f:
        yaml.dump(output, f, sort_keys=False, default_flow_style=False)