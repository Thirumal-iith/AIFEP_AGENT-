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
        arr = np.array(data, dtype=np.float64)
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        return np.ascontiguousarray(arr)

    def convert_nested_lists_to_arrays(self, nested_list):
        """Convert nested lists to properly shaped numpy arrays"""
        if nested_list is None:
            return None
        
        # Convert to numpy array while preserving structure
        return self.safe_array_conversion(nested_list)
    
    def normalize_matrix(self, matrix, axis=0):
        """Normalize matrix so that columns sum to 1.0"""
        matrix = np.array(matrix, dtype=np.float64)
        
        # Handle different dimensions
        if matrix.ndim == 1:
            # For 1D arrays, normalize so sum equals 1
            total = np.sum(matrix)
            if total > 0:
                return matrix / total
            else:
                return np.ones_like(matrix) / len(matrix)
        
        elif matrix.ndim == 2:
            # For 2D arrays, normalize columns (axis=0) or rows (axis=1)
            sums = np.sum(matrix, axis=axis, keepdims=True)
            # Avoid division by zero
            sums = np.where(sums == 0, 1.0, sums)
            return matrix / sums
            
        elif matrix.ndim == 3:
            # For 3D arrays (observation x state x factor), normalize over observation dimension
            sums = np.sum(matrix, axis=0, keepdims=True)
            sums = np.where(sums == 0, 1.0, sums)
            return matrix / sums
            
        else:
            # For higher dimensions, try normalizing over first axis
            sums = np.sum(matrix, axis=0, keepdims=True)
            sums = np.where(sums == 0, 1.0, sums)
            return matrix / sums

    def generative_model_construction(self):
        """Initialize the A, B, C, D, E, pA, pB, pD matrices"""

        # --- A matrix ---
        if self.config.get("A") is not None:
            A_config = self.config["A"]
            self.A = utils.obj_array([len(A_config)])
            for i, a_matrix in enumerate(A_config):
                # Convert the nested list format to numpy array and normalize
                converted_matrix = self.convert_nested_lists_to_arrays(a_matrix)
                # Normalize the A matrix (columns should sum to 1)
                self.A[i] = self.normalize_matrix(converted_matrix, axis=0)
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
            B_config = self.config["B"]
            self.B = utils.obj_array([len(B_config)])
            for i, b_matrix in enumerate(B_config):
                # Convert the nested list format to numpy array and normalize
                converted_matrix = self.convert_nested_lists_to_arrays(b_matrix)
                # Normalize the B matrix (columns should sum to 1)
                self.B[i] = self.normalize_matrix(converted_matrix, axis=0)
        elif all(k in self.config for k in ["num_states", "num_controls"]): 
            self.B = utils.random_B_matrix(
                self.config["num_states"],
                self.config["num_controls"]
            )
        else:
            raise ValueError("Either 'B' or both 'num_states' and 'num_controls' must be specified.")

        # --- C matrix ---
        if self.config.get("C") is not None:
            C_config = self.config["C"]
            self.C = utils.obj_array([len(C_config)])
            for i, c_vector in enumerate(C_config):
                # Convert the list format to numpy array
                self.C[i] = self.convert_nested_lists_to_arrays(c_vector)
        elif "num_obs" in self.config:  
            obs_shapes = [(o,) for o in self.config["num_obs"]]
            self.C = utils.obj_array_uniform(obs_shapes)
        else:
            raise ValueError("Either 'C' or 'num_obs' must be specified.")

        # --- D matrix ---
        if self.config.get("D") is not None:
            D_config = self.config["D"]
            self.D = utils.obj_array([len(D_config)])
            for i, d_vector in enumerate(D_config):
                # Convert the list format to numpy array
                self.D[i] = self.convert_nested_lists_to_arrays(d_vector)
        elif self.B is not None:   
            num_states = [b.shape[0] for b in self.B]
            self.D = utils.obj_array_uniform([(s,) for s in num_states])
        else:
            raise ValueError("D or B matrices must be provided in config")

        # --- Priors ---
        pA_config = self.config.get("pA")
        if pA_config:
            self.pA = utils.obj_array([len(pA_config)])
            for i, pa_matrix in enumerate(pA_config):
                self.pA[i] = self.convert_nested_lists_to_arrays(pa_matrix)
        else:
            self.pA = utils.obj_array_ones([a.shape for a in self.A])

        pB_config = self.config.get("pB")
        if pB_config:
            self.pB = utils.obj_array([len(pB_config)])
            for i, pb_matrix in enumerate(pB_config):
                self.pB[i] = self.convert_nested_lists_to_arrays(pb_matrix)
        else:
            self.pB = utils.obj_array_ones([b.shape for b in self.B])

        pD_config = self.config.get("pD")
        if pD_config:
            self.pD = utils.obj_array([len(pD_config)])
            for i, pd_vector in enumerate(pD_config):
                self.pD[i] = self.convert_nested_lists_to_arrays(pd_vector)
        else:
            self.pD = utils.obj_array_ones([d.shape for d in self.D])

        self.E = self.config.get("E", None)

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

        results = {"states": [], "policies": [], "actions": [], "G": []}
        self.generative_model_construction()

        # Extract learning rates from config (updated to match new format)
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
            lr_pA=self.config.get("lr_pA", 1.0),
            lr_pB=self.config.get("lr_pB", 1.0),
            lr_pD=self.config.get("lr_pD", 1.0),
            gamma=self.config.get("gamma", 16.0),
            alpha=self.config.get("alpha", 16.0),
            use_utility=bool(self.config.get("use_utility", 1)),
            use_states_info_gain=bool(self.config.get("use_states_info_gain", 1)),
            use_param_info_gain=bool(self.config.get("use_param_info_gain", 0)),
            action_selection=self.config.get("action_selection", "deterministic"),
            sampling_mode=self.config.get("sampling_mode", "marginal"),
            policy_len=self.config.get("policy_len", 1),
            inference_horizon=self.config.get("inference_horizon", 1),
            control_fac_idx=self.config.get("control_fac_idx", None),
            policies=self.config.get("policies", None),
            inference_params=self.config.get("inference_params", None)
        )

        qs = self.agent.D
        for t in range(T):
            obs = init_obs[t]
            qs_prev = qs
            qs, q_pi, action, G = self.active_inference_loop(obs)
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
        return results


if __name__ == "__main__":
    with open("grid_example.yaml", "r") as f:
        config = yaml.safe_load(f)

    agent = AIFEPAgent(config)
    output = agent.run_AIFEP()

    with open("grid_output.yaml", "w") as f:
        yaml.dump(output, f, sort_keys=False, default_flow_style=False)
# import numpy as np
# import yaml
# from pymdp import utils
# from pymdp.agent import Agent

# # --- Custom YAML representers for compact matrix-like output ---
# def list_presenter(dumper, data):
#     """Force lists to use flow style (inline [ ... ]) in YAML"""
#     return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

# def float_presenter(dumper, data):
#     """Format floats to 8 decimal places (like 0.73105858)"""
#     text = f"{data:.8f}"
#     return dumper.represent_scalar('tag:yaml.org,2002:float', text)

# yaml.add_representer(list, list_presenter)
# yaml.add_representer(float, float_presenter)


# class AIFEPAgent:
#     def __init__(self, config):
#         self.config = config
        
#     def safe_array_conversion(self, data, target_shape=None):
#         """Safely convert data to numpy array with proper dtype"""
#         arr = np.array(data, dtype=np.float64)
#         if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
#             arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
#         return np.ascontiguousarray(arr)

#     def convert_nested_lists_to_arrays(self, nested_list):
#         """Convert nested lists to properly shaped numpy arrays"""
#         if nested_list is None:
#             return None
        
#         # Convert to numpy array while preserving structure
#         return self.safe_array_conversion(nested_list)
    
#     def normalize_matrix(self, matrix, matrix_type="A"):
#         """Normalize matrix according to pymdp conventions"""
#         matrix = np.array(matrix, dtype=np.float64)
        
#         print(f"Normalizing {matrix_type} matrix with shape: {matrix.shape}")
#         print(f"Matrix before normalization:\n{matrix}")
        
#         if matrix_type == "A":
#             # A matrices: Each column should sum to 1 across observations
#             if matrix.ndim == 2:
#                 # 2D A matrix: (obs_dim, state_dim)
#                 sums = np.sum(matrix, axis=0, keepdims=True)
#                 sums = np.where(sums == 0, 1.0, sums)
#                 result = matrix / sums
#             elif matrix.ndim == 3:
#                 # 3D A matrix: Need to understand the structure better
#                 # Let's assume it's (modality, obs_dim, state_dim) and squeeze first dimension
#                 if matrix.shape[0] == 1:
#                     # Remove singleton first dimension and treat as 2D
#                     matrix_2d = matrix.squeeze(0)
#                     sums = np.sum(matrix_2d, axis=0, keepdims=True)
#                     sums = np.where(sums == 0, 1.0, sums)
#                     result_2d = matrix_2d / sums
#                     # Add back the singleton dimension
#                     result = result_2d[np.newaxis, ...]
#                 else:
#                     # Standard 3D normalization
#                     sums = np.sum(matrix, axis=1, keepdims=True)  # Sum over obs dimension
#                     sums = np.where(sums == 0, 1.0, sums)
#                     result = matrix / sums
#             else:
#                 raise ValueError(f"Unexpected A matrix dimensions: {matrix.ndim}")
                
#         elif matrix_type == "B":
#             # B matrices: normalize over next state dimension
#             if matrix.ndim == 2:
#                 # 2D B matrix: (state_dim, state_dim)
#                 sums = np.sum(matrix, axis=0, keepdims=True)
#                 sums = np.where(sums == 0, 1.0, sums)
#                 result = matrix / sums
#             elif matrix.ndim == 3:
#                 # 3D B matrix: (action_dim, state_dim, state_dim)
#                 # For each action, normalize each transition matrix's columns
#                 result = np.zeros_like(matrix)
#                 for action in range(matrix.shape[0]):
#                     transition_matrix = matrix[action]
#                     sums = np.sum(transition_matrix, axis=0, keepdims=True)
#                     sums = np.where(sums == 0, 1.0, sums)
#                     result[action] = transition_matrix / sums
#             else:
#                 raise ValueError(f"Unexpected B matrix dimensions: {matrix.ndim}")
#         else:
#             # Default normalization for other matrices
#             if matrix.ndim == 1:
#                 total = np.sum(matrix)
#                 if total > 0:
#                     result = matrix / total
#                 else:
#                     result = np.ones_like(matrix) / len(matrix)
#             else:
#                 sums = np.sum(matrix, axis=0, keepdims=True)
#                 sums = np.where(sums == 0, 1.0, sums)
#                 result = matrix / sums
            
#         print(f"Matrix after normalization:\n{result}")
        
#         # Verify normalization
#         if matrix_type == "A":
#             if result.ndim == 2:
#                 column_sums = np.sum(result, axis=0)
#             elif result.ndim == 3:
#                 if result.shape[0] == 1:
#                     # For squeezed case, check the 2D matrix
#                     column_sums = np.sum(result.squeeze(0), axis=0)
#                 else:
#                     column_sums = np.sum(result, axis=1)
#         elif matrix_type == "B":
#             if result.ndim == 2:
#                 column_sums = np.sum(result, axis=0)
#             elif result.ndim == 3:
#                 # Check each action's transition matrix
#                 column_sums = []
#                 for action in range(result.shape[0]):
#                     action_sums = np.sum(result[action], axis=0)
#                     column_sums.append(action_sums)
#                 column_sums = np.array(column_sums)
#         else:
#             if result.ndim == 1:
#                 column_sums = np.sum(result)
#             else:
#                 column_sums = np.sum(result, axis=0)
            
#         print(f"Column sums after normalization: {column_sums}")
#         print("---")
        
#         return result

#     def validate_normalization(self, matrix, matrix_type="A", tolerance=1e-10):
#         """Validate that matrices are normalized according to pymdp standards"""
#         if matrix_type == "A":
#             if matrix.ndim == 2:
#                 # For 2D A matrix, columns should sum to 1
#                 column_sums = np.sum(matrix, axis=0)
#                 expected = np.ones(matrix.shape[1])
#             elif matrix.ndim == 3:
#                 if matrix.shape[0] == 1:
#                     # For singleton first dimension
#                     column_sums = np.sum(matrix.squeeze(0), axis=0)
#                     expected = np.ones(matrix.shape[2])
#                 else:
#                     # For 3D matrix, sum over observation dimension
#                     column_sums = np.sum(matrix, axis=1)
#                     expected = np.ones((matrix.shape[0], matrix.shape[2]))
                    
#             is_normalized = np.allclose(column_sums, expected, atol=tolerance)
            
#         elif matrix_type == "B":
#             if matrix.ndim == 2:
#                 column_sums = np.sum(matrix, axis=0)
#                 expected = np.ones(matrix.shape[1])
#                 is_normalized = np.allclose(column_sums, expected, atol=tolerance)
#             elif matrix.ndim == 3:
#                 is_normalized = True
#                 for action in range(matrix.shape[0]):
#                     action_matrix = matrix[action]
#                     column_sums = np.sum(action_matrix, axis=0)
#                     expected = np.ones(action_matrix.shape[1])
#                     if not np.allclose(column_sums, expected, atol=tolerance):
#                         is_normalized = False
#                         print(f"Action {action} matrix not normalized: sums = {column_sums}")
#                         break
                        
#         print(f"{matrix_type} matrix normalization check: {'PASS' if is_normalized else 'FAIL'}")
#         return is_normalized

#     def generative_model_construction(self):
#         """Initialize the A, B, C, D, E, pA, pB, pD matrices"""

#         # --- A matrix ---
#         if self.config.get("A") is not None:
#             A_config = self.config["A"]
#             self.A = utils.obj_array([len(A_config)])
#             for i, a_matrix in enumerate(A_config):
#                 # Convert the nested list format to numpy array and normalize
#                 converted_matrix = self.convert_nested_lists_to_arrays(a_matrix)
#                 # Normalize the A matrix
#                 self.A[i] = self.normalize_matrix(converted_matrix, matrix_type="A")
#                 # Validate normalization
#                 self.validate_normalization(self.A[i], matrix_type="A")
#         elif all(k in self.config for k in ["num_states", "num_obs"]):  
#             A_factor_list = self.config.get("A_factor_list", None)
#             self.A = utils.random_A_matrix(
#                 self.config["num_obs"],
#                 self.config["num_states"],
#                 A_factor_list=A_factor_list
#             )
#         else:
#             raise ValueError("Either 'A' or both 'num_states' and 'num_obs' must be specified.")

#         # --- B matrix ---
#         if self.config.get("B") is not None:
#             B_config = self.config["B"]
#             self.B = utils.obj_array([len(B_config)])
#             for i, b_matrix in enumerate(B_config):
#                 # Convert the nested list format to numpy array and normalize
#                 converted_matrix = self.convert_nested_lists_to_arrays(b_matrix)
#                 # Normalize the B matrix
#                 self.B[i] = self.normalize_matrix(converted_matrix, matrix_type="B")
#                 # Validate normalization
#                 self.validate_normalization(self.B[i], matrix_type="B")
#         elif all(k in self.config for k in ["num_states", "num_controls"]): 
#             self.B = utils.random_B_matrix(
#                 self.config["num_states"],
#                 self.config["num_controls"]
#             )
#         else:
#             raise ValueError("Either 'B' or both 'num_states' and 'num_controls' must be specified.")

#         # --- C matrix ---
#         if self.config.get("C") is not None:
#             C_config = self.config["C"]
#             self.C = utils.obj_array([len(C_config)])
#             for i, c_vector in enumerate(C_config):
#                 # Convert the list format to numpy array
#                 self.C[i] = self.convert_nested_lists_to_arrays(c_vector)
#         elif "num_obs" in self.config:  
#             obs_shapes = [(o,) for o in self.config["num_obs"]]
#             self.C = utils.obj_array_uniform(obs_shapes)
#         else:
#             raise ValueError("Either 'C' or 'num_obs' must be specified.")

#         # --- D matrix ---
#         if self.config.get("D") is not None:
#             D_config = self.config["D"]
#             self.D = utils.obj_array([len(D_config)])
#             for i, d_vector in enumerate(D_config):
#                 # Convert the list format to numpy array and normalize as probability
#                 converted_vector = self.convert_nested_lists_to_arrays(d_vector)
#                 self.D[i] = self.normalize_matrix(converted_vector, matrix_type="D")
#         elif self.B is not None:   
#             num_states = [b.shape[-1] for b in self.B]  # Use last dimension for state count
#             self.D = utils.obj_array_uniform([(s,) for s in num_states])
#         else:
#             raise ValueError("D or B matrices must be provided in config")

#         # --- Priors ---
#         pA_config = self.config.get("pA")
#         if pA_config:
#             self.pA = utils.obj_array([len(pA_config)])
#             for i, pa_matrix in enumerate(pA_config):
#                 self.pA[i] = self.convert_nested_lists_to_arrays(pa_matrix)
#         else:
#             self.pA = utils.obj_array_ones([a.shape for a in self.A])

#         pB_config = self.config.get("pB")
#         if pB_config:
#             self.pB = utils.obj_array([len(pB_config)])
#             for i, pb_matrix in enumerate(pB_config):
#                 self.pB[i] = self.convert_nested_lists_to_arrays(pb_matrix)
#         else:
#             self.pB = utils.obj_array_ones([b.shape for b in self.B])

#         pD_config = self.config.get("pD")
#         if pD_config:
#             self.pD = utils.obj_array([len(pD_config)])
#             for i, pd_vector in enumerate(pD_config):
#                 self.pD[i] = self.convert_nested_lists_to_arrays(pd_vector)
#         else:
#             self.pD = utils.obj_array_ones([d.shape for d in self.D])

#         self.E = self.config.get("E", None)

#     def active_inference_loop(self, obs):
#         """Perception, policy evaluation, action sampling"""
#         if not (isinstance(obs, list) and all(isinstance(x, int) for x in obs)):
#             obs = [np.argmax(o) if isinstance(o, (list, np.ndarray)) else o for o in obs]
#         qs = self.agent.infer_states(obs)
#         q_pi, neg_efe = self.agent.infer_policies()
#         action = self.agent.sample_action()
#         return qs, q_pi, action, neg_efe

#     def learning_update(self, qs_prev, obs, qs_current):
#         """Update A, B, D parameters"""
#         self.agent.update_A(obs)
#         self.agent.update_B(qs_prev)
#         self.agent.update_D()

#     def run_AIFEP(self):
#         """Run the full active inference loop"""
#         T = self.config["T"]
#         init_obs = self.config["initial_observations"]

#         results = {"states": [], "policies": [], "actions": [], "G": []}
#         self.generative_model_construction()

#         # Extract learning rates from config (updated to match new format)
#         self.agent = Agent(
#             A=self.A,
#             B=self.B,
#             C=self.C,
#             D=self.D,
#             E=self.E,
#             pA=self.pA,
#             pB=self.pB,
#             pD=self.pD,
#             save_belief_hist=self.config.get("save_belief_hist", True),
#             inference_algo=self.config.get("inference_algo", "VANILLA"),
#             lr_pA=self.config.get("lr_pA", 1.0),
#             lr_pB=self.config.get("lr_pB", 1.0),
#             lr_pD=self.config.get("lr_pD", 1.0),
#             gamma=self.config.get("gamma", 16.0),
#             alpha=self.config.get("alpha", 16.0),
#             use_utility=bool(self.config.get("use_utility", 1)),
#             use_states_info_gain=bool(self.config.get("use_states_info_gain", 1)),
#             use_param_info_gain=bool(self.config.get("use_param_info_gain", 0)),
#             action_selection=self.config.get("action_selection", "deterministic"),
#             sampling_mode=self.config.get("sampling_mode", "marginal"),
#             policy_len=self.config.get("policy_len", 1),
#             inference_horizon=self.config.get("inference_horizon", 1),
#             control_fac_idx=self.config.get("control_fac_idx", None),
#             policies=self.config.get("policies", None),
#             inference_params=self.config.get("inference_params", None)
#         )

#         qs = self.agent.D
#         for t in range(T):
#             obs = init_obs[t]
#             qs_prev = qs
#             qs, q_pi, action, G = self.active_inference_loop(obs)
#             self.learning_update(qs_prev, obs, qs)
#             results["states"].append([q.tolist() for q in qs])
#             results["policies"].append(q_pi.tolist())
#             results["actions"].append(action.tolist() if isinstance(action, np.ndarray) else action)
#             results["G"].append(G.tolist())

#         results["final_parameters"] = {
#             "A": [a.tolist() for a in self.agent.A],
#             "B": [b.tolist() for b in self.agent.B],
#             "C": [c.tolist() for c in self.agent.C],
#             "D": [d.tolist() for d in self.agent.D],
#             "pA": [pa.tolist() for pa in self.agent.pA],
#             "pB": [pb.tolist() for pb in self.agent.pB],
#             "pD": [pd.tolist() for pd in self.agent.pD],
#         }
#         return results


# if __name__ == "__main__":
#     with open("grid_example.yaml", "r") as f:
#         config = yaml.safe_load(f)

#     agent = AIFEPAgent(config)
#     output = agent.run_AIFEP()

#     with open("f_output.yaml", "w") as f:
#         yaml.dump(output, f, sort_keys=False, default_flow_style=False)