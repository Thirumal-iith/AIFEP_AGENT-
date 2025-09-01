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

#         # Added this part for conversion--float64
#     def safe_array_conversion(self, data, target_shape=None):
#         """Safely convert data to numpy array with proper dtype"""
#         arr = np.array(data, dtype=np.float64)
#         if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
#             arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
#         return np.ascontiguousarray(arr)

#     def generative_model_construction(self):
#         """Initialize the A, B, C, D, E, pA, pB, pD matrices"""

#         # --- A matrix ---
#         if self.config.get("A") is not None:
#             A_list = [self.safe_array_conversion(a_matrix) for a_matrix in self.config["A"]]
#             self.A = utils.obj_array([len(A_list)])
#             for i, a_arr in enumerate(A_list):
#                 self.A[i] = a_arr
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
#             B_list = [self.safe_array_conversion(b_matrix) for b_matrix in self.config["B"]]
#             self.B = utils.obj_array([len(B_list)])
#             for i, b_arr in enumerate(B_list):
#                 self.B[i] = b_arr
#         elif all(k in self.config for k in ["num_states", "num_controls"]): 
#             self.B = utils.random_B_matrix(
#                 self.config["num_states"],
#                 self.config["num_controls"]
#             )
#         else:
#             raise ValueError("Either 'B' or both 'num_states' and 'num_controls' must be specified.")

#         # --- C matrix ---
#         if self.config.get("C") is not None:
#             C_list = [self.safe_array_conversion(c_vector) for c_vector in self.config["C"]]
#             self.C = utils.obj_array([len(C_list)])
#             for i, c_arr in enumerate(C_list):
#                 self.C[i] = c_arr
#         elif "num_obs" in self.config:  
#             obs_shapes = [(o,) for o in self.config["num_obs"]]
#             self.C = utils.obj_array_uniform(obs_shapes)
#         else:
#             raise ValueError("Either 'C' or 'num_obs' must be specified.")

#         # --- D matrix ---
#         if self.config.get("D") is not None:
#             D_list = [self.safe_array_conversion(d_vector) for d_vector in self.config["D"]]
#             self.D = utils.obj_array([len(D_list)])
#             for i, d_arr in enumerate(D_list):
#                 self.D[i] = d_arr
#         elif self.B is not None:   
#             num_states = [b.shape[0] for b in self.B]
#             self.D = utils.obj_array_uniform([(s,) for s in num_states])
#         else:
#             raise ValueError("D or B matrices must be provided in config")

#         # --- Priors ---
#         pA_config = self.config.get("pA")
#         if pA_config:
#             pA_list = [self.safe_array_conversion(pa_matrix) for pa_matrix in pA_config]
#             self.pA = utils.obj_array([len(pA_list)])
#             for i, pa_arr in enumerate(pA_list):
#                 self.pA[i] = pa_arr
#         else:
#             self.pA = utils.obj_array_ones([a.shape for a in self.A])

#         pB_config = self.config.get("pB")
#         if pB_config:
#             pB_list = [self.safe_array_conversion(pb_matrix) for pb_matrix in pB_config]
#             self.pB = utils.obj_array([len(pB_list)])
#             for i, pb_arr in enumerate(pB_list):
#                 self.pB[i] = pb_arr
#         else:
#             self.pB = utils.obj_array_ones([b.shape for b in self.B])

#         pD_config = self.config.get("pD")
#         if pD_config:
#             pD_list = [self.safe_array_conversion(pd_vector) for pd_vector in pD_config]
#             self.pD = utils.obj_array([len(pD_list)])
#             for i, pd_arr in enumerate(pD_list):
#                 self.pD[i] = pd_arr
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

#         learning_rates = self.config.get("learning_rates", {})
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
#             lr_pA=learning_rates.get("lr_pA", 1.0),
#             lr_pB=learning_rates.get("lr_pB", 1.0),
#             lr_pD=learning_rates.get("lr_pD", 1.0)
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
#     with open("grid_modified.yaml", "r") as f:
#         config = yaml.safe_load(f)

#     agent = AIFEPAgent(config)
#     output = agent.run_AIFEP()

#     with open("grid_output.yaml", "w") as f:
#         yaml.dump(output, f, sort_keys=False, default_flow_style=False)
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
                    
                    # Let's reshape this appropriately
                    flat_data = self._flatten_nested_list(data)
                    print(f"Flattened data length: {len(flat_data)}")
                    
                    # Try to infer the correct shape based on the config
                    if 'num_controls' in self.config and isinstance(self.config['num_controls'], list):
                        num_actions = self.config['num_controls'][0]  # Assuming single factor
                        # Calculate expected dimensions
                        expected_total = len(flat_data)
                        num_states = int((expected_total / num_actions) ** 0.5)
                        
                        if num_states * num_states * num_actions == expected_total:
                            target_shape = (num_states, num_states, num_actions)
                            arr = np.array(flat_data).reshape(target_shape)
                        else:
                            # Fallback: try to use the original nested structure more carefully
                            arr = self._convert_nested_structure(data)
                    else:
                        arr = self._convert_nested_structure(data)
                else:
                    arr = np.array(data, dtype=np.float64)
            else:
                arr = np.array(data, dtype=np.float64)
                
            # Handle NaN and inf values
            if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
            
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
            else:
                result.append(float(item))
        return result

    def _convert_nested_structure(self, data):
        """Convert nested structure by trying different approaches"""
        try:
            # Method 1: Try to convert as-is but handle the irregular structure
            if len(data) == 1 and isinstance(data[0], list):
                # This looks like it has an extra outer dimension
                inner_data = data[0]
                if len(inner_data) == 5:  # 5 actions
                    # Try to stack the action matrices
                    action_matrices = []
                    for action_data in inner_data:
                        if isinstance(action_data, list) and len(action_data) == 4:
                            # This should be a 4x4 transition matrix
                            matrix = np.array(action_data, dtype=np.float64)
                            action_matrices.append(matrix)
                    
                    if len(action_matrices) == 5:
                        # Stack into (4, 4, 5) - (states, states, actions)
                        result = np.stack(action_matrices, axis=2)
                        return result
            
            # Method 2: Fallback to direct conversion
            return np.array(data, dtype=np.float64)
            
        except Exception as e:
            print(f"Error in _convert_nested_structure: {e}")
            # Last resort: flatten and try to reshape
            flat_data = self._flatten_nested_list(data)
            # Try common shapes
            total_elements = len(flat_data)
            
            # For a transition matrix, common shapes are (states, states, actions)
            if total_elements == 80:  # 4*4*5
                return np.array(flat_data).reshape(4, 4, 5)
            elif total_elements == 16:  # 4*4
                return np.array(flat_data).reshape(4, 4)
            else:
                # Just return as 1D array
                return np.array(flat_data)

    def generative_model_construction(self):
        """Initialize the A, B, C, D, E, pA, pB, pD matrices"""

        # --- A matrix ---
        if self.config.get("A") is not None:
            print("Processing A matrix...")
            A_list = [self.safe_array_conversion(a_matrix) for a_matrix in self.config["A"]]
            self.A = utils.obj_array([len(A_list)])
            for i, a_arr in enumerate(A_list):
                self.A[i] = a_arr
                print(f"A[{i}] shape: {a_arr.shape}")
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
    with open("grid_modified.yaml", "r") as f:
        config = yaml.safe_load(f)

    agent = AIFEPAgent(config)
    output = agent.run_AIFEP()

    with open("grid_output.yaml", "w") as f:
        yaml.dump(output, f, sort_keys=False, default_flow_style=False)