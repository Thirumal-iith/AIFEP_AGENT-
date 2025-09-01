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
        # Extract dimensions from config for more robust handling
        self.num_states = self._extract_num_states()
        self.num_controls = self._extract_num_controls()
        self.num_obs = self._extract_num_obs()

    def _extract_num_states(self):
        """Extract number of states from various possible config sources"""
        if "num_states" in self.config:
            return self.config["num_states"]
        elif "B" in self.config and self.config["B"]:
            # Try to infer from B matrix structure
            try:
                B_data = self.config["B"][0]  # First factor
                return self._infer_states_from_B_structure(B_data)
            except:
                return [4]  # Default fallback
        elif "D" in self.config and self.config["D"]:
            try:
                D_data = self.config["D"][0]
                if isinstance(D_data, list):
                    return [len(D_data)]
            except:
                pass
        return [4]  # Default fallback

    def _extract_num_controls(self):
        """Extract number of controls from config"""
        if "num_controls" in self.config:
            return self.config["num_controls"]
        elif "B" in self.config and self.config["B"]:
            # Try to infer from B matrix structure
            try:
                B_data = self.config["B"][0]
                return [self._infer_actions_from_B_structure(B_data)]
            except:
                return [2]  # Default fallback
        return [2]  # Default fallback

    def _extract_num_obs(self):
        """Extract number of observations from config"""
        if "num_obs" in self.config:
            return self.config["num_obs"]
        elif "A" in self.config and self.config["A"]:
            try:
                A_data = self.config["A"][0]
                if isinstance(A_data, list):
                    return [len(A_data)]
            except:
                pass
        return [4]  # Default fallback

    def _infer_states_from_B_structure(self, B_data):
        """Try to infer number of states from B matrix structure"""
        if isinstance(B_data, list) and len(B_data) > 0:
            if isinstance(B_data[0], list) and len(B_data[0]) > 0:
                if isinstance(B_data[0][0], list):
                    return len(B_data[0][0])
        return 4

    def _infer_actions_from_B_structure(self, B_data):
        """Try to infer number of actions from B matrix structure"""
        def count_depth_elements(data, target_depth, current_depth=0):
            if current_depth == target_depth:
                return 1
            if isinstance(data, list) and len(data) > 0:
                return sum(count_depth_elements(item, target_depth, current_depth + 1) for item in data)
            return 0
        
        # For nested B structure, actions are typically at depth 1 or 2
        try:
            if isinstance(B_data, list) and len(B_data) == 1 and isinstance(B_data[0], list):
                return len(B_data[0])  # Actions at second level
            elif isinstance(B_data, list):
                return len(B_data)  # Actions at first level
        except:
            pass
        return 2

    def safe_array_conversion(self, data, target_shape=None):
        """Safely convert data to numpy array with proper dtype"""
        try:
            if isinstance(data, list):
                depth = self._get_max_depth(data)
                
                if depth > 4:  # Complex tensor structure
                    arr = self._convert_nested_structure(data)
                else:
                    try:
                        arr = np.array(data, dtype=np.float64)
                    except ValueError:
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
            raise ValueError(f"Failed to convert data to array: {str(e)}")

    def _get_max_depth(self, lst):
        """Get maximum nesting depth of a list"""
        if not isinstance(lst, list):
            return 0
        if not lst:
            return 1
        return 1 + max(self._get_max_depth(item) for item in lst)

    def _flatten_nested_list(self, nested_list):
        """Recursively flatten a nested list structure"""
        result = []
        for item in nested_list:
            if isinstance(item, list):
                result.extend(self._flatten_nested_list(item))
            elif isinstance(item, str):
                if ' - ' in item:
                    parts = item.split(' - ')
                    for part in parts:
                        result.append(float(part.strip()))
                else:
                    result.append(float(item))
            else:
                result.append(float(item))
        return result

    def _convert_nested_structure(self, data):
        """Convert nested structure with dynamic dimension handling"""
        try:
            num_states = self.num_states[0] if isinstance(self.num_states, list) else self.num_states
            num_actions = self.num_controls[0] if isinstance(self.num_controls, list) else self.num_controls
            
            # Handle the complex nested structure case
            if len(data) == 1 and isinstance(data[0], list):
                inner_data = data[0]
                
                if len(inner_data) == num_actions:
                    action_matrices = []
                    for i, action_data in enumerate(inner_data):
                        if isinstance(action_data, list):
                            matrix = self._process_action_matrix(action_data, num_states)
                            if matrix is not None:
                                action_matrices.append(matrix)
                    
                    if len(action_matrices) == num_actions:
                        result = np.stack(action_matrices, axis=2)
                        return result
            
            # Try direct conversion for simpler structures
            try:
                result = np.array(data, dtype=np.float64)
                return result
            except ValueError:
                pass
            
            # Flatten and reshape approach
            flat_data = self._flatten_nested_list(data)
            total_elements = len(flat_data)
            
            # Try to reshape based on inferred dimensions
            expected_elements = num_states * num_states * num_actions
            if total_elements == expected_elements:
                result = np.array(flat_data, dtype=np.float64).reshape(num_states, num_states, num_actions)
                return result
            elif total_elements == num_states * num_states:
                result = np.array(flat_data, dtype=np.float64).reshape(num_states, num_states)
                return result
            else:
                # Try to fit the data into expected shape
                if total_elements >= expected_elements:
                    result = np.array(flat_data[:expected_elements], dtype=np.float64).reshape(num_states, num_states, num_actions)
                    return result
                else:
                    padded_data = flat_data + [0.0] * (expected_elements - total_elements)
                    result = np.array(padded_data, dtype=np.float64).reshape(num_states, num_states, num_actions)
                    return result
            
        except Exception:
            # Ultimate fallback: create identity matrices with correct dimensions
            num_states = self.num_states[0] if isinstance(self.num_states, list) else self.num_states
            num_actions = self.num_controls[0] if isinstance(self.num_controls, list) else self.num_controls
            
            result = np.zeros((num_states, num_states, num_actions))
            for i in range(num_actions):
                result[:, :, i] = np.eye(num_states)
            return result

    def _process_action_matrix(self, action_data, num_states):
        """Process individual action matrix data"""
        try:
            if len(action_data) == num_states and all(isinstance(row, list) and len(row) == num_states for row in action_data):
                return np.array(action_data, dtype=np.float64)
            elif len(action_data) == num_states:
                processed_rows = []
                for row in action_data:
                    if isinstance(row, list):
                        if len(row) == num_states and all(isinstance(x, (int, float)) for x in row):
                            processed_rows.append(row)
                        else:
                            flat_row = self._flatten_nested_list([row])
                            if len(flat_row) >= num_states:
                                processed_rows.append(flat_row[:num_states])
                            else:
                                processed_rows.append(flat_row + [0.0]*(num_states-len(flat_row)))
                    elif isinstance(row, str):
                        if ' - ' in row:
                            parts = [float(x.strip()) for x in row.split(' - ')]
                            if len(parts) >= num_states:
                                processed_rows.append(parts[:num_states])
                            else:
                                processed_rows.append(parts + [0.0]*(num_states-len(parts)))
                        else:
                            row_data = [float(row)] + [0.0]*(num_states-1)
                            processed_rows.append(row_data)
                    else:
                        row_data = [float(row)] + [0.0]*(num_states-1)
                        processed_rows.append(row_data)
                
                if len(processed_rows) == num_states:
                    return np.array(processed_rows, dtype=np.float64)
            
            return None
        except Exception:
            return None

    def _normalize_if_needed(self, arr):
        """Normalize probability matrices (A, B matrices should be normalized)"""
        if arr.ndim == 2:
            col_sums = arr.sum(axis=0)
            col_sums[col_sums == 0] = 1.0
            return arr / col_sums[np.newaxis, :]
        elif arr.ndim == 3:
            normalized = np.zeros_like(arr)
            for i in range(arr.shape[2]):
                col_sums = arr[:, :, i].sum(axis=0)
                col_sums[col_sums == 0] = 1.0
                normalized[:, :, i] = arr[:, :, i] / col_sums[np.newaxis, :]
            return normalized
        else:
            return arr

    def generative_model_construction(self):
        """Initialize the A, B, C, D, E, pA, pB, pD matrices"""

        # --- A matrix ---
        if self.config.get("A") is not None:
            A_list = [self.safe_array_conversion(a_matrix) for a_matrix in self.config["A"]]
            self.A = utils.obj_array([len(A_list)])
            for i, a_arr in enumerate(A_list):
                if a_arr.ndim == 3 and a_arr.shape[0] == 1:
                    a_arr = a_arr[0]
                self.A[i] = a_arr
        elif all(k in self.config for k in ["num_states", "num_obs"]):  
            A_factor_list = self.config.get("A_factor_list", None)
            self.A = utils.random_A_matrix(
                self.config["num_obs"],
                self.config["num_states"],
                A_factor_list=A_factor_list
            )
        else:
            # Generate based on inferred dimensions
            self.A = utils.random_A_matrix(self.num_obs, self.num_states)

        # --- B matrix ---
        if self.config.get("B") is not None:
            B_list = [self.safe_array_conversion(b_matrix) for b_matrix in self.config["B"]]
            self.B = utils.obj_array([len(B_list)])
            for i, b_arr in enumerate(B_list):
                self.B[i] = b_arr
        elif all(k in self.config for k in ["num_states", "num_controls"]): 
            self.B = utils.random_B_matrix(
                self.config["num_states"],
                self.config["num_controls"]
            )
        else:
            # Generate based on inferred dimensions
            self.B = utils.random_B_matrix(self.num_states, self.num_controls)

        # --- C matrix ---
        if self.config.get("C") is not None:
            C_list = [self.safe_array_conversion(c_vector) for c_vector in self.config["C"]]
            self.C = utils.obj_array([len(C_list)])
            for i, c_arr in enumerate(C_list):
                self.C[i] = c_arr
        elif "num_obs" in self.config:  
            obs_shapes = [(o,) for o in self.config["num_obs"]]
            self.C = utils.obj_array_uniform(obs_shapes)
        else:
            # Generate based on inferred dimensions
            obs_shapes = [(o,) for o in self.num_obs]
            self.C = utils.obj_array_uniform(obs_shapes)

        # --- D matrix ---
        if self.config.get("D") is not None:
            D_list = [self.safe_array_conversion(d_vector) for d_vector in self.config["D"]]
            self.D = utils.obj_array([len(D_list)])
            for i, d_arr in enumerate(D_list):
                self.D[i] = d_arr
        elif self.B is not None:   
            num_states = [b.shape[0] for b in self.B]
            self.D = utils.obj_array_uniform([(s,) for s in num_states])
        else:
            # Generate based on inferred dimensions
            self.D = utils.obj_array_uniform([(s,) for s in self.num_states])

        # --- Priors ---
        pA_config = self.config.get("pA")
        if pA_config:
            pA_list = [self.safe_array_conversion(pa_matrix) for pa_matrix in pA_config]
            self.pA = utils.obj_array([len(pA_list)])
            for i, pa_arr in enumerate(pA_list):
                if pa_arr.ndim == 3 and pa_arr.shape[0] == 1:
                    pa_arr = pa_arr[0]
                self.pA[i] = pa_arr
        else:
            self.pA = utils.obj_array_ones([a.shape for a in self.A])

        pB_config = self.config.get("pB")
        if pB_config:
            pB_list = [self.safe_array_conversion(pb_matrix) for pb_matrix in pB_config]
            self.pB = utils.obj_array([len(pB_list)])
            for i, pb_arr in enumerate(pB_list):
                self.pB[i] = pb_arr
        else:
            self.pB = utils.obj_array_ones([b.shape for b in self.B])

        pD_config = self.config.get("pD")
        if pD_config:
            pD_list = [self.safe_array_conversion(pd_vector) for pd_vector in pD_config]
            self.pD = utils.obj_array([len(pD_list)])
            for i, pd_arr in enumerate(pD_list):
                self.pD[i] = pd_arr
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
        
        if len(init_obs) < T:
            extended_obs = init_obs + [init_obs[-1]] * (T - len(init_obs))
            init_obs = extended_obs
        elif len(init_obs) > T:
            init_obs = init_obs[:T]
        
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

    with open("aifep_agent_output.yaml", "w") as f:
        yaml.dump(output, f, sort_keys=False, default_flow_style=False)