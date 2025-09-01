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
        # First convert to numpy array with explicit float64 dtype
        arr = np.array(data, dtype=np.float64)
        
        # Ensure no NaN or infinite values
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            print(f"Warning: Found NaN or infinite values in array, replacing with zeros")
            arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Ensure array is contiguous in memory
        arr = np.ascontiguousarray(arr)
        
        return arr

    def generative_model_construction(self):
        """Initialize the A, B, C, D, E, pA, pB, pD matrices robustly"""

        print("Starting generative model construction...")

        # --- A matrix ---
        if self.config.get("A") is not None:
            print("Loading A matrices from config...")
            A_list = []
            for i, a_matrix in enumerate(self.config["A"]):
                print(f"Processing A[{i}]...")
                a_np = self.safe_array_conversion(a_matrix)
                print(f"A[{i}] shape: {a_np.shape}, dtype: {a_np.dtype}")
                
                # Verify normalization
                column_sums = np.sum(a_np, axis=0)
                print(f"A[{i}] column sums: {column_sums}")
                is_normalized = np.allclose(column_sums, 1.0, rtol=1e-5, atol=1e-8)
                print(f"A[{i}] is normalized: {is_normalized}")
                
                if not is_normalized:
                    raise ValueError(f"A[{i}] is not properly normalized!")
                
                A_list.append(a_np)
            
            # Create obj_array but ensure each element is a proper numpy array
            self.A = utils.obj_array([len(A_list)])
            for i, a_arr in enumerate(A_list):
                self.A[i] = a_arr
        else:
            raise ValueError("A matrices must be provided in config")

        # --- B matrix ---
        if self.config.get("B") is not None:
            print("Loading B matrices from config...")
            B_list = []
            for i, b_matrix in enumerate(self.config["B"]):
                b_np = self.safe_array_conversion(b_matrix)
                print(f"B[{i}] shape: {b_np.shape}, dtype: {b_np.dtype}")
                B_list.append(b_np)
            self.B = utils.obj_array([len(B_list)])
            for i, b_arr in enumerate(B_list):
                self.B[i] = b_arr
        else:
            raise ValueError("B matrices must be provided in config")

        # --- C matrix (preferences) ---
        if self.config.get("C") is not None:
            print("Loading C matrices from config...")
            C_list = []
            for i, c_vector in enumerate(self.config["C"]):
                c_np = self.safe_array_conversion(c_vector)
                print(f"C[{i}] shape: {c_np.shape}, dtype: {c_np.dtype}")
                C_list.append(c_np)
            self.C = utils.obj_array([len(C_list)])
            for i, c_arr in enumerate(C_list):
                self.C[i] = c_arr
        else:
            raise ValueError("C matrices must be provided in config")

        # --- D matrix (priors over states) ---
        if self.config.get("D") is not None:
            print("Loading D matrices from config...")
            D_list = []
            for i, d_vector in enumerate(self.config["D"]):
                d_np = self.safe_array_conversion(d_vector)
                print(f"D[{i}] shape: {d_np.shape}, dtype: {d_np.dtype}")
                D_list.append(d_np)
            self.D = utils.obj_array([len(D_list)])
            for i, d_arr in enumerate(D_list):
                self.D[i] = d_arr
        else:
            print("Generating uniform D matrices...")
            num_states = [b.shape[0] for b in self.B]
            self.D = utils.obj_array_uniform([(s,) for s in num_states])
            # Convert to proper dtype and create obj_array
            D_list = []
            for i, d in enumerate(self.D):
                d_np = self.safe_array_conversion(d.tolist())
                D_list.append(d_np)
            self.D = utils.obj_array([len(D_list)])
            for i, d_arr in enumerate(D_list):
                self.D[i] = d_arr

        # --- Priors ---
        print("Processing prior matrices...")
        
        pA_config = self.config.get("pA")
        if pA_config:
            pA_list = []
            for i, pa_matrix in enumerate(pA_config):
                pa_np = self.safe_array_conversion(pa_matrix)
                pA_list.append(pa_np)
            self.pA = utils.obj_array([len(pA_list)])
            for i, pa_arr in enumerate(pA_list):
                self.pA[i] = pa_arr
        else:
            self.pA = utils.obj_array_ones([a.shape for a in self.A])

        pB_config = self.config.get("pB")
        if pB_config:
            pB_list = []
            for i, pb_matrix in enumerate(pB_config):
                pb_np = self.safe_array_conversion(pb_matrix)
                pB_list.append(pb_np)
            self.pB = utils.obj_array([len(pB_list)])
            for i, pb_arr in enumerate(pB_list):
                self.pB[i] = pb_arr
        else:
            self.pB = utils.obj_array_ones([b.shape for b in self.B])

        pD_config = self.config.get("pD")
        if pD_config:
            pD_list = []
            for i, pd_vector in enumerate(pD_config):
                pd_np = self.safe_array_conversion(pd_vector)
                pD_list.append(pd_np)
            self.pD = utils.obj_array([len(pD_list)])
            for i, pd_arr in enumerate(pD_list):
                self.pD[i] = pd_arr
        else:
            self.pD = utils.obj_array_ones([d.shape for d in self.D])

        # --- Policy prior (optional) ---
        self.E = self.config.get("E", None)
        
        print("Generative model construction completed!")

    def active_inference_loop(self, obs):
        """On receiving an observation, execute perception, policy evaluation, and action sampling"""
        if not (isinstance(obs, list) and all(isinstance(x, int) for x in obs)):
            obs = [np.argmax(o) if isinstance(o, (list, np.ndarray)) else o for o in obs]

        qs = self.agent.infer_states(obs)
        q_pi, neg_efe = self.agent.infer_policies()
        action = self.agent.sample_action()

        return qs, q_pi, action, neg_efe

    def learning_update(self, qs_prev, obs, qs_current):
        """Update the generative model parameters from the experience (A, B, D)"""
        self.agent.update_A(obs)
        self.agent.update_B(qs_prev)
        self.agent.update_D()

    def run_AIFEP(self):
        """Runs the multistep perception-action loop up to time T"""
        T = self.config["T"]
        init_obs = self.config["initial_observations"]

        results = {"states": [], "policies": [], "actions": [], "G": []}

        self.generative_model_construction()

        # Get learning rates with defaults
        learning_rates = self.config.get("learning_rates", {})

        print("Creating pymdp Agent...")
        try:
            # Create the pymdp Agent with all parameters
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
                lr_pA=learning_rates.get("lr_pA", 1.0),
                lr_pB=learning_rates.get("lr_pB", 1.0),
                lr_pD=learning_rates.get("lr_pD", 1.0)
            )
            print("✅ Agent created successfully!")
        except Exception as e:
            print(f"❌ Error creating Agent: {e}")
            # Debug information
            print("Debug info:")
            for i, a in enumerate(self.A):
                print(f"A[{i}] shape: {a.shape}, dtype: {a.dtype}")
                try:
                    print(f"A[{i}] has NaN: {np.any(np.isnan(a))}")
                    print(f"A[{i}] has Inf: {np.any(np.isinf(a))}")
                except:
                    print(f"A[{i}] - cannot check for NaN/Inf (dtype issue)")
            raise

        # Initialize with prior
        qs = self.agent.D

        print(f"Starting {T} timestep simulation...")
        for t in range(T):
            print(f"Timestep {t+1}/{T}")
            obs = init_obs[t]

            qs_prev = qs
            qs, q_pi, action, G = self.active_inference_loop(obs)
            self.learning_update(qs_prev, obs, qs)

            results["states"].append([q.tolist() for q in qs])
            results["policies"].append(q_pi.tolist())
            results["actions"].append(action.tolist() if isinstance(action, np.ndarray) else action)
            results["G"].append(G.tolist())

        print("Saving final parameters...")
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
    print("Loading configuration...")
    try:
        # Use the corrected configuration file
        with open("active_inference_loop_corrected.yaml", "r") as f:
            config = yaml.safe_load(f)
        print("✅ Configuration loaded successfully")
    except FileNotFoundError:
        print("❌ Corrected config file not found. Please run the debug script first.")
        exit(1)

    print("Creating AIFEP Agent...")
    agent = AIFEPAgent(config)
    
    print("Running AIFEP simulation...")
    output = agent.run_AIFEP()

    print("Saving results...")
    with open("aifep_output.yaml", "w") as f:
        yaml.dump(output, f, sort_keys=False, default_flow_style=False)

    print("✅ AIFEP run completed. Results saved to aifep_output.yaml")