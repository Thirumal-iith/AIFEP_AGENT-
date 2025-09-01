# import numpy as np
# import yaml
# from pymdp import utils
# from pymdp.agent import Agent

# class AIFEPAgent:
#     def __init__(self, config):
#         self.config = config

#     def generative_model_construction(self):
#         """Initialize the A, B, C, D, E, pA, pB, pD matrices robustly"""

#         A_factor_list = self.config["A_factor_list"]

#         # --- A matrix ---
#         if self.config.get("A") is not None:
#             self.A = utils.obj_array_from_list([np.array(a) for a in self.config["A"]])
#         elif all(k in self.config for k in ["num_states", "num_obs"]):
#             self.A = utils.random_A_matrix(
#                 self.config["num_obs"], 
#                 self.config["num_states"], 
#                 A_factor_list=A_factor_list
#             )
#         else:
#             raise ValueError("Either 'A' or both 'num_states' and 'num_obs' must be specified.")

#         # --- B matrix ---
#         if self.config.get("B") is not None:
#             self.B = utils.obj_array_from_list([np.array(b) for b in self.config["B"]])
#         elif all(k in self.config for k in ["num_states", "num_controls"]):
#             self.B = utils.random_B_matrix(self.config["num_states"], self.config["num_controls"])
#         else:
#             raise ValueError("Either 'B' or both 'num_states' and 'num_controls' must be specified.")

#         # --- C matrix (preferences) ---
#         if self.config.get("C") is not None:
#             self.C = utils.obj_array_from_list([np.array(c) for c in self.config["C"]])
#         elif "num_obs" in self.config:
#             obs_shapes = [(o,) for o in self.config["num_obs"]]
#             self.C = utils.obj_array_uniform(obs_shapes)
#         else:
#             raise ValueError("Either 'C' or 'num_obs' must be specified.")

#         # --- D matrix (priors over states) ---
#         if self.config.get("D") is not None:
#             self.D = utils.obj_array_from_list([np.array(d) for d in self.config["D"]])
#         elif self.B is not None:
#             num_states = [b.shape[0] for b in self.B]  
#             self.D = utils.obj_array_uniform([(s,) for s in num_states])
#         else:
#             raise ValueError("Either 'D' or 'B' must be provided to infer state sizes.")

#         # --- Priors (Corrected Section) ---
#         # Ensure priors are float arrays to allow for fractional updates during learning
#         pA_config = self.config.get("pA")
#         if pA_config:
#             self.pA = utils.obj_array_from_list([np.array(pa, dtype=float) for pa in pA_config])
#         else:
#             self.pA = utils.obj_array_ones([a.shape for a in self.A])

#         pB_config = self.config.get("pB")
#         if pB_config:
#             self.pB = utils.obj_array_from_list([np.array(pb, dtype=float) for pb in pB_config])
#         else:
#             self.pB = utils.obj_array_ones([b.shape for b in self.B])

#         pD_config = self.config.get("pD")
#         if pD_config:
#             self.pD = utils.obj_array_from_list([np.array(pd, dtype=float) for pd in pD_config])
#         else:
#             self.pD = utils.obj_array_ones([d.shape for d in self.D])

#         # --- Policy prior (optional) ---
#         self.E = self.config.get("E", None)

#     def active_inference_loop(self, obs):
#         """On receiving an observation, execute perception, policy evaluation, and action sampling"""
#         if not (isinstance(obs, list) and all(isinstance(x, int) for x in obs)):
#             obs = [np.argmax(o) if isinstance(o, (list, np.ndarray)) else o for o in obs]

#         qs = self.agent.infer_states(obs)
#         q_pi, neg_efe = self.agent.infer_policies()
#         action = self.agent.sample_action()
        
#         return qs, q_pi, action, neg_efe

#     def learning_update(self, qs_prev, obs, qs_current):
#         """Update the generative model parameters from the experience (A, B, D)"""
#         self.agent.update_A(obs)
#         self.agent.update_B(qs_prev)
#         self.agent.update_D()  

#     def run_AIFEP(self):
#         """Runs the multistep perception-action loop up to time T"""
#         T = self.config["T"]
#         init_obs = self.config["initial_observations"]

#         results = {"states": [], "policies": [], "actions": [], "G": []}
        
#         self.generative_model_construction()

#         # Get learning rates with defaults
#         learning_rates = self.config.get("learning_rates", {})

#         # Create the pymdp Agent with all parameters
#         self.agent = Agent(
#             A=self.A, 
#             B=self.B, 
#             C=self.C, 
#             D=self.D,
#             A_factor_list=self.config["A_factor_list"],
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

#         # Initialize with prior
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
#     with open("active_inference_loop.yaml", "r") as f:
#         config = yaml.safe_load(f)

#     agent = AIFEPAgent(config)
#     output = agent.run_AIFEP()

#     with open("aifep_output.yaml", "w") as f:
#         yaml.dump(output, f)

#     print("✅ AIFEP run completed. Results saved to aifep_output.yaml")


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

    def generative_model_construction(self):
        """Initialize the A, B, C, D, E, pA, pB, pD matrices robustly"""

        A_factor_list = self.config["A_factor_list"]

        # --- A matrix ---
        if self.config.get("A") is not None:
            self.A = utils.obj_array_from_list([np.array(a, dtype=float) for a in self.config["A"]])
        elif all(k in self.config for k in ["num_states", "num_obs"]):
            self.A = utils.random_A_matrix(
                self.config["num_obs"],
                self.config["num_states"],
                A_factor_list=A_factor_list
            )
        else:
            raise ValueError("Either 'A' or both 'num_states' and 'num_obs' must be specified.")

        # --- B matrix ---
        if self.config.get("B") is not None:
            self.B = utils.obj_array_from_list([np.array(b, dtype=float) for b in self.config["B"]])
        elif all(k in self.config for k in ["num_states", "num_controls"]):
            self.B = utils.random_B_matrix(self.config["num_states"], self.config["num_controls"])
        else:
            raise ValueError("Either 'B' or both 'num_states' and 'num_controls' must be specified.")

        # --- C matrix (preferences) ---
        if self.config.get("C") is not None:
            self.C = utils.obj_array_from_list([np.array(c, dtype=float) for c in self.config["C"]])
        elif "num_obs" in self.config:
            obs_shapes = [(o,) for o in self.config["num_obs"]]
            self.C = utils.obj_array_uniform(obs_shapes)
        else:
            raise ValueError("Either 'C' or 'num_obs' must be specified.")

        # --- D matrix (priors over states) ---
        if self.config.get("D") is not None:
            self.D = utils.obj_array_from_list([np.array(d, dtype=float) for d in self.config["D"]])
        elif self.B is not None:
            num_states = [b.shape[0] for b in self.B]
            self.D = utils.obj_array_uniform([(s,) for s in num_states])
        else:
            raise ValueError("Either 'D' or 'B' must be provided to infer state sizes.")


        # --- Priors ---
        pA_config = self.config.get("pA")
        if pA_config:
            self.pA = utils.obj_array_from_list([np.array(pa, dtype=float) for pa in pA_config])
        else:
            self.pA = utils.obj_array_ones([a.shape for a in self.A])

        pB_config = self.config.get("pB")
        if pB_config:
            self.pB = utils.obj_array_from_list([np.array(pb, dtype=float) for pb in pB_config])
        else:
            self.pB = utils.obj_array_ones([b.shape for b in self.B])

        pD_config = self.config.get("pD")
        if pD_config:
            self.pD = utils.obj_array_from_list([np.array(pd, dtype=float) for pd in pD_config])
        else:
            self.pD = utils.obj_array_ones([d.shape for d in self.D])

        # --- Policy prior (optional) ---
        self.E = self.config.get("E", None)

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

        # Create the pymdp Agent with all parameters
        self.agent = Agent(
            A=self.A,
            B=self.B,
            C=self.C,
            D=self.D,
            # A_factor_list=self.config["A_factor_list"],
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

        # Initialize with prior
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
    with open("active_inference_loop.yaml", "r") as f:
        config = yaml.safe_load(f)

    agent = AIFEPAgent(config)
    output = agent.run_AIFEP()

    with open("aifep_output.yaml", "w") as f:
        yaml.dump(output, f, sort_keys=False, default_flow_style=False)

    print("✅ AIFEP run completed. Results saved to aifep_output.yaml")
