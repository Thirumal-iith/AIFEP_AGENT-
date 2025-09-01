import yaml
import numpy as np
from pymdp import utils
from pymdp.agent import Agent as BaseAgent


class InferAgent(BaseAgent):
    """
    Extended Agent class with multi-factor observation prediction.
    Inherits all methods from pymdp.agent.Agent.
    """

    def predict_observations_multi(self, A, qs):
        """
        Predict observations given a multi-factor posterior over hidden states.

        Args:
            A (list of np.ndarray): Likelihood matrices (one per modality).
                Each A[m] has shape (obs_dim, s1, s2, ..., sF)
            qs (list of np.ndarray): Posterior beliefs over each hidden state factor.
                qs[f] has shape (num_states_f,)

        Returns:
            list of np.ndarray: Predicted observation distributions per modality.
        """
        po = utils.obj_array(len(A))
        for m, A_m in enumerate(A):
            # Copy A_m so we don’t overwrite
            p_obs = A_m.copy()

            # Sequentially contract over hidden state factors
            for f, qs_f in enumerate(qs):
                p_obs = np.tensordot(p_obs, qs_f, axes=([1], [0]))
                # each contraction removes one hidden factor axis

            po[m] = p_obs  # final shape: obs_dim
        return po


if __name__ == "__main__":
    # Load input YAML
    with open("final_output.yaml", "r") as f:
        config_parameters = yaml.safe_load(f)

    with open("infer_observations.yaml", "r") as f:
        config_obs = yaml.safe_load(f)

    # Reconstruct model arrays
    A = utils.obj_array(len(config_parameters["final_parameters"]["A"]))
    for i, a in enumerate(config_parameters["final_parameters"]["A"]):
        A[i] = np.array(a)

    B = utils.obj_array(len(config_parameters["final_parameters"]["B"]))
    for i, b in enumerate(config_parameters["final_parameters"]["B"]):
        B[i] = np.array(b)

    C = utils.obj_array(len(config_parameters["final_parameters"]["C"]))
    for i, c in enumerate(config_parameters["final_parameters"]["C"]):
        C[i] = np.array(c)

    D = utils.obj_array(len(config_parameters["final_parameters"]["D"]))
    for i, d in enumerate(config_parameters["final_parameters"]["D"]):
        D[i] = np.array(d)

    obs = config_obs["observations"]       # integer indices for observation
    # obs= [[0, 1, 2],[1, 1, 0],[2, 0, 1],[0, 2, 2],[1, 0, 0]]

    policies = config_parameters.get("policies", [])

    # Initialise extended Agent
    agent = InferAgent(A=A, B=B, C=C, D=D)

    # Inference using built-in methods
    qs = agent.infer_states(obs)
    q_pi, G = agent.infer_policies()

    # Custom multi-factor prediction
    po = agent.predict_observations_multi(A, qs)

    # Save outputs
    output = {
        "qs": [q.tolist() for q in qs],
        "q_pi": q_pi.tolist(),
        "G": G.tolist(),
        "po": [p.tolist() for p in po]
    }

    with open("agent2_output.yaml", "w") as f:
        yaml.safe_dump(output, f)
