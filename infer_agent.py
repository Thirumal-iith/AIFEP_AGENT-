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
            # Copy A_m so we don't overwrite
            p_obs = A_m.copy()

            # Sequentially contract over hidden state factors
            for f, qs_f in enumerate(qs):
                p_obs = np.tensordot(p_obs, qs_f, axes=([1], [0]))
                # each contraction removes one hidden factor axis

            po[m] = p_obs  # final shape: obs_dim
        return po


def load_config_files():
    """Load configuration from YAML files."""
    with open("aifep_agent_output.yaml", "r") as f:
        config_parameters = yaml.safe_load(f)

    with open("infer_observations.yaml", "r") as f:
        config_obs = yaml.safe_load(f)
    
    return config_parameters, config_obs


def reconstruct_model_arrays(config_parameters):
    """Reconstruct model arrays A, B, C, D from configuration."""
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

    return A, B, C, D


def initialize_agent(A, B, C, D):
    """Initialize the InferAgent with model arrays."""
    return InferAgent(A=A, B=B, C=C, D=D)


def run_inference(agent, obs):
    """Run state and policy inference."""
    qs = agent.infer_states(obs)
    q_pi, G = agent.infer_policies()
    return qs, q_pi, G


def predict_observations(agent, A, qs):
    """Predict observations using custom multi-factor method."""
    return agent.predict_observations_multi(A, qs)


def save_outputs(qs, q_pi, G, po, filename="agent2_output.yaml"):
    """Save all outputs to YAML file."""
    output = {
        "qs": [q.tolist() for q in qs],
        "q_pi": q_pi.tolist(),
        "G": G.tolist(),
        "po": [p.tolist() for p in po]
    }

    with open(filename, "w") as f:
        yaml.safe_dump(output, f)


def main():
    """Main function that orchestrates the entire process."""
    # Load configuration files
    config_parameters, config_obs = load_config_files()
    
    # Reconstruct model arrays
    A, B, C, D = reconstruct_model_arrays(config_parameters)
    
    # Get observations
    obs = config_obs["observations"]
    # obs = [[0, 1, 2],[1, 1, 0],[2, 0, 1],[0, 2, 2],[1, 0, 0]]
    
    # Get policies (if available)
    policies = config_parameters.get("policies", [])
    
    # Initialize agent
    agent = initialize_agent(A, B, C, D)
    
    # Run inference
    qs, q_pi, G = run_inference(agent, obs)
    
    # Predict observations
    po = predict_observations(agent, A, qs)
    
    # Save outputs
    save_outputs(qs, q_pi, G, po)
    
    print("Processing completed. Results saved to agent2_output.yaml")


if __name__ == "__main__":
    main()