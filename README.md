# AIFEP_AGENT-

AIFEP_AGENT is a Python-based agent system built with **Active Inference** principles, using the **pymdp** library. This agent is designed to take observations, infer hidden states, and choose actions to minimise a free energy (or expected free energy) objective.  

---

## Table of Contents

- [Overview](#overview)  
- [Features](#features)  
- [Conceptual Example (Pi-hole)](#conceptual‐example-pi-hole)  
- [Repository Structure](#repository‐structure)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Configuration / YAML Specs](#configuration‐yaml‐specs)  
- [Requirements](#requirements)  
- [Contributing](#contributing)  
- [License & Contact](#license‐contact)  

---

## Overview

This agent framework:

- Uses the `pymdp` library to implement components of an Active Inference agent (e.g. generative model, inference over hidden states, policy selection).  
- Supports defining observation models, state transitions, preferences, and action policies via YAML files.  
- Is intended for experimentation — observe how different models / preferences / transition definitions affect the agent’s behavior.

---

## Features

- Modular separation of inference, action selection, and environment / observation configuration  
- Multiple example configurations in YAML to try out different agent behaviors (`agent_input.yaml`, `grid_example.yaml`, etc.)  
- Support for running via scripts (e.g. `aifep.py`, `infer_agent.py`, etc.)  
- Debugging utilities (e.g. `debug_matrix.py`) to inspect internal matrices (transition, observation, preferences)

---

## Conceptual Example (Pi-hole)

## 🧩 Conceptual Example: Pi-hole as an Active Inference Agent

To better understand the agent, let’s map the **Pi-hole** example into a **Reinforcement Learning (RL)–style matrix**, then compare it with **Active Inference**.

---

### RL View (Q-Matrix)

In Reinforcement Learning, we might represent the situation as:

| **State**         | **Action: Allow** | **Action: Block** |
|--------------------|-------------------|-------------------|
| **Benign Domain**  | +1 (✅ correct)   | -1 (🚫 false block)|
| **Malicious Domain** | -10 (❌ threat)  | +5 (✅ blocked)   |

- **Benign domain + Allow** → Positive reward  
- **Benign domain + Block** → Negative reward (annoying false positive)  
- **Malicious domain + Allow** → Very negative reward (security breach)  
- **Malicious domain + Block** → Positive reward (safe)  

This is how **Pi-hole** would be trained in a pure RL setting.

---

### Active Inference View

In Active Inference, instead of learning Q-values directly, the agent:  

1. **Infers hidden states**: “Is this request malicious or benign?”  
2. **Uses observation model**: Based on DNS request features (domain name, frequency, source).  
3. **Selects actions to minimise expected free energy (EFE)**:  
   - Prefers accurate predictions (epistemic value)  
   - Prefers outcomes aligned with security preferences (pragmatic value).  

---


The **Pi-hole analogy** shows how this repo’s agent works:  
- It continuously observes requests,  
- Infers whether they are benign/malicious,  
- Acts (block/allow),  
- And updates its beliefs to minimise future uncertainty + unwanted outcomes.


---

## Repository Structure

Here’s how the files are organised:
AIFEP_AGENT-/
├── aifep.py # Main script, possibly orchestrating inference & action loop
├── aifep_Agent.py # Agent class / implementation using pymdp
├── infer_agent.py # Script focused on inference & action selection
├── agent_infer_runner.py # Runner combining multiple steps / configs
├── debug_matrix.py # Tools to inspect or visualise model matrices
├── final.py # Final version of the agent loop (after experimentation)
├── test*.py # Tests / small scripts to check behavior
├── *.yaml # Many YAML configuration files (observations, input, grid examples etc.)
├── requirements.txt # Dependencies
└── README.md # This file


---

## Installation

```bash
# Clone the repo
git clone https://github.com/Thirumal-iith/AIFEP_AGENT-.git
cd AIFEP_AGENT-

# (Optional) Create virtual environment
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```
## Usage
```bash
# Simple inference+action loop
python aifep.py --config agent_input.yaml

# Using the inference runner
python agent_infer_runner.py --config grid_example.yaml

# Debug / inspect matrices
python debug_matrix.py --config agent_input.yaml
```

## Configuration / YAML Specs
```bash
# Many aspects of behaviour are controlled via the YAML files:

# agent_input.yaml
#   → Defines observation model, transition probabilities, actions, preferences

# grid_example.yaml
#   → Example environment / domain (e.g. grid scenarios)

# active_inference_loop.yaml / active_inference_loop_corrected.yaml
#   → Full loop definitions, possibly corrected variants

# Other *.yaml
#   → Inputs, expected outputs, debugging config, etc.
```
