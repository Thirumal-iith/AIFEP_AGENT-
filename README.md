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

Here’s how you can think of a Pi-hole example in the context of Active Inference + this agent:

Imagine you run a Pi-hole (a network-level ad and tracker blocker). It observes DNS requests (observations), and can **act** by blocking or allowing certain domain name lookups. Your hidden state might be "Is this domain malicious or not?" or "Is this request safe or suspicious?". The Pi-hole has preferences: for example, it prefers to **minimise malicious domains passing through**, but it also wants to **avoid false positives** (blocking good domains).  

- **Observation model**: The Pi-hole sees domain requests, maybe features of requests (source, domain name, frequency).  
- **Hidden states**: Whether each domain is malicious vs benign.  
- **Actions / policies**: Block or allow a domain.  
- **Transition model**: Domains might change over time; maybe a benign domain gets compromised, or a malicious domain becomes less active.  
- **Preferences**: High preference for allowing benign domains, strong negative preference for letting malicious requests through.  

Using Active Inference, the Pi-hole agent infers whether a request is from a malicious or benign domain, predicts possible next observations under different actions (block vs allow), then selects the action which minimises expected free energy — effectively balancing the trade-off between blocking threats and avoiding overblocking.

This agent repo works similarly: you define your observation / transition models, your preferences, and the agent infers and acts.

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
