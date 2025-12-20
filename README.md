Small Description:- Used GNN and DQN to train an ambulance(agent) to find the optimal route considering congestion to get the patient to the nearest hospital.
# NeuroRoute: GNN-DQN Ambulance Routing 🚑🏙️

A Deep Reinforcement Learning project that trains an intelligent ambulance agent to navigate a smart city grid. By combining **Graph Neural Networks (GNN)** with **Deep Q-Learning (DQN)**, the agent learns to perceive dynamic traffic congestion and choose the most efficient path to pick up patients and deliver them to hospitals.

![Project Status](https://img.shields.io/badge/status-active-brightgreen)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/framework-PyTorch-orange)

 Key Features
* **Graph-Aware State:** Uses GNNs to process the city's topology as a graph rather than a flat vector.
* **Dynamic Traffic Simulation:** The environment simulates "live" congestion (Green/Yellow/Red) that updates based on road usage and random events.
* **3D Visualization:** A high-fidelity 3D rendering engine built with **Ursina** to watch the AI make real-time routing decisions.
* **Reward Shaping:** Includes penalties for "missed opportunities" (choosing slow paths when green ones were available) and backtracking.

 Technical Architecture

### 1. Environment (`environment.py`)
A custom `gymnasium` environment representing a 20x20 (adjustable) grid graph. 
- **Edges:** Travel time in seconds.
- **Congestion:** Edges increase in weight as they are used, simulating short-term traffic buildup.

### 2. Brain (`gnn.py`)
The agent uses a **NeuroRouteGNN** architecture:
- **GCN Layers:** Aggregates features from neighboring nodes to understand local traffic context.
- **MLP Head:** Maps node embeddings + edge attributes to Q-values for potential next moves.
- **DQN Training:** Uses a Replay Buffer and Target Network for stable convergence.

### 3. Visualizer (`visualize_ursina.py`)
A 3D simulation where:
- **Orange Cube:** Ambulance searching for patient.
- **Cyan Cube:** Ambulance carrying patient to hospital.
- **Road Colors:** Gradient from Green (Fast) to Red (Congested).

# Getting Started
### Prerequisites
* Python 3.8+
* [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) (Follow their specific install guide for your CUDA/CPU version)

### Installation Procedure
# Clone the repo
git clone [https://github.com/tennissta99660/RL_ambulance_routing_dqn-gnn.git](https://github.com/tennissta99660/RL_ambulance_routing_dqn-gnn)

cd Rl_ambulance_routing_dqn-gnn

# Install dependencies
pip install -r requirements.txt
##  Running the Project

Follow these steps to initialize, train, and visualize the NeuroRoute agent.

### 1️ Initialize the Environment
Verify the grid setup and basic environment parameters.
```bash
python environment.py
```

### 2️ Train the Agent (GNN + DQN)
Start the training loop. This will use the GNN to learn optimal routing and save the model to neuroroute_model_batched_final.pth.

```bash
python gnn.py
```

### 3️ Run the 3D Visualization
Launch the Ursina engine to watch the trained ambulance navigate the city in real-time.
```bash
python visualize_ursina.py
```

# Video representation of the project
![Video Project](https://github.com/user-attachments/assets/97ec17cf-1e1f-482c-a5db-26bfe4d55ed4)
