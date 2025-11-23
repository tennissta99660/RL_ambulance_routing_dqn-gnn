# f1.py
import gymnasium as gym
from gymnasium import spaces
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

class SmartCityEnv(gym.Env):
    """
    Grid-graph environment where edge 'weight' represents travel_time (seconds).
    - color classes (for visualization): GREEN <= 0.5s, YELLOW <= 1.0s, RED > 1.0s
    - env.random_traffic toggles background traffic noise (use False for stable training)
    - env.last_node is exposed for training algorithms that want to penalize backtracking
    """

    def __init__(self, grid_size=20):
        super(SmartCityEnv, self).__init__()
        self.grid_size = grid_size

        # 1. Create the City Map (Grid Graph)
        self.G = nx.grid_2d_graph(grid_size, grid_size)
        # Convert tuple nodes (0,1) to integers 0, 1, 2... for easier indexing
        self.G = nx.convert_node_labels_to_integers(self.G)
        self.num_nodes = self.G.number_of_nodes()

        # Initialize edge travel times (seconds). Typical baseline between 0.2 and 1.2s
        for u, v in self.G.edges():
            self.G[u][v].setdefault('weight', float(np.random.uniform(0.2, 1.2)))

        # Action: choose index into current node's neighbor list
        self.action_space = spaces.Discrete(4)  # up to 4 neighbors in grid

        # Observation: simplified flattened vector (ambulance, patient, avg traffic per node)
        self.observation_space = spaces.Box(low=0.0, high=10.0, shape=(self.num_nodes + 2,), dtype=np.float32)

        # State bookkeeping
        self.ambulance_loc = 0
        self.patient_loc = 0
        self.hospital_loc = self.num_nodes - 1
        self.has_patient = False

        # Training/visualization helpers
        self.last_node = None           # set on each step
        self.random_traffic = True     # background random updates (disable during training for stability)

    def reset(self, seed=None):
        super().reset(seed=seed)
        # place ambulance at node 0, patient randomly, hospital at last node
        self.ambulance_loc = 0
        self.patient_loc = np.random.randint(1, self.num_nodes - 1)
        self.has_patient = False
        self.last_node = None
        return self._get_obs(), {}

    def _get_obs(self):
        # observation: [ambulance_node, patient_node, avg_outgoing_traffic for each node]
        obs = np.zeros(self.num_nodes + 2, dtype=np.float32)
        obs[0] = float(self.ambulance_loc)
        obs[1] = float(self.patient_loc)
        for i in range(self.num_nodes):
            edges = list(self.G.edges(i, data='weight'))
            if edges:
                avg_traffic = np.mean([w for _, _, w in edges])
            else:
                avg_traffic = 0.0
            obs[i + 2] = float(avg_traffic)
        return obs

    def step(self, action):
        """
        action: integer index into neighbors list of current node.
        returns: obs, reward, terminated, truncated(False), info
        """

        # thresholds (seconds)
        GREEN_MAX = 0.5
        YELLOW_MAX = 1.0
        RED_MAX = 2.0  # cap

        prev_node = self.ambulance_loc
        neighbors = list(self.G.neighbors(self.ambulance_loc))

        # choose target based on action index
        if action < len(neighbors):
            target_node = neighbors[action]
            travel_time = float(self.G[self.ambulance_loc][target_node].get('weight', 0.6))
        else:
            # invalid action -> heavy time cost (discourages do-nothing)
            target_node = self.ambulance_loc
            travel_time = RED_MAX

        # base reward: negative travel time (minimize time)
        reward = -travel_time
        terminated = False

        # move ambulance
        self.ambulance_loc = target_node

        # pickup / deliver rewards
        if (not self.has_patient) and (self.ambulance_loc == self.patient_loc):
            reward += 50.0
            self.has_patient = True

        if self.has_patient and (self.ambulance_loc == self.hospital_loc):
            reward += 100.0
            terminated = True

        # MISSED-OPPORTUNITY penalty:
        # If there existed a GREEN neighbor at the PREVIOUS node, but agent chose a slower edge,
        # penalize proportionally to extra time (encourages choosing green when available).
        if neighbors:
            prev_neighbor_times = [float(self.G[prev_node][n].get('weight', 0.6)) for n in neighbors]
            min_time = min(prev_neighbor_times)
            chosen_time = travel_time
            if min_time <= GREEN_MAX and chosen_time > min_time + 1e-9:
                missed_penalty_factor = 4.2  # tunable
                missed_penalty = (chosen_time - min_time) * missed_penalty_factor
                reward -= missed_penalty

        # small per-step penalty to encourage shorter routes
        reward -= 0.01

        # Dynamic traffic update (smooth perturbation) - only if enabled
        if self.random_traffic and (np.random.rand() < 0.15):
            edges = list(self.G.edges())
            if edges:
                u, v = edges[np.random.randint(len(edges))]
                new_w = self.G[u][v].get('weight', 0.6) + np.random.randn() * 0.08
                self.G[u][v]['weight'] = float(np.clip(new_w, 0.1, RED_MAX))

        # Usage update: traversed edge gets a small increment (short-term congestion)
        if (prev_node is not None) and (prev_node != self.ambulance_loc):
            u, v = prev_node, self.ambulance_loc
            inc = 0.06
            self.G[u][v]['weight'] = float(np.clip(self.G[u][v].get('weight', 0.6) + inc, 0.1, RED_MAX))

        # Global decay toward baseline to avoid runaway congestion
        baseline = 0.6
        decay_rate = 0.01
        for u, v, data in self.G.edges(data=True):
            w = float(data.get('weight', baseline))
            w = w + (baseline - w) * decay_rate
            self.G[u][v]['weight'] = float(np.clip(w, 0.1, RED_MAX))

        # expose last_node for trainers
        self.last_node = prev_node

        # info for visualization
        def time_to_color(t):
            if t <= GREEN_MAX:
                return 'green'
            if t <= YELLOW_MAX:
                return 'yellow'
            return 'red'

        info = {'chosen_time': float(travel_time), 'color': time_to_color(travel_time)}
        return self._get_obs(), float(reward), terminated, False, info

    def render(self):
        # simple matplotlib render: node colors for debugging
        color_map = []
        for node in self.G:
            if node == self.ambulance_loc:
                color_map.append('blue')
            elif node == self.patient_loc and not self.has_patient:
                color_map.append('red')
            elif node == self.hospital_loc:
                color_map.append('green')
            else:
                color_map.append('lightgray')

        pos = {n: (n % self.grid_size, n // self.grid_size) for n in self.G.nodes()}
        nx.draw(self.G, pos=pos, node_color=color_map, with_labels=True)
        plt.show()

if __name__ == "__main__":
    env = SmartCityEnv(grid_size=4)
    obs, _ = env.reset()
    print("Environment Created!")
    print(f"Initial Observation shape: {obs.shape}")

    total_reward = 0.0
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        print(f"step reward={reward:.3f} info={info}")
        if done:
            break

    print(f"Test Run Complete. Total Reward: {total_reward:.3f}")
