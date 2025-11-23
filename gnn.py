# gnn.py — updated, full file

"""
Batched GNN training utilities for the SmartCityEnv.

Provides:
- NeuroRouteGNN: GCN -> pairwise MLP producing Q(current->neighbor)
- get_graph_state(env): builds torch_geometric.data.Data from env
- ReplayBuffer: lightweight CPU-buffer for training
- train_batched(...): example DQN-style batched trainer

Keep env.random_traffic = False during training for stability.
"""

import random
import copy
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv

from f1 import SmartCityEnv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NeuroRouteGNN(nn.Module):
    """
    GCN that returns node embeddings; then an MLP maps [h_src, h_dst, edge_attr] -> Q.
    Provides vectorized helpers for batched training.
    """

    def __init__(self, num_node_features=4, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        # head receives src_emb + dst_emb + edge_attr(1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward_embeddings(self, batch: Batch):
        """
        Run GCN on a PyG Batch and return node embeddings h (total_nodes x hidden_dim).
        Safe if batch has no edges.
        """
        device = batch.x.device
        x = batch.x.to(device)
        ei = batch.edge_index.to(device) if hasattr(batch, "edge_index") else None

        if ei is None or ei.numel() == 0:
            empty_ei = torch.empty((2, 0), dtype=torch.long, device=device)
            h = F.relu(self.conv1(x, empty_ei))
            h = F.relu(self.conv2(h, empty_ei))
        else:
            if hasattr(batch, "edge_attr") and batch.edge_attr is not None and batch.edge_attr.numel() > 0:
                ea = batch.edge_attr.to(device)
                h = F.relu(self.conv1(x, ei, edge_weight=ea))
                h = F.relu(self.conv2(h, ei, edge_weight=ea))
            else:
                h = F.relu(self.conv1(x, ei))
                h = F.relu(self.conv2(h, ei))
        return h
    def forward(self, data: Data, current_node: int, potential_next_node: int):
        """
        Single-sample Q computation for visualization / inference.
        Returns scalar tensor Q(current_node -> potential_next_node).
        """
        batch = Batch.from_data_list([data]).to(data.x.device)
        h = self.forward_embeddings(batch)
        
        src = h[current_node].unsqueeze(0)
        dst = h[potential_next_node].unsqueeze(0)
        
        # find edge_attr if present
        edge_val = torch.tensor([0.0], device=h.device, dtype=h.dtype)
        if hasattr(data, "edge_index") and data.edge_index.numel() > 0 and hasattr(data, "edge_attr"):
            ei = data.edge_index
            ea = data.edge_attr
            mask = (ei[0] == current_node) & (ei[1] == potential_next_node)
            if mask.any():
                idx = mask.nonzero(as_tuple=True)[0][0]
                edge_val = ea[idx].unsqueeze(0).to(h.device)
            else:
                mask = (ei[0] == potential_next_node) & (ei[1] == current_node)
                if mask.any():
                    idx = mask.nonzero(as_tuple=True)[0][0]
                    edge_val = ea[idx].unsqueeze(0).to(h.device)
        
        feat = torch.cat([src, dst, edge_val.to(src.dtype).unsqueeze(0)], dim=-1)
        q = self.fc(feat)
        return q.view(-1)

    def q_from_embeddings_batched(self, h: torch.Tensor, ptr: torch.Tensor, graph_idx: torch.Tensor,
                                  curr_local: torch.Tensor, dst_local: torch.Tensor, edge_attr_vals: torch.Tensor):
        """
        Vectorized Q computation for many (graph_idx, curr_local, dst_local) pairs.
        - h: node embeddings [total_nodes, hidden_dim]
        - ptr: LongTensor [batch_size+1] with start offsets (Batch.ptr)
        - graph_idx/curr_local/dst_local: LongTensors length N_pairs
        - edge_attr_vals: FloatTensor length N_pairs

        Returns: q tensor length N_pairs
        """
        device = h.device
        ptr = ptr.to(device)
        graph_idx = graph_idx.to(device)
        curr_local = curr_local.to(device)
        dst_local = dst_local.to(device)
        edge_attr_vals = edge_attr_vals.to(device).unsqueeze(1)  # [N,1]

        start_offsets = ptr[graph_idx]  # [N]
        global_curr = start_offsets + curr_local
        global_dst = start_offsets + dst_local

        src_h = h[global_curr]  # [N, hidden]
        dst_h = h[global_dst]   # [N, hidden]

        feat = torch.cat([src_h, dst_h, edge_attr_vals], dim=-1)  # [N, 2*hidden+1]
        q = self.fc(feat).view(-1)
        return q


def get_graph_state(env: SmartCityEnv):
    """
    Convert env graph to torch_geometric.data.Data.
    Node features: [is_ambulance, is_patient(if not picked), is_hospital, is_last_node]
    Edge attr: normalized inverse travel_time in [0,1] (higher==faster/greener)
    """
    node_features = torch.zeros((env.num_nodes, 4), dtype=torch.float32)
    node_features[env.ambulance_loc, 0] = 1.0
    if not env.has_patient:
        node_features[env.patient_loc, 1] = 1.0
    node_features[env.hospital_loc, 2] = 1.0
    if getattr(env, "last_node", None) is not None:
        node_features[env.last_node, 3] = 1.0

    edges = []
    weights = []
    for u, v, data in env.G.edges(data=True):
        edges.append((u, v))
        weights.append(float(data.get("weight", 1.0)))

    if len(weights) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0,), dtype=torch.float32)
        return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

    invs = np.array([1.0 / max(w, 1e-6) for w in weights], dtype=np.float32)
    # normalize to [0,1] more aggressively: scale by time thresholds
    # GREEN (0.5s) -> 1.0, RED (2.0s) -> 0.0, linear in between
    normalized = np.zeros_like(invs)
    for i, w in enumerate(weights):
        if w <= 0.5:
            normalized[i] = 1.0  # green
        elif w <= 2.0:
            normalized[i] = (2.0 - w) / 1.5  # linear decay
        else:
            normalized[i] = 0.0  # red

    edge_index = []
    edge_attr_vals = []
    for i, (u, v) in enumerate(edges):
        val = float(normalized[i])
        edge_index.append([u, v])
        edge_attr_vals.append(val)
        edge_index.append([v, u])
        edge_attr_vals.append(val)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr_vals, dtype=torch.float32)

    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

class ReplayBuffer:
    """
    Minimal replay buffer storing copies of Data objects + metadata on CPU.
    """

    def __init__(self, capacity=50000):
        self.buf = deque(maxlen=capacity)

    def push(self, pre_data: Data, curr_idx: int, chosen_dst_idx: int, chosen_edge_attr: float,
             reward: float, next_data: Data, next_node_idx: int, next_neighbors: list, next_edge_attrs: list, done: bool):
        # deep-copy to avoid mutation and ensure CPU storage
        pre_data_c = copy.deepcopy(pre_data)
        next_data_c = copy.deepcopy(next_data)

        # keep on cpu
        pre_data_c.x = pre_data_c.x.cpu()
        pre_data_c.edge_index = pre_data_c.edge_index.cpu() if getattr(pre_data_c, "edge_index", None) is not None else pre_data_c.edge_index
        pre_data_c.edge_attr = pre_data_c.edge_attr.cpu() if getattr(pre_data_c, "edge_attr", None) is not None else pre_data_c.edge_attr

        next_data_c.x = next_data_c.x.cpu()
        next_data_c.edge_index = next_data_c.edge_index.cpu() if getattr(next_data_c, "edge_index", None) is not None else next_data_c.edge_index
        next_data_c.edge_attr = next_data_c.edge_attr.cpu() if getattr(next_data_c, "edge_attr", None) is not None else next_data_c.edge_attr

        self.buf.append({
            "pre_data": pre_data_c,
            "curr_idx": int(curr_idx),
            "chosen_dst_idx": int(chosen_dst_idx),
            "chosen_edge_attr": float(chosen_edge_attr),
            "reward": float(reward),
            "next_data": next_data_c,
            "next_node_idx": int(next_node_idx),
            "next_neighbors": list(map(int, next_neighbors)),
            "next_edge_attrs": list(map(float, next_edge_attrs)),
            "done": bool(done)
        })

    def sample(self, batch_size):
        return random.sample(self.buf, batch_size)

    def __len__(self):
        return len(self.buf)


def find_edge_attr_in_data(data: Data, src_local: int, dst_local: int):
    """
    Returns edge_attr (float) for directed edge src_local -> dst_local if present, else 0.0.
    Works on CPU Data objects (uses numpy safe path).
    """
    if getattr(data, "edge_index", None) is None or data.edge_index.numel() == 0:
        return 0.0
    if getattr(data, "edge_attr", None) is None or data.edge_attr.numel() == 0:
        return 0.0
    ei = data.edge_index.numpy()
    ea = data.edge_attr.numpy()
    mask = np.where((ei[0] == src_local) & (ei[1] == dst_local))[0]
    if mask.size > 0:
        return float(ea[mask[0]])
    return 0.0


def train_batched(
    episodes=2000,
    steps_per_episode=600,
    lr=1e-4,
    gamma=0.95,
    batch_size=64,
    replay_capacity=50000,
    target_update_steps=2000,
    hidden_dim=64,
    seed=42,
    min_replay_size=512
):
    """
    Example batched training loop. Intended as a starting point — tune hyperparams,
    implement more robust replay sampling priorities, and extend logging as needed.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env = SmartCityEnv(grid_size=4)
    env.random_traffic = False

    model = NeuroRouteGNN(num_node_features=4, hidden_dim=hidden_dim).to(DEVICE)
    target = NeuroRouteGNN(num_node_features=4, hidden_dim=hidden_dim).to(DEVICE)
    target.load_state_dict(model.state_dict())
    target.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    replay = ReplayBuffer(capacity=replay_capacity)

    epsilon = 0.95
    epsilon_min = 0.05
    eps_decay = 0.995

    total_steps = 0

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        env.last_node = None
        total_reward = 0.0
        done = False
        step_cnt = 0

        while not done and step_cnt < steps_per_episode:
            step_cnt += 1
            total_steps += 1

            pre_data = get_graph_state(env)
            curr = env.ambulance_loc
            neighbors = list(env.G.neighbors(curr))
            if not neighbors:
                break

            # epsilon-greedy choice using single GCN forward per decision
            if random.random() < epsilon:
                chosen_dst = random.choice(neighbors)
            else:
                # batchify pre_data and run embeddings
                batch = Batch.from_data_list([pre_data]).to(DEVICE)
                with torch.no_grad():
                    h = model.forward_embeddings(batch)
                    ptr = batch.ptr  # device tensor
                    graph_idx = torch.zeros(len(neighbors), dtype=torch.long, device=DEVICE)
                    curr_local = torch.tensor([curr] * len(neighbors), dtype=torch.long, device=DEVICE)
                    dst_local = torch.tensor(neighbors, dtype=torch.long, device=DEVICE)
                    edge_attrs_list = [find_edge_attr_in_data(pre_data, curr, n) for n in neighbors]
                    edge_attrs = torch.tensor(edge_attrs_list, dtype=torch.float32, device=DEVICE)
                    qvals = model.q_from_embeddings_batched(h, ptr, graph_idx, curr_local, dst_local, edge_attrs)
                    chosen_idx = int(torch.argmax(qvals).item())
                    chosen_dst = neighbors[chosen_idx]

            action_idx = neighbors.index(chosen_dst)
            prev_node = env.ambulance_loc
            _, reward, done, _, _ = env.step(action_idx)
            total_reward += reward

            next_data = get_graph_state(env)
            next_node_idx = env.ambulance_loc
            next_neighbors = list(env.G.neighbors(next_node_idx))
            next_edge_attrs = [find_edge_attr_in_data(next_data, next_node_idx, n) for n in next_neighbors]
            chosen_edge_attr = find_edge_attr_in_data(pre_data, curr, chosen_dst)

            replay.push(pre_data, curr, chosen_dst, chosen_edge_attr, reward,
                        next_data, next_node_idx, next_neighbors, next_edge_attrs, done)

            env.last_node = prev_node

            # learning step
            if len(replay) >= max(min_replay_size, batch_size):
                minibatch = replay.sample(batch_size)

                pre_list = [itm["pre_data"] for itm in minibatch]
                next_list = [itm["next_data"] for itm in minibatch]

                pre_batch = Batch.from_data_list(pre_list).to(DEVICE)
                next_batch = Batch.from_data_list(next_list).to(DEVICE)

                h_pre = model.forward_embeddings(pre_batch)
                h_next = target.forward_embeddings(next_batch)

                ptr_pre = pre_batch.ptr
                ptr_next = next_batch.ptr

                # prepare tensors for current chosen actions
                graph_idx_curr = []
                curr_local_idx = []
                dst_local_idx = []
                chosen_edge_attrs = []
                rewards = []
                dones = []

                next_graph_idx_list = []
                next_src_local_list = []
                next_dst_local_list = []
                next_edge_attr_list = []

                for i, itm in enumerate(minibatch):
                    graph_idx_curr.append(i)
                    curr_local_idx.append(int(itm["curr_idx"]))
                    dst_local_idx.append(int(itm["chosen_dst_idx"]))
                    chosen_edge_attrs.append(float(itm["chosen_edge_attr"]))
                    rewards.append(float(itm["reward"]))
                    dones.append(bool(itm["done"]))

                    neighs = itm["next_neighbors"]
                    neigh_attrs = itm["next_edge_attrs"]
                    if len(neighs) > 0:
                        for j, nb in enumerate(neighs):
                            next_graph_idx_list.append(i)
                            next_src_local_list.append(int(itm["next_node_idx"]))
                            next_dst_local_list.append(int(nb))
                            next_edge_attr_list.append(float(neigh_attrs[j]))

                graph_idx_curr = torch.tensor(graph_idx_curr, dtype=torch.long, device=DEVICE)
                curr_local_idx = torch.tensor(curr_local_idx, dtype=torch.long, device=DEVICE)
                dst_local_idx = torch.tensor(dst_local_idx, dtype=torch.long, device=DEVICE)
                chosen_edge_attrs = torch.tensor(chosen_edge_attrs, dtype=torch.float32, device=DEVICE)
                rewards = torch.tensor(rewards, dtype=torch.float32, device=DEVICE)
                dones = torch.tensor(dones, dtype=torch.bool, device=DEVICE)

                q_curr = model.q_from_embeddings_batched(h_pre, ptr_pre, graph_idx_curr, curr_local_idx, dst_local_idx, chosen_edge_attrs)

                # compute max next q per sample (flattened -> aggregate)
                if len(next_graph_idx_list) > 0:
                    next_graph_idx = torch.tensor(next_graph_idx_list, dtype=torch.long, device=DEVICE)
                    next_src_local = torch.tensor(next_src_local_list, dtype=torch.long, device=DEVICE)
                    next_dst_local = torch.tensor(next_dst_local_list, dtype=torch.long, device=DEVICE)
                    next_edge_attr_vals = torch.tensor(next_edge_attr_list, dtype=torch.float32, device=DEVICE)

                    q_next_flat = target.q_from_embeddings_batched(h_next, ptr_next, next_graph_idx, next_src_local, next_dst_local, next_edge_attr_vals)

                    # aggregate max per minibatch sample (fallback to python loop — batch_size moderate)
                    num_samples = len(minibatch)
                    max_next_qs = torch.zeros(num_samples, dtype=torch.float32, device=DEVICE)
                    # default 0 for samples with no neighbors
                    max_next_qs[:] = 0.0
                    for i in range(num_samples):
                        mask = (next_graph_idx == i)
                        if mask.any():
                            max_next_qs[i] = q_next_flat[mask].max()
                else:
                    max_next_qs = torch.zeros(len(minibatch), dtype=torch.float32, device=DEVICE)

                target_q = rewards + (gamma * max_next_qs) * (~dones)

                model.train()
                loss = loss_fn(q_curr, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if total_steps % target_update_steps == 0:
                    target.load_state_dict(model.state_dict())

        epsilon = max(epsilon_min, epsilon * eps_decay)
        print(f"[EP {ep}/{episodes}] reward={total_reward:.2f} eps={epsilon:.3f} replay={len(replay)}")

        if ep % 100 == 0:
            torch.save(model.state_dict(), f"neuroroute_model_batched_ep{ep}.pth")

    torch.save(model.state_dict(), "neuroroute_model_batched_final.pth")
    print("Training complete. Model saved to 'neuroroute_model_batched_final.pth'")
    return model


if __name__ == "__main__":
    trained = train_batched(
        episodes=5000,
        steps_per_episode=600,
        lr=5e-5,
        gamma=0.95,
        batch_size=64,
        replay_capacity=50000,
        target_update_steps=1000,
        hidden_dim=64,
        seed=42,
        min_replay_size=512
    )
