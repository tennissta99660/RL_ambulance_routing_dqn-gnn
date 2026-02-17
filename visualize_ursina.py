# visualize.py
import torch
import numpy as np
from ursina import *
from environment import SmartCityEnv
from gnn import NeuroRouteGNN, get_graph_state, DEVICE
from torch_geometric.data import Data


# --- USER CONFIG ---
GRID_SIZE = 4
MODEL_PATH = "neuroroute_model_batched_final.pth"
HIDDEN_DIM = 64  # must match training
SOFTMAX_TEMPERATURE = 0.6  # <1 => sharper; >1 => smoother/random. Tune to reduce oscillations.

# --- SETUP ---
app = Ursina(vsync=True)
device = torch.device("cpu")  # visualize on CPU to avoid GPU transfer latency
print(f"Loading model on: {device}")

env = SmartCityEnv(grid_size=GRID_SIZE)
model = NeuroRouteGNN(4, hidden_dim=HIDDEN_DIM).to(device)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("Model loaded successfully!")
except Exception as e:
    print(f"ERROR: {e}\nUsing untrained random model.")

model.eval()

# --- BUILD CITY ---
ground = Entity(model='plane', scale=(20, 1, 20), color=color.rgb(30,30,30), texture='white_cube')

node_entities = []
for i in range(GRID_SIZE * GRID_SIZE):
    x = i % GRID_SIZE
    z = i // GRID_SIZE
    node_entities.append(Entity(model='cube', position=(x*2, 0, z*2), scale=(0.5, 0.1, 0.5), color=color.dark_gray))

road_entities = {}
for u, v in env.G.edges():
    ux, uz = u % GRID_SIZE, u // GRID_SIZE
    vx, vz = v % GRID_SIZE, v // GRID_SIZE
    mid_x = (ux*2 + vx*2) / 2
    mid_z = (uz*2 + vz*2) / 2
    scale = (2, 0.05, 0.2) if ux != vx else (0.2, 0.05, 2)
    road = Entity(model='cube', position=(mid_x, 0, mid_z), scale=scale, color=color.black)
    edge_key = tuple(sorted((u, v)))
    road_entities[edge_key] = road

ambulance = Entity(model='cube', color=color.orange, scale=(0.8, 0.5, 1.2), position=(0, 0.25, 0))
Entity(parent=ambulance, model='cube', color=color.cyan, scale=(0.8, 0.2, 0.2), position=(0, 0.6, 0.3))
Entity(parent=ambulance, model='cube', color=color.red, scale=(0.8, 0.2, 0.2), position=(0, 0.6, -0.3))

patient_marker = Entity(model='sphere', color=color.red, scale=0.8)
hospital_marker = Entity(model='cube', color=color.white, scale=1.2)
hospital_text = Text(text='HOSPITAL', parent=hospital_marker, y=1.5, scale=5, color=color.white, billboard=True)

def weight_to_color(weight, vmin=0.1, vmax=2.0):
    # map weight (travel time) to RGB using gradient: green (fast) -> yellow -> red (slow)
    t = float((weight - vmin) / max(1e-6, (vmax - vmin)))
    t = max(0.0, min(1.0, t))
    if t <= 0.5:
        tt = t / 0.5
        r = int(255 * tt); g = 255; b = 0
    else:
        tt = (t - 0.5) / 0.5
        r = 255; g = int(255 * (1 - tt)); b = 0
    return color.rgb(r, g, b)

def update_traffic_visuals():
    weights = [d['weight'] for _,_,d in env.G.edges(data=True)]
    if not weights: return
    vmin, vmax = float(np.min(weights)), float(np.max(weights))
    if vmin == vmax:
        vmax = vmin + 1.0
    for u, v, data in env.G.edges(data=True):
        edge_key = tuple(sorted((u, v)))
        if edge_key in road_entities:
            road_entities[edge_key].color = weight_to_color(data['weight'], vmin=vmin, vmax=vmax)

# INITIALIZE positions
obs, _ = env.reset()
px, pz = env.patient_loc % GRID_SIZE, env.patient_loc // GRID_SIZE
patient_marker.position = (px*2, 0.5, pz*2)
hx, hz = env.hospital_loc % GRID_SIZE, env.hospital_loc // GRID_SIZE
hospital_marker.position = (hx*2, 0.5, hz*2)
ambulance.position = ((env.ambulance_loc % GRID_SIZE)*2, 0.25, (env.ambulance_loc // GRID_SIZE)*2)
update_traffic_visuals()

# CAMERA
CAM_OFFSET = Vec3(0, 10, -14)
CAM_SMOOTH = 4.0
camera.position = ambulance.position + CAM_OFFSET
camera.rotation_x = 35
camera.look_at(ambulance.position+Vec3(0, 0.5, 0))

def update():
    target_cam = ambulance.position + CAM_OFFSET
    camera.position = lerp(camera.position, target_cam, time.dt * CAM_SMOOTH)
    camera.look_at(ambulance.position)

def softmax_with_temperature(q_array, temp=1.0):
    q = np.array(q_array, dtype=np.float64)
    if temp <= 0:
        # treat as argmax
        probs = np.zeros_like(q)
        probs[np.argmax(q)] = 1.0
        return probs
    q = q / float(temp)
    q = q - np.max(q)
    e = np.exp(q)
    probs = e / (np.sum(e) + 1e-12)
    return probs

# AI loop using softmax temperature for action selection (reduces oscillation)
def ai_step():
    try:
        if env.has_patient and env.ambulance_loc == env.hospital_loc:
            print(">>> DELIVERED! Resetting...")
            env.reset()
            px, pz = env.patient_loc % GRID_SIZE, env.patient_loc // GRID_SIZE
            patient_marker.position = (px*2, 0.5, pz*2)
            patient_marker.visible = True
            hx, hz = env.hospital_loc % GRID_SIZE, env.hospital_loc // GRID_SIZE
            hospital_marker.position = (hx*2, 0.5, hz*2)
            ambulance.position = ((env.ambulance_loc % GRID_SIZE)*2, 0.25, (env.ambulance_loc // GRID_SIZE)*2)
            ambulance.color = color.orange
            update_traffic_visuals()
            invoke(ai_step, delay=2.0)
            return

        graph_data = get_graph_state(env)
        curr_node = env.ambulance_loc
        neighbors = list(env.G.neighbors(curr_node))
        if not neighbors:
            invoke(ai_step, delay=0.6)
            return

        q_values = []
        with torch.no_grad():
            d = Data(x=graph_data.x, edge_index=graph_data.edge_index, edge_attr=graph_data.edge_attr)
            for neighbor in neighbors:
                q = model(d, curr_node, neighbor)
                q_values.append(q.item())

        # apply softmax with temperature to produce smoother distribution
        probs = softmax_with_temperature(q_values, temp=SOFTMAX_TEMPERATURE)
        # pick according to probabilities (or argmax if that's desired)
        if np.isnan(probs).any() or probs.sum() == 0:
            chosen_idx = int(np.argmax(q_values))
        else:
            chosen_idx = int(np.random.choice(len(neighbors), p=probs))

        chosen_neighbor = neighbors[chosen_idx]
        all_possible = list(env.G.neighbors(curr_node))
        action = all_possible.index(chosen_neighbor)

        _, _, done, _, _ = env.step(action)

        # Update visuals
        update_traffic_visuals()
        target_x = (env.ambulance_loc % GRID_SIZE) * 2
        target_z = (env.ambulance_loc // GRID_SIZE) * 2
        ambulance.look_at((target_x, 0.25, target_z))
        ambulance.animate_position((target_x, 0.25, target_z), duration=0.4, curve=curve.in_out_quad)

        if env.has_patient:
            patient_marker.visible = False
            ambulance.color = color.cyan
        else:
            ambulance.color = color.orange

        invoke(ai_step, delay=0.6)

    except Exception as e:
        print(f"CRITICAL ERROR IN AI STEP: {e}")
        invoke(ai_step, delay=1.0)

invoke(ai_step, delay=1.0)
app.run()

