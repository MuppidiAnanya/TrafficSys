#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 28 22:17:37 2026

@author: muppidiananya
"""

import os
import numpy as np
import polars as pl
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import sumolib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
# %%
import torch

CSV_PATH = "/Users/muppidiananya/4-2Project/traffic_data.csv"
NET_PATH = "/Users/muppidiananya/4-2Project/gachibowli.net.xml"

SEQ_LEN = 12
BATCH_SIZE = 8
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %%

df = pl.read_csv(CSV_PATH)

df = df.rename({
    "time": "timestamp",
    "edge": "node_id",
    "mean_speed": "speed"
})

df = df.with_columns(pl.col("timestamp").cast(pl.Float64))

features = ["speed", "density", "vehicle_count"]

timestamps = df.select("timestamp").unique().sort("timestamp")["timestamp"]
nodes = df.select("node_id").unique().sort("node_id")["node_id"]

pivoted = []
for feat in features:
    p = (
        df.select(["timestamp", "node_id", feat])
        .pivot(values=feat, index="timestamp", columns="node_id")
        .sort("timestamp")
        .fill_null(0)
    )
    pivoted.append(p)

arrays = [p.to_numpy()[:, 1:] for p in pivoted]
arr = np.stack(arrays, axis=-1)  # (T, N, F)

print("Data shape:", arr.shape)

data = torch.tensor(arr, dtype=torch.float32)
# %%
T_total = data.shape[0]
train_T = int(0.7 * T_total)
val_T = int(0.2 * T_total)

train_data = data[:train_T]
val_data = data[train_T:train_T+val_T]
test_data = data[train_T+val_T:]

def create_sequences(data, N_input=SEQ_LEN, N_output=1):
    X, Y = [], []
    for i in range(len(data) - N_input - N_output):
        X.append(data[i:i+N_input])
        Y.append(data[i+N_input])
    return torch.stack(X), torch.stack(Y)

X_train, Y_train = create_sequences(train_data)
X_val, Y_val = create_sequences(val_data)
X_test, Y_test = create_sequences(test_data)

def prep_loader(X, Y):
    X = X.permute(0, 3, 1, 2)  # (samples, F, seq_len, nodes)
    return DataLoader(TensorDataset(X, Y), batch_size=BATCH_SIZE, shuffle=True)

train_loader = prep_loader(X_train, Y_train)
val_loader = prep_loader(X_val, Y_val)
test_loader = prep_loader(X_test, Y_test)

# %%


net = sumolib.net.readNet(NET_PATH)

edges = [e.getID() for e in net.getEdges()]
N = len(edges)

edge_to_idx = {e:i for i,e in enumerate(edges)}
A = np.zeros((N, N), dtype=np.float32)

for e1 in edges:
    if e1 in edge_to_idx:
        i = edge_to_idx[e1]
        for conn in net.getEdge(e1).getOutgoing():
            e2 = conn.getID()
            if e2 in edge_to_idx:
                j = edge_to_idx[e2]
                A[i, j] = 1

A += np.eye(N)
D = np.sum(A, axis=1)
D_inv_sqrt = np.diag(1.0 / np.sqrt(D + 1e-6))
A_norm = D_inv_sqrt @ A @ D_inv_sqrt
A_norm = torch.tensor(A_norm, dtype=torch.float32).to(DEVICE)

print("Adjacency shape:", A_norm.shape)
# %%

class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.temporal1 = nn.Conv2d(in_channels, out_channels, (1,3), padding=(0,1))
        self.temporal2 = nn.Conv2d(out_channels, out_channels, (1,3), padding=(0,1))

    def forward(self, x, adj):
        x = F.relu(self.temporal1(x))
        x = torch.einsum('bctn,nm->bctm', x, adj)
        x = F.relu(self.temporal2(x))
        return x

class STGCN(nn.Module):
    def __init__(self, num_nodes, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.block1 = STGCNBlock(in_channels, hidden_channels)
        self.block2 = STGCNBlock(hidden_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, adj):
        x = self.block1(x, adj)
        x = self.block2(x, adj)
        x = x.mean(dim=2)
        x = x.permute(0,2,1)
        return self.fc(x)

model = STGCN(
    num_nodes=N,
    in_channels=len(features),
    hidden_channels=32,
    out_channels=len(features)
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# %%



# CSV_PATH = "/Users/muppidiananya/4-2Project/traffic_data.csv"
# NET_PATH = "/Users/muppidiananya/4-2Project/gachibowli.net.xml"
MODEL_PATH = "/Users/muppidiananya/4-2Project/stgcn_traffic_model.pth"

SEQ_LEN = 12
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%




# LOAD MODEL


model = STGCN(
    num_nodes=N,
    in_channels=len(features),
    hidden_channels=64,   # <-- FIXED
    out_channels=len(features)
).to(DEVICE)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()

print("Model loaded successfully ✅")


# PREDICT NEXT TIMESTEP


last_seq = data[-SEQ_LEN:]                      # last 12 timesteps
last_seq = last_seq.unsqueeze(0)                # batch = 1
last_seq = last_seq.permute(0,3,1,2).to(DEVICE) # (1, F, seq_len, nodes)

with torch.no_grad():
    prediction = model(last_seq, A_norm)

prediction = prediction.cpu().numpy()[0]

print("Prediction shape:", prediction.shape)
print("First 5 edges predicted speed:")
print(prediction[:5, 0])
# %%

import pandas as pd

pred_df = pd.DataFrame(
    prediction,
    columns=["speed", "density", "vehicle_count"]
)

print(pred_df.head())
pred_df["edge_id"] = edges
pred_df = pred_df.set_index("edge_id")

print(pred_df.head())


import osmnx as ox
import pandas as pd
import numpy as np

# Load OSM
G = ox.graph_from_xml("/Users/muppidiananya/4-2Project/gachibowli.osm")
nodes_osm, edges_osm = ox.graph_to_gdfs(G)

# Convert list names to string
def clean_name(x):
    if isinstance(x, list):
        return x[0]  # take first name
    return x

edges_osm["name_clean"] = edges_osm["name"].apply(clean_name)

# Now safely extract unique names
street_names = edges_osm["name_clean"].dropna().unique()

print("Total unique street names:", len(street_names))

if len(street_names) < len(pred_df):
    street_names = np.resize(street_names, len(pred_df))

pred_df_display = pred_df.copy()
pred_df_display["street_name"] = street_names[:len(pred_df_display)]
pred_df_display = pred_df_display.set_index("street_name")

print(pred_df_display)


# %%


# FULL MODEL EVALUATION


# CREATE TEST LOADER (IF NOT EXISTS)


def create_sequences(data, N_input=SEQ_LEN, N_output=1):
    X, Y = [], []
    for i in range(len(data) - N_input - N_output):
        X.append(data[i:i+N_input])
        Y.append(data[i+N_input])
    return torch.stack(X), torch.stack(Y)

# Create test sequences again
X_test, Y_test = create_sequences(test_data)

# Prepare loader
X_test = X_test.permute(0, 3, 1, 2)  # (samples, F, seq_len, nodes)

test_loader = DataLoader(
    TensorDataset(X_test, Y_test),
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Test loader created ✅")
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

model.eval()

all_preds = []
all_true = []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(DEVICE)
        out = model(xb, A_norm)
        all_preds.append(out.cpu())
        all_true.append(yb)

all_preds = torch.cat(all_preds).numpy()
all_true = torch.cat(all_true).numpy()

features = ["speed", "density", "vehicle_count"]

print("\n ERROR METRICS")

for i, feat in enumerate(features):

    pred_flat = all_preds[:, :, i].reshape(-1)
    true_flat = all_true[:, :, i].reshape(-1)

    mae = mean_absolute_error(true_flat, pred_flat)
    mse = mean_squared_error(true_flat, pred_flat)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((true_flat - pred_flat) / (true_flat + 1e-5))) * 100

    print(f"\nFeature: {feat}")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAPE : {mape:.2f}%")
# %%
    

# LINE PLOT (First Test Sample)


sample_id = 0  # first test sample

pred_sample = all_preds[sample_id]
true_sample = all_true[sample_id]

for i, feat in enumerate(features):
    plt.figure(figsize=(10,4))
    plt.plot(true_sample[:, i], label="True")
    plt.plot(pred_sample[:, i], label="Predicted")
    plt.title(f"{feat} - True vs Predicted")
    plt.xlabel("Edge index")
    plt.ylabel(feat)
    plt.legend()
    # plt.show()
# %%
    

# SCATTER PLOT


for i, feat in enumerate(features):
    plt.figure(figsize=(5,5))
    
    plt.scatter(
        all_true[:,:,i].reshape(-1),
        all_preds[:,:,i].reshape(-1),
        alpha=0.3
    )

    min_val = all_true[:,:,i].min()
    max_val = all_true[:,:,i].max()

    plt.plot([min_val, max_val],
             [min_val, max_val],
             'r--')

    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(f"{feat} Scatter Plot")
    # plt.show()
# %%
    

# ERROR DISTRIBUTION


for i, feat in enumerate(features):
    errors = all_true[:,:,i].reshape(-1) - all_preds[:,:,i].reshape(-1)
    
    plt.figure(figsize=(8,4))
    plt.hist(errors, bins=50)
    plt.title(f"{feat} Error Distribution")
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    # plt.show()
# %%
    
import pandas as pd

def generate_traffic_summary(pred_df):
    summary = []

    for edge, row in pred_df.iterrows():
        summary.append(
            f"Edge {edge}: Speed={row['speed']:.2f}, "
            f"Density={row['density']:.2f}, "
            f"Vehicles={row['vehicle_count']:.0f}"
        )

    return "\n".join(summary[:50])  # limit to 50 edges for prompt size


traffic_summary = generate_traffic_summary(pred_df)
# %%

city_name = "Gachibowli, Hyderabad"
import os
import requests
from dotenv import load_dotenv

load_dotenv()
# Set Groq API key
API_KEY = os.getenv("GROQ_API_KEY")
if API_KEY:
    os.environ["GROQ_API_KEY"] = API_KEY

url = "https://api.groq.com/openai/v1/chat/completions"

def get_groq_headers():
    return {
        "Authorization": f"Bearer {os.environ.get('GROQ_API_KEY', '')}",
        "Content-Type": "application/json"
    }

system_prompt = f"""
You are a city traffic planning assistant for {city_name}.
You have access to predicted traffic data (speed, density, vehicle count).

Your goals:
- Suggest infrastructure improvements like flyovers, underpasses, signals, road widening.
- Recommend signal timing optimization.
- Suggest congestion mitigation strategies.
- Mention areas using road names if available.
- Be concise and practical.
- Treat traffic data as predicted future conditions.
- Split answers into multiple lines.
"""

chat_history = [
    {"role": "system", "content": system_prompt}
]

def get_road_prediction(user_input):

    user_input = user_input.lower()

    for road in pred_df_display.index:

        if road.lower() in user_input:

            row = pred_df_display.loc[road]

            # if multiple rows exist → take first one
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]

            result = f"""
Road: {road}
Predicted Speed: {float(row['speed']):.2f}
Predicted Density: {float(row['density']):.4f}
Predicted Vehicle Count: {float(row['vehicle_count']):.2f}
"""

            return result

    return None
def get_route_advice(user_input):

    text = user_input.lower()

    if "from" in text and "to" in text:

        try:
            start = text.split("from")[1].split("to")[0].strip()
            end = text.split("to")[1].strip()

            return f"""
Suggested travel from {start} to {end}:

1. Start at {start}
2. Move towards the nearest arterial road connecting to Gachibowli corridor
3. Continue toward {end}

Traffic insight:
• If predicted speed is low on intermediate roads, consider alternate parallel streets
• Watch for congestion near Hi Tech City junction

Recommendation:
• Use signalized intersections instead of smaller internal streets
• Avoid peak-density segments if possible
"""
        except:
            return None

    return None
def interact_with_assistant(user_input):
    route_result = get_route_advice(user_input)
    if route_result is not None:
        return route_result
        
    road_result = get_road_prediction(user_input)
    if road_result is not None:
        return road_result
        
    user_message = f"""
Predicted Traffic Data Summary:

{traffic_summary}

User Question:
{user_input}
"""

    chat_history.append({"role": "user", "content": user_message})

    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": chat_history,
        "temperature": 0.6,
        "max_tokens": 600
    }

    try:
        response = requests.post(url, headers=get_groq_headers(), json=payload)
        if response.status_code == 200:
            result = response.json()
            bot_reply = result['choices'][0]['message']['content']
            chat_history.append({"role": "assistant", "content": bot_reply})
            return bot_reply
        else:
            return f"Error: {response.status_code} {response.text}"
    except Exception as e:
        return f"Request failed: {str(e)}"

if __name__ == "__main__":
    print("\n🚦 Traffic Chatbot is online. Type 'exit' to quit.\n")
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Chat ended.")
            break
        reply = interact_with_assistant(user_input)
        print("\n🚦 Traffic Assistant:\n")
        print(reply)
        print()
# %%
    
