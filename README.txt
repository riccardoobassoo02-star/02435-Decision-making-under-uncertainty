# SP Policy — Code Walkthrough

## Code Structure

The code has four main sections:

1. **Parameter extraction** — loads system constants at module level
2. **`build_tree()`** — constructs the scenario tree using Branch & Cluster
3. **`solve_sp()`** — builds and solves the MILP on the scenario tree
4. **`select_action()`** — entry point called by the environment

---

## 1. Parameter Extraction

```python
data = get_fixed_data()
P_max, zeta_exch, zeta_conv, zeta_loss, zeta_cool, zeta_occ, ...
```

All physical system constants are loaded once at module level. These never change between calls. The initial conditions (T1, T2, H) are **not** loaded here — they come from the environment state at runtime.

`T_out` is the outdoor temperature array, one value per hour. Unlike price and occupancy, outdoor temperature is **deterministic and known in advance** — no scenarios needed.

`M = 1000` is the big-M constant used in all linearization constraints.

---

## 2. Scenario Tree Builder — `build_tree()`

### Purpose

Builds a scenario tree using the **Repeated Branch & Cluster** method. Takes the current state and generates a tree of possible future realizations for price and occupancy.

### Inputs

| Parameter | Description |
|-----------|-------------|
| `state` | Current state dictionary from the environment |
| `L` | Lookahead horizon — number of future stages beyond hour *t* |
| `B` | Branching factor — number of children per node |
| `N_samples` | Monte Carlo samples generated per node before clustering (default: 100) |

### Output

A list of node dictionaries representing the full scenario tree.

### Root Node (tau=0)

```python
root = {
    "id":         0,          # unique identifier
    "tau":        0,          # depth level in the tree (0 = here-and-now)
    "parent_id":  None,       # root has no parent
    "price":      ...,        # realized price at current hour
    "price_prev": ...,        # previous price (needed for AR process)
    "occ1":       ...,        # current occupancy room 1
    "occ2":       ...,        # current occupancy room 2
    "prob":       1.0         # certainty — known present
}
```

Everything is known with certainty (prob=1.0). Both `price` and `price_prev` are stored because the price process is autoregressive — it needs two lags to simulate the next price.

### BFS Loop

The tree is built level by level using a **breadth-first search (BFS)** queue:

- `queue.pop(0)` takes from the front (FIFO — first in, first out)
- `queue.append(child)` adds to the back
- This ensures all tau=1 nodes are created before any tau=2 node

### Stopping Condition

```python
if parent["tau"] >= H:
    continue
```

If a node has reached the maximum depth, it's a leaf — skip it. `continue` moves to the next node in the queue (unlike `break`, which would exit the entire loop).

### Three Phases Per Node

**Phase 1 — Branch (Monte Carlo):** Generate `N_samples=100` random next-step realizations conditioned on this parent's price and occupancy. Each sample is an independent draw from the stochastic process.

**Phase 2 — Cluster (K-means):** Reduce 100 samples to `B` representative scenarios. The feature matrix has shape (100, 3) with columns [price, occ1, occ2]. K-means finds B clusters and returns centroids (the mean of each cluster).

**Phase 3 — Create Children:** For each cluster, create one child node. The probability is computed as:

```
child_prob = parent_prob × (samples_in_cluster / N_samples)
```

This is the **chain rule** — conditional probability times parent probability. The child's `price_prev` is set to the parent's `price` to maintain the autoregressive chain.

### Node ID vs Tau

- **`id`**: unique identifier — every node has a different one. Used to index Pyomo variables and track parent-child relationships.
- **`tau`**: depth level — many nodes share the same tau. Used only for the depth check and for computing `t_parent` in constraints.

### Tree Size

With H=3 and B=2: 1 (root) + 2 + 4 + 8 = 15 nodes total. With H=3 and B=3: 1 + 3 + 9 + 27 = 40 nodes.

---

## 3. SP MILP Solver — `solve_sp()`

### Purpose

Builds and solves the multi-stage SP MILP on the scenario tree. Returns only the here-and-now decisions for tau=0.

---

### 3.1 Setup

```python
node_by_id   = {n["id"]: n for n in nodes}
nodes_future = [n for n in nodes if n["tau"] >= 1]
```

- **`node_by_id`**: dictionary mapping id → node dictionary. Allows instant lookup when walking ancestor chains. It's a "mega dictionary" — a dictionary of dictionaries.
- **`nodes_future`**: filtered list excluding the root. The root has separate variables (p0, v0, s0) and is not part of `model.NODES`.

```python
low_override_init = {1: state["low_override_r1"], 2: state["low_override_r2"]}
```

Maps each room to its current overrule status. Used by `u_par()` when a tau=1 node needs the parent's overrule state.

```python
vent_counter     = state["vent_counter"]
remaining_forced = max(0, min_up_time - vent_counter) if vent_counter > 0 else 0
v_prev           = 1 if vent_counter > 0 else 0
```

- `vent_counter`: consecutive hours ventilation has been ON
- `remaining_forced`: how many more hours it must stay ON (e.g., vent_counter=1, min_up_time=3 → remaining_forced=2)
- `v_prev`: whether ventilation was ON in the previous hour (for startup detection)

---

### 3.2 Sets

```python
model.R     = RangeSet(1, 2)                                      # rooms {1, 2}
model.NODES = Set(initialize=[n["id"] for n in nodes_future])     # future node IDs
```

`model.NODES` tells Pyomo which indices are valid for future variables. When you write `model.v = Var(model.NODES, ...)`, Pyomo creates one variable per element in the set.

---

### 3.3 Variables

**Here-and-now (tau=0)** — single decision, not indexed by scenario:

| Variable | Description |
|----------|-------------|
| `model.p0[r]` | Heating power per room |
| `model.v0` | Ventilation on/off |
| `model.s0` | Ventilation startup indicator (1 if turning ON now) |

**Future nodes (tau≥1)** — one variable per node:

| Variable | Description |
|----------|-------------|
| `model.p[r, nid]` | Heating power per room per node |
| `model.v[nid]` | Ventilation on/off per node |
| `model.s[nid]` | Ventilation startup indicator per node |
| `model.temp[r, nid]` | Temperature per room per node (state variable, not a decision) |
| `model.hum[nid]` | Humidity per node (state variable, not a decision) |
| `model.y_low[r, nid]` | 1 if temp < T_low (detection variable) |
| `model.y_ok[r, nid]` | 1 if temp > T_ok (detection variable) |
| `model.u[r, nid]` | 1 if low-temp overrule active (has memory/hysteresis) |
| `model.y_high[r, nid]` | 1 if temp ≥ T_high (detection variable, no memory) |

---

### 3.4 Helper Functions

Every constraint links a node to its parent. The parent's values come from **two different sources** depending on whether the parent is the root or not. The helpers hide this distinction.

**Pattern:** if `node["tau"] == 1`, the parent is the root → return here-and-now variable or state constant. Otherwise → return Pyomo variable of the parent node.

| Helper | tau=1 returns | tau≥2 returns |
|--------|--------------|---------------|
| `v_par(n)` | `model.v0` (decision variable) | `model.v[parent_id]` |
| `p_par(r, n)` | `model.p0[r]` (decision variable) | `model.p[r, parent_id]` |
| `temp_par(r, n)` | `state["T1"]` or `state["T2"]` (constant) | `model.temp[r, parent_id]` |
| `temp_other_par(r, n)` | temperature of the OTHER room (constant) | `model.temp[other_r, parent_id]` |
| `hum_par(n)` | `state["H"]` (constant) | `model.hum[parent_id]` |
| `occ_par(r, n)` | `state["Occ1"]` or `state["Occ2"]` (constant) | centroid value from tree (always constant) |
| `u_par(r, n)` | `int(low_override_init[r])` (constant) | `model.u[r, parent_id]` |

Note: `occ_par` is special — occupancy is **never** a Pyomo variable. It's always a known parameter, either from the state or from the scenario tree data.

Note: `int()` in `u_par` converts boolean True/False from the environment to 0/1 for Pyomo compatibility.

---

### 3.5 Objective Function

```python
obj = here_and_now_cost + expected_future_cost
```

**Here-and-now cost:** `price_t × (p0[1] + p0[2] + P_vent × v0)` — deterministic, no probability weight.

**Expected future cost:** sum over all future nodes of `prob × price × (p[1,nid] + p[2,nid] + P_vent × v[nid])` — each node weighted by its probability (already includes chain rule from tree construction).

These are two lines of code but **one single objective** that Gurobi minimizes simultaneously.

---

### 3.6 Constraints

#### Here-and-Now Constraints (tau=0)

These replicate what the environment does when it overrides the policy's decisions. Adding them ensures the optimizer knows the true costs.

**Overrule controllers:**

- If low-temp overrule is active → `p0[r].fix(P_max)` (environment forces max heating)
- If temperature ≥ T_high → `p0[r].fix(0)` (environment forces zero heating)
- If humidity > H_high → `v0.fix(1)` (environment forces ventilation on)

**Ventilation startup detection:**

Three constraints force `s0 = 1` if and only if ventilation was OFF (v_prev=0) and is being turned ON (v0=1):

| v_prev | v0 | s0 |
|--------|----|----|
| 0 | 1 | **1** (startup) |
| 1 | 1 | 0 (was already on) |
| 0 | 0 | 0 (stays off) |
| 1 | 0 | 0 (turning off) |

**Forced ventilation:**

If `remaining_forced ≥ 1` → `v0.fix(1)` (can't turn off). If `remaining_forced ≥ 2` → also fix all tau=1 nodes to 1.

#### Future Nodes Constraints (tau≥1)

The loop iterates over every future node. For each node:

```python
t_parent = t_now + tau - 1
t_out = T_out[min(t_parent, len(T_out) - 1)]
```

`t_parent` is the hour of the parent node — the hour during which the transition to this node occurred. The outdoor temperature at that hour drives the thermal dynamics. The `min(...)` prevents array out-of-bounds access.

**Temperature dynamics (eq. 2):** Computes the temperature at this node as a function of the parent's state and decisions. Each term represents a physical effect: heat exchange between rooms, thermal loss to outside, heating from heater, cooling from ventilation, heat from occupants. This is an equality constraint — temperature is completely determined by physics.

**Low-temp overrule controller (eq. 8-16):** Hysteresis controller with three phases:

1. *Detection:* y_low=1 if temp < T_low; y_ok=1 if temp > T_ok (big-M constraints)
2. *Activation/memory:* u turns ON when temp drops below T_low, stays ON until temp exceeds T_ok (cannot activate from nothing — needs either previous activation or fresh trigger)
3. *Action:* when u=1, heating forced to P_max; when temp > T_ok, u forced to 0

**High-temp overrule controller (eq. 5-7):** Stateless — no memory. If temp ≥ T_high, heating forced to zero.

**Humidity dynamics (eq. 3):** Humidity at this node = parent humidity + occupancy contribution − ventilation removal. Occupancy from both rooms contributes because humidity is global.

**Humidity overrule (eq. 21):** Single constraint: `hum ≤ H_high + M × v`. If ventilation is off and humidity exceeds the threshold, the solver is forced to turn ventilation on.

**Ventilation inertia (eq. 17-20):**

*Startup detection:* Three constraints force s[nid]=1 if and only if ventilation was OFF at parent and ON at this node.

*Minimum uptime:* Walks backward through ancestor chain (up to min_up_time−1 steps). For each ancestor: if ventilation was started there (s=1), it must still be ON at this node (v=1). If the walk reaches the root (tau=0), uses `s0` instead of `s[ancestor_id]`.

---

### 3.7 Solve and Extract

```python
solver = SolverFactory('gurobi')
result = solver.solve(model)
```

Solves the MILP with Gurobi. Extracts only the here-and-now decisions:

- `p1, p2`: heating power for rooms 1 and 2
- `v`: ventilation on/off, thresholded at 0.5 to convert solver's floating-point output (e.g., 0.9999) to a clean integer

All future variable values are discarded — they exist only to make the here-and-now decision better informed about future consequences.

---

## 4. Entry Point — `select_action()`

```python
def select_action(state):
```

This is what the environment calls. It:

1. Computes H (reduced near end of day) and B
2. Builds the scenario tree
3. Solves the MILP
4. Returns decisions in the format the environment expects

The `try/except` is a safety net: if anything fails, returns zeros instead of crashing the entire 100-day simulation.

### Lookahead Horizon Logic

```python
H = min(3, 9 - state["current_time"])
```

- Hours 0–6: H=3 (full lookahead)
- Hour 7: H=2
- Hour 8: H=1
- Hour 9: H=0 → **edge case**, should be H=1 with B=1 to ensure constraints exist

---

## Key Concepts

### Rolling Horizon

The MILP is solved every hour. Only the here-and-now decision is implemented. Future decisions are discarded. Next hour, a completely new tree is built with updated state from the environment.

### Non-Anticipativity

Nodes sharing the same parent at an intermediate stage must make the same decision at that parent — this is implicit in the tree structure. Different branches from the same node represent different future scenarios, but the decision at the node itself is shared.

### Environment Override

The environment applies its own safety checks after receiving the policy's decision:

- Low-temp overrule active → P = P_max (regardless of policy)
- High-temp overrule → P = 0
- Humidity too high or minimum uptime not satisfied → V = 1

The here-and-now constraints in the model replicate this logic so the optimizer accounts for inevitable costs.